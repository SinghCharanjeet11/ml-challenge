"""Number-aware decision: accept a record's best candidate when the estimated test precision of its
cell (house number agrees / differs / missing / no address x probability band) is >= C.

Cell precision = holdout precision (corrected holdout) x pi / w, where w is the test/holdout
density ratio of the cell and pi that ratio in the clean cell (numbers agree, p >= 0.98).
The table comes from scripts/number_cells.py. C ~= 0.65 reproduces the leaderboard's behaviour
around a single 0.85 threshold.
    python cell_decision.py test_scored_cascade_anc.parquet 0.65 NAME
"""
import sys

import polars as pl

from io_utils import write_id_lists_chunked

W, OUT = "../../../work", "../../../output"
scores, C, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
BANDS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
LAB = ["<.3", ".3-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.85", ".85-.9", ".9-.95", ".95-.98", ">=.98"]
table = pl.read_parquet(f"{W}/number_cells.parquet").select("grp", pl.col("band").cast(pl.String),
                                                             (pl.col("prec_test_est") >= C).alias("accept"))
sc = pl.read_parquet(f"{W}/{scores}")
b = sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
op = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "entity_id", "num0", "ad", "country"])
s1 = pl.read_parquet(f"{W}/test_s1p.parquet", columns=["i1", "entity_id", "num0"])
b = b.join(op.select("io", "num0", "ad", "country"), on="io").join(s1.select("i1", pl.col("num0").alias("num0_1")), on="i1")
grp = (pl.when(pl.col("ad") == "").then(pl.lit("no_addr"))
         .when(pl.col("num0").is_null() | pl.col("num0_1").is_null()).then(pl.lit("num_missing"))
         .when(pl.col("num0") == pl.col("num0_1")).then(pl.lit("num_agree")).otherwise(pl.lit("num_differs")))
b = b.with_columns(grp.alias("grp"), pl.col("p").cut(BANDS, labels=LAB).cast(pl.String).alias("band")).join(
    table, on=["grp", "band"], how="left")
keep = b.filter(pl.col("accept").fill_null(False))
base = b.filter(pl.col("p") >= 0.85)
print("accepted cells:", table.filter("accept").sort("grp", "band").rows())
print(f"{keep.height:,} matches (single 0.85 threshold: {base.height:,}); "
      f"removed {base.join(keep, on=['io', 'i1'], how='anti').height:,}, added {keep.join(base, on=['io', 'i1'], how='anti').height:,}")
print("by country:", keep.group_by(pl.col("country").cast(pl.String)).len().sort("country").rows())
write_id_lists_chunked(f"{OUT}/matching_results_{name}.tsv", s1.select("i1", pl.col("entity_id").alias("s1_id")),
                       keep.select("io", "i1"), op.select("io", pl.col("entity_id").alias("o_id")), "matched_entity_ids")
