"""Matching files with per-country thresholds (and optional rejected cells) from saved cascade scores.

    python segment_threshold.py NAME US=0.85,India=0.85,France=0.95 [--reject_cells]

--reject_cells additionally drops matches in the regions whose estimated test precision is low
(from scripts/cell_shift.py): US house number differing only in the last digit, and India legal
form differing, when p < 0.95.
"""
import argparse

import polars as pl

from decide import assign
from io_utils import write_id_lists_chunked

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("taus")
ap.add_argument("--reject_cells", action="store_true")
args = ap.parse_args()
W, OUT = "../../../work", "../../../output"
taus = {k: float(v) for k, v in (x.split("=") for x in args.taus.split(","))}

sc = pl.read_parquet(f"{W}/test_scored_cascade.parquet")
best = assign(sc, 0.0)
op = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "entity_id", "country", "legal", "num0"])
s1 = pl.read_parquet(f"{W}/test_s1p.parquet", columns=["i1", "entity_id", "legal", "num0"])
best = best.join(op.select("io", pl.col("country").cast(pl.String), "legal", "num0"), on="io").join(
    s1.select("i1", pl.col("legal").alias("legal_1"), pl.col("num0").alias("num0_1")), on="i1")
tau = pl.col("country").replace_strict(taus, return_dtype=pl.Float64)
keep = best.filter(pl.col("p") >= tau)
if args.reject_cells:
    last_digit = ((pl.col("num0").str.len_chars() == pl.col("num0_1").str.len_chars()) & (pl.col("num0") != pl.col("num0_1"))
                  & (pl.col("num0").str.slice(0, pl.col("num0").str.len_chars() - 1)
                     == pl.col("num0_1").str.slice(0, pl.col("num0_1").str.len_chars() - 1))).fill_null(False)
    legal_diff = ((pl.col("legal") != "") & (pl.col("legal_1") != "") & (pl.col("legal") != pl.col("legal_1"))).fill_null(False)
    bad = (pl.col("p") < 0.95) & (((pl.col("country") == "US") & last_digit) | ((pl.col("country") == "India") & legal_diff))
    print("rejected by cells:", keep.filter(bad).group_by("country").len().rows())
    keep = keep.filter(~bad)
out = f"{OUT}/matching_results_{args.name}.tsv"
write_id_lists_chunked(out, s1.select("i1", pl.col("entity_id").alias("s1_id")), keep.select("io", "i1"),
                       op.select("io", pl.col("entity_id").alias("o_id")), "matched_entity_ids")
print(f"{out}: {keep.height:,} matches", keep.group_by("country").len().sort("country").rows())
