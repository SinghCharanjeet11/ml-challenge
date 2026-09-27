"""Number-aware decision cells: house number agrees / differs / missing x probability band.
Holdout precision (corrected, anchored run A) vs test density (same model on test), giving an
estimated test precision per cell. Read-only.
    python number_cells.py holdout_scored_cascade_anc.parquet test_scored_cascade_anc.parquet"""
import sys

import polars as pl

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(80)
BANDS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
LAB = ["<.3", ".3-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.85", ".85-.9", ".9-.95", ".95-.98", ">=.98"]


def cells(scored, split):
    b = scored.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
    op = pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "num0", "ad", "country"])
    s1 = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1", "num0"]).rename({"num0": "num0_1"})
    b = b.join(op, on="io").join(s1, on="i1")
    grp = (pl.when(pl.col("ad") == "").then(pl.lit("no_addr"))
             .when(pl.col("num0").is_null() | pl.col("num0_1").is_null()).then(pl.lit("num_missing"))
             .when(pl.col("num0") == pl.col("num0_1")).then(pl.lit("num_agree"))
             .otherwise(pl.lit("num_differs")))
    return b.with_columns(grp.alias("grp"), pl.col("p").cut(BANDS, labels=LAB).alias("band"),
                          pl.col("country").cast(pl.String))


h = cells(pl.read_parquet(f"{W}/{sys.argv[1]}"), "train")
t = cells(pl.read_parquet(f"{W}/{sys.argv[2]}"), "test")
hc = h.group_by("grp", "band").agg(pl.len().alias("n_h"), pl.col("y").mean().alias("prec_h"))
tc = t.group_by("grp", "band").agg(pl.len().alias("n_t"))
x = hc.join(tc, on=["grp", "band"], how="full", coalesce=True).fill_null(0).with_columns(
    (pl.col("n_h") / h.height).alias("sh_h"), (pl.col("n_t") / t.height).alias("sh_t"))
clean = x.filter((pl.col("grp") == "num_agree") & (pl.col("band") == ">=.98"))
pi = float((clean["sh_t"] / clean["sh_h"])[0])
x = x.with_columns((pl.col("sh_t") / pl.col("sh_h")).alias("w")).with_columns(
    (pl.col("prec_h") * pl.min_horizontal(pl.lit(1.0), pl.lit(pi) / pl.col("w"))).alias("prec_test_est"))
print(f"pi = {pi:.3f}")
print(x.sort("grp", "band").select("grp", "band", "n_h", "n_t", pl.col("prec_h").round(3), pl.col("w").round(2),
                                   pl.col("prec_test_est").round(3)))
print("\ntest records per group with p >= 0.85:", t.filter(pl.col("p") >= 0.85).group_by("grp").len().sort("grp").rows())
x.write_parquet(f"{W}/number_cells.parquet")
