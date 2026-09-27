"""Cell-level train->test shift table (US, India): where does test have excess records, and what
precision does that imply? Uses the saved best-pair profiles (records with an address).
cell = country x fingerprint x probability band."""
import polars as pl

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(200)

BANDS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95, 0.98]
LAB = ["<.05", ".05-.1", ".1-.2", ".2-.35", ".35-.5", ".5-.7", ".7-.85", ".85-.95", ".95-.98", ">=.98"]
both_num = (pl.col("n_num_o") > 0) & (pl.col("n_num_1") > 0)
FP = (pl.when(both_num & (pl.col("num0_eq") == 0) & ((pl.col("hn_near") > 0) | (pl.col("hn_min_logdiff") < 3.1)))
        .then(pl.lit("A_hn_shift"))
        .when(pl.col("legal_conflict") > 0).then(pl.lit("B_legal"))
        .when((pl.col("core_eq") == 0) & (pl.col("ad_tset") >= 90) & (pl.col("num0_eq") > 0)).then(pl.lit("C_name_diff_same_addr"))
        .otherwise(pl.lit("D_normal")).alias("fp"))

hold = pl.read_parquet(f"{W}/holdout_midband_profile.parquet").with_columns(FP, pl.col("p").cut(BANDS, labels=LAB).alias("band"))
test = pl.read_parquet(f"{W}/test_midband_profile.parquet").with_columns(FP, pl.col("p").cut(BANDS, labels=LAB).alias("band"))

rows = []
for c in ("US", "India"):
    h = hold.filter(pl.col("country") == c)
    t = test.filter(pl.col("country") == c)
    hc = h.group_by("fp", "band").agg(pl.len().alias("n_h"), pl.col("y").mean().alias("prec_h"))
    tc = t.group_by("fp", "band").agg(pl.len().alias("n_t"))
    x = hc.join(tc, on=["fp", "band"], how="full", coalesce=True).fill_null(0).with_columns(
        (pl.col("n_h") / h.height).alias("sh_h"), (pl.col("n_t") / t.height).alias("sh_t"))
    clean = x.filter((pl.col("fp") == "D_normal") & (pl.col("band") == ">=.98"))
    pi = float((clean["sh_t"] / clean["sh_h"])[0])
    x = x.with_columns((pl.col("sh_t") / pl.col("sh_h")).alias("w")).with_columns(
        (pl.col("prec_h") * pl.min_horizontal(pl.lit(1.0), pl.lit(pi) / pl.col("w"))).alias("prec_test_est"),
        pl.lit(c).alias("country"), pl.lit(pi).alias("pi"))
    rows.append(x)
tab = pl.concat(rows).sort("country", "fp", "band")
print(tab.select("country", "fp", "band", "n_h", "n_t", pl.col("prec_h").round(3), pl.col("w").round(2),
                 pl.col("pi").round(3), pl.col("prec_test_est").round(3)))
tab.write_parquet(f"{W}/cell_shift_table.parquet")
