"""Which test records does model C accept that the old cascade rejected (and vice versa), at the
same number of matches? Compare their feature profiles. Read-only.
    python swap_analysis.py test_scored_C.parquet 0.936 test_scored_cascade.parquet 0.85"""
import sys

import polars as pl

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(40)
new_f, new_t, old_f, old_t = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])


def best(f, t):
    s = pl.read_parquet(f"{W}/{f}")
    b = s.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
    return b.filter(pl.col("p") >= t).select("io", "i1", "p")


n, o = best(new_f, new_t), best(old_f, old_t)
added = n.join(o, on=["io", "i1"], how="anti")
removed = o.join(n, on=["io", "i1"], how="anti")
kept = n.join(o, on=["io", "i1"], how="semi")
print(f"new {n.height:,}  old {o.height:,}  added {added.height:,}  removed {removed.height:,}  common {kept.height:,}")
op = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "country", "src", "nm", "ad", "core", "num0"])
s1 = pl.read_parquet(f"{W}/test_s1p.parquet", columns=["i1", "nm", "ad", "core", "num0"]).rename(
    {"nm": "nm_1", "ad": "ad_1", "core": "core_1", "num0": "num0_1"})
# twin similarity with the other accepted records of the same S1
acc = pl.concat([kept, added, removed]).unique(["io", "i1"]).join(op.select("io", "nm", "ad", "src"), on="io")
from rapidfuzz import fuzz  # noqa: E402

acc = acc.with_columns((pl.col("nm") + " | " + pl.col("ad")).alias("t"))
claims = kept.join(op.select("io", "nm", "ad", "src"), on="io").select(
    "i1", pl.col("io").alias("io2"), (pl.col("nm") + " | " + pl.col("ad")).alias("t2"), pl.col("src").alias("src2"))


def twin(d):
    x = d.join(op.select("io", "nm", "ad", "src"), on="io").with_columns((pl.col("nm") + " | " + pl.col("ad")).alias("t"))
    pr = x.select("io", "i1", "t", "src").join(claims, on="i1").filter(pl.col("io") != pl.col("io2"))
    sim = [fuzz.ratio(a, b) for a, b in zip(pr["t"].to_list(), pr["t2"].to_list())]
    pr = pr.with_columns(pl.Series("sim", sim))
    g = pr.group_by("io").agg(pl.col("sim").max().alias("twin_max"), pl.len().alias("n_claims"))
    return d.join(g, on="io", how="left").with_columns(pl.col("twin_max").fill_null(0), pl.col("n_claims").fill_null(0))


for name, d in (("ADDED by new model", added), ("REMOVED by new model", removed), ("common", kept.sample(200_000, seed=1))):
    t = twin(d).join(op.select("io", "country", "src", "core", "num0", "ad"), on="io").join(
        s1.select("i1", "core_1", "num0_1"), on="i1")
    print(f"\n== {name}: {t.height:,}")
    print(t.select(pl.col("p").mean().round(3).alias("p"), pl.col("twin_max").mean().round(1).alias("twin_max"),
                   (pl.col("twin_max") >= 90).mean().round(3).alias("twin>=90"), pl.col("n_claims").mean().round(2).alias("claims"),
                   (pl.col("core") == pl.col("core_1")).mean().round(3).alias("core_eq"),
                   (pl.col("num0") == pl.col("num0_1")).mean().round(3).alias("num0_eq"),
                   (pl.col("ad") == "").mean().round(3).alias("no_addr")))
    print(t.group_by("country").len().sort("country").rows())
ex = added.join(op.select("io", "nm", "ad", "country"), on="io").join(s1.select("i1", "nm_1", "ad_1"), on="i1").sample(15, seed=3)
print("\nexamples ADDED by the new model:")
for r in ex.iter_rows(named=True):
    print(f"[{r['country']} p={r['p']:.2f}] S1: {r['nm_1']} | {r['ad_1']}\n            rec: {r['nm']} | {r['ad']}")
