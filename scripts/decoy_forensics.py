"""What do the extra test records look like? Unlabelled test vs labelled train comparison.

Run on the box that has work/ (train/test prepared frames, cascade test scores, v2 train
candidates). Read-only.
"""
import sys

import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, ".")
from io_utils import gt_pairs, load_split  # noqa: E402

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_rows(40)


def frames(split):
    s1 = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1", "entity_id", "country", "nm", "ad", "core", "num0"])
    op = pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "entity_id", "country", "src", "nm", "ad", "core", "num0"])
    return s1.with_columns(pl.col("country").cast(pl.String)), op.with_columns(pl.col("country").cast(pl.String))


s1t, opt = frames("test")
s1r, opr = frames("train")
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
truth = gt_pairs(gt).join(s1r.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id").join(
    opr.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", "i1")

print("== missing address share of S2/S3 records")
for name, op in (("train", opr), ("test", opt)):
    print(name, op.group_by("country", "src").agg((pl.col("ad") == "").mean().alias("missing_addr"), pl.len())
          .sort("country", "src").rows())
print("train, matched vs unmatched records:",
      opr.join(truth, on="io", how="left").group_by(pl.col("i1").is_null().alias("unmatched")).agg(
          (pl.col("ad") == "").mean().alias("missing_addr"), pl.len()).rows())

# ---- exact duplicates inside S2/S3 (same name+address text) and their counts
for name, op in (("train", opr), ("test", opt)):
    d = op.filter(pl.col("ad") != "").group_by("country", "nm", "ad").len().filter(pl.col("len") > 1)
    print(f"== {name}: groups of identical (name, address) records: {d.height:,}, records in them {int(d['len'].sum()):,}")

# ---- the unmatched train records: how similar is each to its closest S1 (via v2 candidates)?
cand = pl.read_parquet(f"{W}/train_cand_v2.parquet", columns=["io", "i1", "block_score"])
top = cand.sort(["io", "block_score"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
unm = opr.join(truth, on="io", how="anti").select("io", "nm", "ad", "core", "num0", "country").join(top, on="io")
unm = unm.sample(min(200_000, unm.height), seed=1).join(
    s1r.select("i1", pl.col("core").alias("core_1"), pl.col("ad").alias("ad_1"), pl.col("num0").alias("num0_1")), on="i1")
unm = unm.with_columns(
    pl.Series("ns", process.cpdist(unm["core"].to_list(), unm["core_1"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)),
    pl.Series("as", process.cpdist(unm["ad"].to_list(), unm["ad_1"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
print("== train unmatched records vs their best S1 candidate: name>=90 & addr>=90 share",
      float(((unm["ns"] >= 90) & (unm["as"] >= 90)).mean()), " name>=90 share", float((unm["ns"] >= 90).mean()))

# ---- test: records the cascade gives probability in the grey zone, and their competitors
sc = pl.read_parquet(f"{W}/test_scored_cascade.parquet")
best = sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
print("== test best-candidate probability bands, by country")
b = best.join(opt.select("io", "country", "src"), on="io").with_columns(
    pl.col("p").cut([0.2, 0.5, 0.85, 0.95], labels=["<0.2", "0.2-0.5", "0.5-0.85", "0.85-0.95", ">=0.95"]).alias("band"))
print(b.group_by("country", "band").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
      .sort("country", "band"))
print("records with no candidate after the filter:", opt.height - best.height, f"of {opt.height:,}")

# grey-zone examples: record kept out (0.3<=p<0.85) while the same S1 has a confident record
conf = best.filter(pl.col("p") >= 0.85).select("i1", pl.col("io").alias("io_conf"), pl.col("p").alias("p_conf"))
grey = best.filter((pl.col("p") >= 0.3) & (pl.col("p") < 0.85)).join(conf, on="i1").unique("io").sample(40, seed=2)
ex = (grey.join(s1t.select("i1", pl.col("nm").alias("s1_name"), pl.col("ad").alias("s1_addr"), "country"), on="i1")
          .join(opt.select("io", pl.col("nm").alias("grey_name"), pl.col("ad").alias("grey_addr"), pl.col("src").alias("grey_src")), on="io")
          .join(opt.select(pl.col("io").alias("io_conf"), pl.col("nm").alias("conf_name"), pl.col("ad").alias("conf_addr"),
                           pl.col("src").alias("conf_src")), on="io_conf"))
for r in ex.iter_rows(named=True):
    print(f"\n[{r['country']}] S1   : {r['s1_name']} | {r['s1_addr']}")
    print(f"   kept p={r['p_conf']:.2f} S{r['conf_src']}: {r['conf_name']} | {r['conf_addr']}")
    print(f"   grey p={r['p']:.2f} S{r['grey_src']}: {r['grey_name']} | {r['grey_addr']}")
