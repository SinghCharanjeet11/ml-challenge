"""Twin hypothesis: is a false record a near-copy of ANOTHER record claiming the same S1?
For every record reaching the final model, compare it with the other records whose best
candidate is the same S1 (p >= 0.5). Check AUC inside the holdout's unsure band, and how test
looks. Read-only."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

W = "../../../work"


def twin_features(best, op):
    """best: (io, i1, p) best candidate per record. Returns per-record twin features."""
    rec = best.join(op, on="io")
    rec = rec.with_columns((pl.col("nm") + " | " + pl.col("ad")).alias("txt"))
    claim = rec.filter(pl.col("p") >= 0.5).select("i1", pl.col("io").alias("io2"), pl.col("txt").alias("txt2"),
                                                   pl.col("src").alias("src2"), pl.col("num0").alias("num02"))
    pairs = rec.join(claim, on="i1").filter(pl.col("io") != pl.col("io2"))
    a, b = pairs["txt"].to_list(), pairs["txt2"].to_list()
    sim = np.fromiter((fuzz.ratio(x, y) for x, y in zip(a, b)), float, len(a))
    ed = np.fromiter((Levenshtein.distance(x, y) for x, y in zip(a, b)), float, len(a))
    pairs = pairs.with_columns(pl.Series("sim", sim), pl.Series("ed", ed))
    agg = pairs.group_by("io").agg(pl.col("sim").max().alias("twin_sim"), pl.col("ed").min().alias("twin_edits"),
                                   (pl.col("src") == pl.col("src2")).any().alias("same_src_claim"),
                                   pl.len().alias("n_other_claims"))
    return rec.select("io", "i1", "p", "country").join(agg, on="io", how="left").with_columns(
        pl.col("twin_sim").fill_null(0), pl.col("twin_edits").fill_null(999), pl.col("n_other_claims").fill_null(0))


def best_of(sc):
    return sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)


cols = ["io", "nm", "ad", "src", "num0", "country"]
op_r = pl.read_parquet(f"{W}/train_op.parquet", columns=cols).with_columns(pl.col("country").cast(pl.String))
op_t = pl.read_parquet(f"{W}/test_op.parquet", columns=cols).with_columns(pl.col("country").cast(pl.String))
h = pl.read_parquet(f"{W}/holdout_scored_cascade.parquet")
hb = best_of(h.select("io", "i1", "p", "y"))
ht = twin_features(hb.select("io", "i1", "p"), op_r).join(hb.select("io", "y"), on="io")
mid = (pl.col("p") >= 0.2) & (pl.col("p") < 0.85)
u = ht.filter(mid)
y = u["y"].to_numpy()
print(f"holdout unsure records {u.height:,}, true share {y.mean():.3f}")
for c in ("twin_sim", "twin_edits", "n_other_claims"):
    x = u[c].to_numpy().astype(float)
    auc = roc_auc_score(y, x)
    print(f"  {c:16s} AUC {max(auc, 1 - auc):.3f}  mean true {x[y].mean():8.2f}  false {x[~y].mean():8.2f}")
print("  twin_sim >= 90 share: true", float(u.filter(pl.col('y'))['twin_sim'].ge(90).mean()),
      " false", float(u.filter(~pl.col('y'))['twin_sim'].ge(90).mean()))
tb = best_of(pl.read_parquet(f"{W}/test_scored_cascade.parquet"))
tt = twin_features(tb, op_t).filter(mid)
for c in ("US", "India", "France"):
    x = tt.filter(pl.col("country") == c)
    print(f"test {c} unsure: twin_sim mean {x['twin_sim'].mean():.2f}  share>=90 {float(x['twin_sim'].ge(90).mean()):.3f}  "
          f"n_other_claims {x['n_other_claims'].mean():.2f}")
hh = ht.filter(pl.col("p") >= 0.85)
print("holdout p>=0.85: twin_sim>=90 share true", float(hh.filter(pl.col('y'))['twin_sim'].ge(90).mean()),
      " false", float(hh.filter(~pl.col('y'))['twin_sim'].ge(90).mean()), " n_false", hh.filter(~pl.col('y')).height)
