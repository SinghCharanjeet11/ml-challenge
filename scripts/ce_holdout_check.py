"""Does the cross-encoder improve the cascade on the corrected holdout? Read-only."""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from decide import assign  # noqa: E402
from io_utils import gt_pairs, load_split  # noqa: E402
from metric import macro_f05  # noqa: E402

W = "../../../work"
s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
s1ids = s1.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
val = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)
ids = val["s1_id"].to_list()
truth = gt_pairs(gt).join(val, on="s1_id", how="semi")

h = pl.read_parquet(f"{W}/holdout_scored_cascade_anc.parquet").join(
    pl.read_parquet(f"{W}/ce_scores_holdout.parquet"), on=["io", "i1"], how="left")
h = h.with_columns(pl.col("ce").max().over("io").alias("ce_rec_max")).with_columns(
    (pl.col("ce") - pl.col("ce_rec_max")).alias("ce_gap"), pl.col("ce").is_not_null().alias("scored"))
hb = h.filter(pl.col("scored"))
y = hb["y"].to_numpy()
F = ["p", "ce", "ce_gap"]
X = hb.select(F).to_numpy()
print(f"band pairs {hb.height:,}  AUC cascade {roc_auc_score(y, hb['p'].to_numpy()):.4f}  "
      f"cross-encoder {roc_auc_score(y, hb['ce'].to_numpy()):.4f}")
fold = (hb["i1"].hash(seed=41) % 2).to_numpy()
oof = np.empty(len(y))
for k in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1),
                  lgb.Dataset(X[fold != k], y[fold != k]), 300)
    oof[fold == k] = m.predict(X[fold == k])
print(f"combined (out-of-fold) AUC {roc_auc_score(y, oof):.4f}")
hc = h.join(hb.select("io", "i1").with_columns(pl.Series("pc", oof)), on=["io", "i1"], how="left").with_columns(
    pl.coalesce("pc", "p").alias("pc"))


def f05(col, tau, dup):
    k = assign(hc.select("io", "i1", pl.col(col).alias("p"), "unowned"), tau)
    pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    if dup:
        extra = k.filter(pl.col("unowned")).join(s1ids, on="i1").join(oids, on="io").select("s1_id", pl.col("o_id") + "_dup")
        pred = pl.concat([pred, extra])
    return macro_f05(ids, pred, truth)


for tau in (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
    print(f"tau={tau}: holdout {f05('p', tau, False):.5f} -> {f05('pc', tau, False):.5f}   "
          f"test-density est {f05('p', tau, True):.5f} -> {f05('pc', tau, True):.5f}", flush=True)
