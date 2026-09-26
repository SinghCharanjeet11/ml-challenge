"""Train on all training entities.

The full feature table is ~20 GB, so features are written to work/feat_full/ chunk by chunk.
10% of entities are held out (all their pairs) for early stopping and the tau sweep. For the
other 90% we keep positives + top-2 hard negatives and 1 in `easy_every` easy negatives
with weight `easy_every`.
"""
import argparse
import glob
import json
import os
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign
from features import FEATURES, FEATURES_V1
from io_utils import gt_pairs, load_split
from metric import macro_f05
from pipeline import block_cached, featurise_chunks, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
ap.add_argument("--keep_k", type=int, default=10, help="0 = keep every candidate")
ap.add_argument("--blocking", default="v1", choices=["v1", "v2"])
ap.add_argument("--val_mod", type=int, default=10, help="1/val_mod of entities held out for validation")
ap.add_argument("--easy_every", type=int, default=10, help="keep 1 in N easy negatives (weight N)")
ap.add_argument("--rounds", type=int, default=2000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--skip_featurise", action="store_true", help="reuse work/feat_full/ from a previous run")
args = ap.parse_args()
feat_dir = f"{args.work}/feat_full" if args.blocking == "v1" else f"{args.work}/feat_full_{args.blocking}"
if args.blocking == "v1":
    FEATURES = FEATURES_V1

# Source 1 id tables are needed for scoring after the raw frames are freed.
nd = pl.read_parquet(f"{args.work}/name_dict.parquet")
ad = pl.read_parquet(f"{args.work}/addr_dict.parquet")
s1p, op = prep_cached(args.work, "train", None, nd, ad)
s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split(args.data, "train")
pairs = gt_pairs(gt)
del gt
owner = (pairs.join(s1ids, on="s1_id").join(oids, on="o_id")
              .select("io", pl.col("i1").alias("owner")))
val_s1 = s1ids.filter(pl.col("i1").hash(seed=5) % args.val_mod == 0)
log(f"train: s1={s1p.height:,} others={op.height:,} links={owner.height:,} val entities={val_s1.height:,}")

if not args.skip_featurise:
    cand = block_cached(args.work, "train", args.blocking)
    log(f"candidates: {cand.height:,}")
    shutil.rmtree(feat_dir, ignore_errors=True)
    os.makedirs(feat_dir)
    n_kept = 0
    for n, f in enumerate(featurise_chunks(cand, s1p, op, keep_k=args.keep_k)):
        f = f.join(owner, on="io", how="left").with_columns(
            (pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
            (pl.coalesce("owner", "io").hash(seed=5) % args.val_mod == 0).alias("val"))
        hard = (pl.col("rank_combo") <= 2) | (pl.col("rank_block") <= 2)
        keep_easy = (pl.struct("io", "i1").hash(seed=11) % args.easy_every == 0)
        f = (f.filter(pl.col("val") | pl.col("y") | hard | keep_easy)
              .with_columns(pl.when(pl.col("val") | pl.col("y") | hard).then(1.0)
                              .otherwise(float(args.easy_every)).cast(pl.Float32).alias("w"))
              .drop("owner"))
        f.write_parquet(f"{feat_dir}/part_{n:04d}.parquet")
        n_kept += f.height
        if n % 5 == 0:
            log(f"featurised chunk {n}: {n_kept:,} rows written")
    del cand
    log(f"featurised: {n_kept:,} rows written to {feat_dir}")
del s1p, op

parts = sorted(glob.glob(f"{feat_dir}/part_*.parquet"))
# Fill one preallocated float32 matrix part by part (collecting then converting would
# briefly hold two copies of the ~several-GB training matrix).
counts = [pl.scan_parquet(p).filter(~pl.col("val")).select(pl.len()).collect().item() for p in parts]
log(f"training matrix: {sum(counts):,} rows = {sum(counts) * len(FEATURES) * 4 / 1e9:.1f} GB")
X = np.empty((sum(counts), len(FEATURES)), dtype=np.float32)
y = np.empty(sum(counts), dtype=np.float32)
w = np.empty(sum(counts), dtype=np.float32)
at = 0
for p, c in zip(parts, counts):
    d = pl.read_parquet(p).filter(~pl.col("val"))
    X[at:at + c] = d.select(FEATURES).to_numpy()
    y[at:at + c] = d["y"].to_numpy()
    w[at:at + c] = d["w"].to_numpy()
    at += c
log(f"training rows {len(y):,} (positives {int(y.sum()):,}, weighted size {w.sum():,.0f})")
dtrain = lgb.Dataset(X, y, weight=w, free_raw_data=True)
dtrain.construct()
del X, y, w

va = pl.scan_parquet(parts).filter(pl.col("val")).select(["io", "i1", "y"] + FEATURES).collect()
Xv = va.select(FEATURES).to_numpy()
dval = lgb.Dataset(Xv, va["y"].to_numpy(), reference=dtrain)
log(f"validation rows {va.height:,} (positives {int(va['y'].sum()):,})")
va = va.select("io", "i1")

params = dict(objective="binary", learning_rate=args.lr, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
booster = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
log(f"trained {booster.best_iteration} rounds")

scored = va.select("io", "i1").with_columns(
    pl.Series("p", booster.predict(Xv, num_iteration=booster.best_iteration)))
del Xv, dval
val_ids = val_s1["s1_id"]
owned = pairs.filter(pl.col("s1_id").is_in(val_ids.implode()))
best = (0.0, 0.5)
for tau in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    pred = assign(scored, tau).join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    f = macro_f05(val_ids.to_list(), pred, owned)
    log(f"tau={tau:.2f} macroF0.5={f:.5f} pairs={pred.height:,}")
    best = max(best, (f, tau))
log(f"BEST tau={best[1]} F0.5={best[0]:.5f}")

booster.save_model(f"{args.work}/model.txt", num_iteration=booster.best_iteration)
json.dump({"tau": best[1], "keep_k": args.keep_k, "val_f05": best[0], "trained_on": "full",
           "blocking": args.blocking, "features": FEATURES},
          open(f"{args.work}/config.json", "w"))
imp = sorted(zip(booster.feature_importance("gain"), FEATURES), reverse=True)[:12]
log("saved model; top features: " + ", ".join(n for _, n in imp))
