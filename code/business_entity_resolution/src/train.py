"""Train the matcher on the training split and measure out-of-fold macro F0.5.

1. Prepare and block the *full* training split (all Source 2/3 records), exactly as the test
   run will, so candidate-context features have the same distribution at train and test time.
2. Sample a fraction of Source 1 entities with all of their Source 2/3 records, plus the same
   fraction of unmatched records, and featurise their candidates.
3. 2-fold LightGBM (folds by owning entity) gives out-of-fold probabilities; sweep the
   threshold on exact macro F0.5 and print an error breakdown.
4. Refit on all sampled pairs and save the model + chosen threshold.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign
from features import FEATURES
from io_utils import gt_pairs, load_split
from metric import macro_f05
from pipeline import block_cached, featurise, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
ap.add_argument("--frac", type=float, default=0.15)
ap.add_argument("--keep_k", type=int, default=10)
ap.add_argument("--rounds", type=int, default=600)
args = ap.parse_args()
os.makedirs(args.work, exist_ok=True)

# Run `python block_split.py train` first: it learns the dictionaries and caches the
# prepared frames and candidates, which are loaded here.
nd = pl.read_parquet(f"{args.work}/name_dict.parquet")
ad = pl.read_parquet(f"{args.work}/addr_dict.parquet")
_, _, gt = load_split(args.data, "train")
pairs = gt_pairs(gt)
del gt
prep_cached(args.work, "train", lambda: load_split(args.data, "train")[:2], nd, ad)
cand = block_cached(args.work, "train")
s1p, op = prep_cached(args.work, "train", None, nd, ad)
log(f"prepared train: s1={s1p.height:,} others={op.height:,}")
log(f"candidates: {cand.height:,}")

# Sample entities (with all their records) and the same share of unmatched records.
rng = np.random.default_rng(0)
eval_s1 = s1p.filter(pl.Series(rng.random(s1p.height) < args.frac))["entity_id"]
owned = pairs.filter(pl.col("s1_id").is_in(eval_s1.implode()))
owner = (pairs.join(s1p.select(pl.col("entity_id").alias("s1_id"), pl.col("i1").alias("owner")), on="s1_id")
              .join(op.select(pl.col("entity_id").alias("o_id"), "io"), on="o_id").select("io", "owner"))
unowned = op.select("io").join(owner, on="io", how="anti")["io"]
unowned = unowned.filter(pl.Series(rng.random(unowned.len()) < args.frac))
q_owned = owner.join(s1p.filter(pl.col("entity_id").is_in(eval_s1.implode())).select(pl.col("i1").alias("owner")),
                     on="owner", how="semi")["io"]
queries = pl.concat([q_owned, unowned])
log(f"sample: {eval_s1.len():,} entities, {queries.len():,} queries")

feat = featurise(cand, s1p, op, keep_k=args.keep_k, queries=queries)
feat = feat.join(owner, on="io", how="left").with_columns(
    (pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
    (pl.coalesce("owner", "io").hash(seed=3) % 2).alias("fold"))
log(f"features: {feat.height:,} pairs, positives {int(feat['y'].sum()):,}")

params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
X = feat.select(FEATURES).to_numpy()
y = feat["y"].to_numpy()
fold = feat["fold"].to_numpy()
oof = np.zeros(feat.height, dtype=np.float32)
for k in (0, 1):
    tr, te = fold != k, fold == k
    oof[te] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=args.rounds).predict(X[te])
    log(f"fold {k} done")

scored = feat.select("io", "i1").with_columns(pl.Series("p", oof))
s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
best = (0.0, 0.5)
for tau in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    pred = assign(scored, tau).join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    f = macro_f05(eval_s1.to_list(), pred, owned)
    log(f"tau={tau:.2f} macroF0.5={f:.5f} pairs={pred.height:,}")
    best = max(best, (f, tau))

tau = best[1]
tl = owner.join(queries.to_frame("io"), on="io", how="semi").rename({"owner": "i1"})
kept = assign(scored, tau)
in_cand = tl.join(feat.select("io", "i1"), on=["io", "i1"], how="semi").height
top_ok = tl.join(assign(scored, 0.0).select("io", "i1"), on=["io", "i1"], how="semi").height
got = tl.join(kept.select("io", "i1"), on=["io", "i1"], how="semi").height
fp = kept.join(tl, on=["io", "i1"], how="anti")
fp_unowned = fp.join(tl.select("io"), on="io", how="anti").height
log(f"BEST tau={tau} F0.5={best[0]:.5f}")
log(f"true links={tl.height:,} | not in top-{args.keep_k}={tl.height - in_cand:,} | "
    f"another S1 ranked first={in_cand - top_ok:,} | first but below tau={top_ok - got:,} | matched={got:,}")
log(f"false matches={fp.height:,} (distractors {fp_unowned:,}, owned->wrong S1 {fp.height - fp_unowned:,})")

booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=args.rounds)
booster.save_model(f"{args.work}/model.txt")
json.dump({"tau": tau, "keep_k": args.keep_k, "val_f05": best[0]}, open(f"{args.work}/config.json", "w"))
imp = sorted(zip(booster.feature_importance("gain"), FEATURES), reverse=True)[:12]
log("saved model; top features: " + ", ".join(n for _, n in imp))
