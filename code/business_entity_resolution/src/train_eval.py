"""Validation run on training data: blocking -> features -> 2-fold LightGBM -> decision -> macro F0.5.

Evaluation set: a random fraction of Source 1 entities with *all* of their Source 2/3 records,
plus the same fraction of unmatched (distractor) Source 2/3 records. Every Source 1 record
stays in the index so candidates face their real competition.
"""
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign, expected_f05_prefix
from features import FEATURES
from io_utils import gt_pairs, load_split
from metric import macro_f05
from pipeline import block, featurise, learn_dictionaries, log, prep_records

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--countries", default="")  # comma list; empty = all labels in the data
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--keep_k", type=int, default=10)
ap.add_argument("--save", default="")
args = ap.parse_args()

nd, ad = learn_dictionaries(args.data)
s1, oth, gt = load_split(args.data, "train")
if args.countries:
    cs = args.countries.split(",")
    s1, oth = s1.filter(pl.col("country").is_in(cs)), oth.filter(pl.col("country").is_in(cs))
pairs = gt_pairs(gt)

rng = np.random.default_rng(0)
eval_s1 = s1.filter(pl.Series(rng.random(s1.height) < args.frac))["entity_id"]
owned = pairs.filter(pl.col("s1_id").is_in(eval_s1.implode()))
unowned = oth.join(pairs.select(pl.col("o_id").alias("entity_id")), on="entity_id", how="anti")
unowned = unowned.filter(pl.Series(rng.random(unowned.height) < args.frac))["entity_id"]
oth = oth.filter(pl.col("entity_id").is_in(pl.concat([owned["o_id"], unowned]).implode()))
log(f"eval S1 entities={eval_s1.len():,} queries={oth.height:,} (owned {owned.height:,})")

s1p, op = prep_records(s1, oth, nd, ad)
log("prepared")
cand = block(s1p, op)
log(f"blocked: {cand.height:,} candidates")
feat = featurise(cand, s1p, op, keep_k=args.keep_k)
log(f"features: {feat.height:,} pairs")

# Labels and folds (by owning entity, so an entity's records never straddle train/predict).
owner = (pairs.join(s1p.select(pl.col("entity_id").alias("s1_id"), pl.col("i1").alias("owner")), on="s1_id")
              .join(op.select(pl.col("entity_id").alias("o_id"), "io"), on="o_id").select("io", "owner"))
feat = feat.join(owner, on="io", how="left").with_columns(
    (pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
    (pl.coalesce("owner", "io").hash(seed=3) % 2).alias("fold"))
cov = owner.join(feat.filter("y").select("io"), on="io", how="semi").height / owner.height
log(f"recall ceiling after top-{args.keep_k}: {cov:.4f}")

params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              num_threads=0)
oof = np.zeros(feat.height, dtype=np.float32)
X = feat.select(FEATURES).to_numpy()
y = feat["y"].to_numpy()
fold = feat["fold"].to_numpy()
for k in (0, 1):
    tr, te = fold != k, fold == k
    booster = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
    oof[te] = booster.predict(X[te])
    log(f"fold {k}: trained on {tr.sum():,} pairs")
imp = sorted(zip(booster.feature_importance("gain"), FEATURES), reverse=True)[:15]
log("top features: " + ", ".join(f"{n}" for _, n in imp))

scored = feat.select("io", "i1").with_columns(pl.Series("p", oof))
ids = (s1p.select("i1", pl.col("entity_id").alias("s1_id")), op.select("io", pl.col("entity_id").alias("o_id")))
truth = owned
best = (0, None)
for tau in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    kept = assign(scored, tau)
    pred = kept.join(ids[0], on="i1").join(ids[1], on="io").select("s1_id", "o_id")
    f = macro_f05(eval_s1.to_list(), pred, truth)
    print(f"tau={tau:.2f} macroF0.5={f:.5f}  pairs={pred.height:,}", flush=True)
    best = max(best, (f, tau))

# Error breakdown at the best threshold.
tau = best[1]
kept = assign(scored, tau)
tl = owner.rename({"owner": "i1"})
in_cand = tl.join(feat.select("io", "i1"), on=["io", "i1"], how="semi")
best_any = assign(scored, 0.0)
top_ok = tl.join(best_any.select("io", "i1"), on=["io", "i1"], how="semi")
got = tl.join(kept.select("io", "i1"), on=["io", "i1"], how="semi")
fp = kept.join(tl, on=["io", "i1"], how="anti")
fp_unowned = fp.join(tl.select("io"), on="io", how="anti").height
print(f"\nbest tau={tau}: F0.5={best[0]:.5f}")
print(f"true links={tl.height:,}  not in candidates={tl.height - in_cand.height:,}  "
      f"in candidates but another S1 ranked first={in_cand.height - top_ok.height:,}  "
      f"ranked first but below tau={top_ok.height - got.height:,}  matched={got.height:,}")
print(f"false matches={fp.height:,} (distractor records {fp_unowned:,}, owned records sent to wrong S1 {fp.height - fp_unowned:,})")
single = s1p.filter(pl.col("entity_id").is_in(eval_s1.implode())).join(
    pairs.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="anti")
fp_single = fp.join(single.select("i1"), on="i1", how="semi").select("i1").unique().height
print(f"singletons in eval={single.height:,}  singletons wrongly given a match={fp_single:,}")

if args.save:
    feat.with_columns(pl.Series("p", oof)).write_parquet(args.save)
