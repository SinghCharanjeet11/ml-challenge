"""Final matcher with the cross-encoder scores as features.

Runs after `stage2.py --anchor --save_comp --tag _f` and `reranker.py score train|test`. Same
pruned pairs, split and weighting as stage2.py; on top of its features the model sees the
cross-encoder score and how it compares with the record's other candidates and with the other
records claiming the same S1 entity (the cross-encoder version of the competition features).
Train scores are out of fold (reranker.py cross-fits two models), so the holdout is honest.

Writes work/test_scored_RR.parquet (then `python rethreshold.py RR <tau>`) and prints the
threshold that gives the same number of matches as a reference submission.
"""
import argparse
import glob

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign
from features import FEATURES
from io_utils import gt_pairs, load_split
from metric import macro_f05
from pipeline import log

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
ap.add_argument("--tag", default="_f", help="tag of the stage2 run whose pruned pairs we use")
ap.add_argument("--rr_tag", default="")
ap.add_argument("--out_tag", default="RR")
ap.add_argument("--neg_weight", type=float, default=2.0)
ap.add_argument("--val_mod", type=int, default=10)
ap.add_argument("--final_rounds", type=int, default=3000)
ap.add_argument("--match_count", type=int, default=5_762_921,
                help="number of matches of the reference submission (cascade + small cross-encoder, 0.9808)")
args = ap.parse_args()
W = args.work
PARAMS = dict(objective="binary", num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
COMP = ["p1", "q_rank", "q_gap", "q_2nd", "q_n", "q_best", "s_rank", "s_n", "s_n_best", "s_n_strong", "s_sum",
        "s_max", "s_n_best_src", "s_n_strong_src", "s_max_src", "s_gap", "s_gap_src"]
CEF = ["ce", "ce_rank", "ce_gap", "ce_2nd", "ce_best", "ce_s_rank", "ce_s_gap", "ce_s_2nd", "ce_s_npos",
       "ce_s_npos_src", "ce_s_gap_src"]
FINAL_FEATURES = FEATURES + COMP + CEF


def ce_features(comp, split):
    d = comp.join(pl.read_parquet(f"{W}/rr_scores_{split}{args.rr_tag}.parquet"), on=["io", "i1"], how="left")
    d = d.join(pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "src"]), on="io")
    c = pl.col("ce").fill_null(-20.0)
    d = d.with_columns(
        c.rank("ordinal", descending=True).over("io").alias("ce_rank"),
        (c.max().over("io") - c).alias("ce_gap"),
        c.sort(descending=True).get(1, null_on_oob=True).over("io").fill_null(-20.0).alias("ce_2nd"),
    ).with_columns((pl.col("ce_rank") == 1).alias("ce_best"))
    best = pl.when(pl.col("ce_best")).then(c).otherwise(None)
    d = d.with_columns(
        c.rank("ordinal", descending=True).over("i1").alias("ce_s_rank"),
        (c.max().over("i1") - c).alias("ce_s_gap"),
        c.sort(descending=True).get(1, null_on_oob=True).over("i1").fill_null(-20.0).alias("ce_s_2nd"),
        (best > 0).sum().over("i1").alias("ce_s_npos"),
        (best > 0).sum().over(["i1", "src"]).alias("ce_s_npos_src"),
        (best.max().over(["i1", "src"]).fill_null(-20.0) - c).alias("ce_s_gap_src"),
    )
    miss = d["ce"].null_count()
    if miss:
        log(f"{split}: {miss:,} pairs without a cross-encoder score")
    return d.drop("src").sort("io")


def with_comp(d, comp):
    io = comp["io"]
    lo, hi = io.search_sorted(d["io"].min(), "left"), io.search_sorted(d["io"].max(), "right")
    return d.join(comp.slice(lo, hi - lo), on=["io", "i1"])


def matrix(d, cols):
    return d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


s1ids = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"]).rename({"entity_id": "s1_id"})
oids = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"]).rename({"entity_id": "o_id"})
_, _, gt = load_split(args.data, "train")
pairs = gt_pairs(gt)
owner = pairs.join(s1ids, on="s1_id").join(oids, on="o_id").select("io", pl.col("i1").alias("owner"))
del gt

comp_train = ce_features(pl.read_parquet(f"{W}/comp_train{args.tag}.parquet"), "train")
comp_test = ce_features(pl.read_parquet(f"{W}/comp_test{args.tag}.parquet"), "test")
log(f"pairs: train {comp_train.height:,}, test {comp_test.height:,}")

xs, ys, vs, keys = [], [], [], []
for part in sorted(glob.glob(f"{W}/feat_all_train/part_*.parquet")):
    d = with_comp(pl.read_parquet(part, columns=["io", "i1"] + FEATURES),
                  comp_train.select(["io", "i1", "y", "val"] + COMP + CEF))
    xs.append(matrix(d, FINAL_FEATURES)); ys.append(d["y"].to_numpy()); vs.append(d["val"].to_numpy())
    keys.append(d.select("io", "i1"))
X, y, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
keys = pl.concat(keys)
unowned = keys.join(owner, on="io", how="left")["owner"].is_null().to_numpy()
del xs, ys, vs
log(f"rows: train {int((~v).sum()):,}, holdout {int(v.sum()):,}, positives {int(y.sum()):,}")
dtr = lgb.Dataset(X[~v], y[~v], weight=np.where(unowned[~v], args.neg_weight, 1.0))
dva = lgb.Dataset(X[v], y[v], reference=dtr)
final = lgb.train({**PARAMS, "learning_rate": 0.05}, dtr, num_boost_round=args.final_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
log(f"trained {final.best_iteration} rounds")
imp = sorted(zip(final.feature_importance("gain"), FINAL_FEATURES), reverse=True)[:15]
log("top features: " + ", ".join(f"{n} {g / 1e6:.1f}M" for g, n in imp))

val_keys = keys.filter(pl.Series(v)).with_columns(pl.Series("y", y[v]), pl.Series("unowned", unowned[v]))
Xv = X[v]
del X, dtr, dva
val_keys = val_keys.with_columns(pl.Series("p", final.predict(Xv, num_iteration=final.best_iteration)))
val_ids = s1ids.filter(pl.col("i1").hash(seed=5) % args.val_mod == 0)["s1_id"]
truth = pairs.filter(pl.col("s1_id").is_in(val_ids.implode()))
for tau in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
    k = assign(val_keys.select("io", "i1", "p", pl.col("unowned").alias("un")), tau)
    pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    dup = k.filter(pl.col("un")).join(s1ids, on="i1").join(oids, on="io").select("s1_id", pl.col("o_id") + "_dup")
    log(f"tau={tau:.2f} macroF0.5={macro_f05(val_ids.to_list(), pred, truth):.5f}  "
        f"test-density estimate={macro_f05(val_ids.to_list(), pl.concat([pred, dup]), truth):.5f}")
final.save_model(f"{W}/model_final_{args.out_tag}.txt", num_iteration=final.best_iteration)
val_keys.write_parquet(f"{W}/holdout_scored_{args.out_tag}.parquet")

scored = []
for part in sorted(glob.glob(f"{W}/feat_all_test/part_*.parquet")):
    d = with_comp(pl.read_parquet(part, columns=["io", "i1"] + FEATURES), comp_test)
    scored.append(d.select("io", "i1").with_columns(
        pl.Series("p", final.predict(matrix(d, FINAL_FEATURES), num_iteration=final.best_iteration), dtype=pl.Float32)))
scored = pl.concat(scored)
scored.write_parquet(f"{W}/test_scored_{args.out_tag}.parquet")
b = scored.group_by("io").agg(pl.col("p").max())["p"].sort(descending=True)
log("test matches by threshold: " + " ".join(f"{t}:{int((b >= t).sum()):,}" for t in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95)))
log(f"threshold giving {args.match_count:,} matches: {float(b[args.match_count - 1]):.4f}")
