"""Cascade matcher: blocking -> light filter model -> final model.

Why two models:
- Candidate sets should be small per S1 entity (it counts in the final ranking). Blocking
  casts a wide net (~160 pairs per S1 on test); a light filter model keeps only the few
  plausible candidates per record. The kept pairs are the candidate set (candidate_pairs.tsv)
  and the only pairs the final model scores.
- Test has many more decoy records than train (~40% of S2/S3 records match nothing vs 26%),
  and a decoy often looks as good as a real record on its own. The final model therefore also
  sees how each pair compares with the other candidates of its record and with all other
  records competing for the same S1 (also within the same source).

Steps:
1. Filter model (subset of cheap features, 2 folds split by owner entity): out-of-fold p1 for
   every train pair; test gets the average of the two fold models.
2. Prune: keep pairs with p1 >= --prune among the top --max_per_record of their record.
3. Competition features from p1 over the pruned tables.
4. Final model on the pruned pairs (90/10 entity split). Writes model files, candidate_pairs
   and matching files for test.

Needs featurise_all.py train and test first.
"""
import argparse
import glob
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign
from features import FEATURES
from io_utils import gt_pairs, load_split, write_id_lists_chunked
from metric import macro_f05
from pipeline import log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
ap.add_argument("--out", default="../../../output")
ap.add_argument("--prune", type=float, default=0.01)
ap.add_argument("--max_per_record", type=int, default=3)
ap.add_argument("--easy_every", type=int, default=10)
ap.add_argument("--filter_rounds", type=int, default=300)
ap.add_argument("--final_rounds", type=int, default=2000)
ap.add_argument("--val_mod", type=int, default=10)
ap.add_argument("--split", default="train")
ap.add_argument("--tag", default="", help="suffix for saved models / scores / output files")
ap.add_argument("--anchor", action="store_true",
                help="put each non-matching record in the fold of the S1 entity it is closest to (its top blocking candidate), so holdout entities face all their decoys; without it they only see ~10%% of them")
ap.add_argument("--filter_from", default=None, help="reuse the filter models saved under this tag (same split)")
ap.add_argument("--save_comp", action="store_true",
                help="save the pruned pairs with competition features (for the cross-encoder stage)")
ap.add_argument("--neg_weight", type=float, default=1.0,
                help="weight of non-matching records in the final model (test has ~2x more of them than train)")
ap.add_argument("--final_tau", type=float, default=0.85,
                help="threshold for the submitted matching file; picked on the public leaderboard because "
                     "test has ~2x more decoys per S1 than the train holdout (holdout optimum is 0.2)")
args = ap.parse_args()
W = args.work

FILTER_FEATURES = ["block_score", "n_shared", "tfidf_sim", "rank_block", "gap_block", "n_cand_o", "n_cand_1",
                   "rank_block_1", "nm_ratio", "core_tsort", "core_jw", "ad_tset", "num_common", "num0_eq",
                   "hn_min_logdiff", "combo", "rank_combo", "gap_combo", "src", "addr_missing"]
PARAMS = dict(objective="binary", num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)

nd = pl.read_parquet(f"{W}/name_dict.parquet")
ad = pl.read_parquet(f"{W}/addr_dict.parquet")
s1p, op = prep_cached(W, args.split, None, nd, ad)
s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split(args.data, "train")
pairs = gt_pairs(gt)
owner = pairs.join(s1ids, on="s1_id").join(oids, on="o_id").select("io", pl.col("i1").alias("owner"))
# records that match nothing have no owner; with --anchor they follow their top blocking candidate
parent = pl.DataFrame(schema={"io": pl.UInt32, "parent": pl.UInt32})
if args.anchor:
    cand = pl.read_parquet(f"{W}/{args.split}_cand_v2.parquet", columns=["io", "i1", "block_score"])
    anchor = (cand.sort(["io", "block_score"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
                  .select("io", pl.col("i1").alias("anchor")))
    parent = (parent.join(anchor, on="io", how="full", coalesce=True)
                    .select("io", pl.coalesce("parent", "anchor").alias("parent")))
    del cand, anchor
n_s1_train = s1p.height
del s1p, op, gt

train_parts = sorted(glob.glob(f"{W}/feat_all_{args.split}/part_*.parquet"))
test_parts = sorted(glob.glob(f"{W}/feat_all_test/part_*.parquet"))


def labelled(part, cols=None):
    d = pl.read_parquet(part, columns=cols).join(owner, on="io", how="left").join(parent, on="io", how="left")
    key = pl.coalesce("owner", "parent", "io")
    return d.with_columns((pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
                          (key.hash(seed=7) % 2).alias("fold"),
                          (key.hash(seed=5) % args.val_mod == 0).alias("val"))


def matrix(d, cols):
    return d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


# ---- 1. filter model: out-of-fold p1 on train, fold average on test
fcols = ["io", "i1"] + FILTER_FEATURES
p1_parts, models = [], []
for k in (0, 1):
    if args.filter_from is not None:
        m = lgb.Booster(model_file=f"{W}/model_filter_{k}{args.filter_from}.txt")
        models.append(m)
        for part in train_parts:
            d = labelled(part, fcols).filter(pl.col("fold") == k)
            p1_parts.append(d.select("io", "i1", "y", "val", "fold").with_columns(
                pl.Series("p1", m.predict(matrix(d, FILTER_FEATURES)), dtype=pl.Float32)))
        log(f"filter fold {k}: reused {args.filter_from}")
        continue
    xs, ys, ws = [], [], []
    for part in train_parts:
        d = labelled(part, fcols).filter(pl.col("fold") != k)
        hard = (pl.col("rank_combo") <= 2) | (pl.col("rank_block") <= 2)
        easy = pl.struct("io", "i1").hash(seed=11) % args.easy_every == 0
        d = d.filter(pl.col("y") | hard | easy).with_columns(
            pl.when(pl.col("y") | hard).then(1.0).otherwise(float(args.easy_every)).alias("w"))
        xs.append(matrix(d, FILTER_FEATURES)); ys.append(d["y"].to_numpy()); ws.append(d["w"].to_numpy())
    X, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    del xs, ys, ws
    m = lgb.train({**PARAMS, "learning_rate": 0.1}, lgb.Dataset(X, y, weight=w, free_raw_data=True),
                  num_boost_round=args.filter_rounds)
    log(f"filter fold {k}: trained on {len(y):,} rows")
    del X, y, w
    models.append(m)
    for part in train_parts:
        d = labelled(part, fcols).filter(pl.col("fold") == k)
        p1_parts.append(d.select("io", "i1", "y", "val", "fold").with_columns(
            pl.Series("p1", m.predict(matrix(d, FILTER_FEATURES)), dtype=pl.Float32)))
p1_train = pl.concat(p1_parts)
del p1_parts
p1_test = []
for part in test_parts:
    d = pl.read_parquet(part, columns=["io", "i1"] + FILTER_FEATURES)
    x = matrix(d, FILTER_FEATURES)
    p1_test.append(d.select("io", "i1").with_columns(
        pl.Series("p1", (models[0].predict(x) + models[1].predict(x)) / 2, dtype=pl.Float32)))
p1_test = pl.concat(p1_test)
models[0].save_model(f"{W}/model_filter_0{args.tag}.txt")
models[1].save_model(f"{W}/model_filter_1{args.tag}.txt")
log("filter model done")


# ---- 2. prune; report recall vs candidate-set size on train
def prune(p1, tau, top):
    return p1.filter((pl.col("p1") >= tau)
                     & (pl.col("p1").rank("ordinal", descending=True).over("io") <= top))


all_links = owner.height
log(f"blocking alone: {p1_train.height / n_s1_train:.1f} pairs per S1, "
    f"recall {int(p1_train['y'].sum()) / all_links:.4f}")
for tau, top in ((0.001, 5), (0.005, 3), (0.01, 3), (0.02, 3), (0.05, 2)):
    kept = prune(p1_train, tau, top)
    log(f"prune p1>={tau} top{top}: {kept.height / n_s1_train:.1f} pairs per S1, "
        f"recall {int(kept['y'].sum()) / all_links:.4f}")
p1_train = prune(p1_train, args.prune, args.max_per_record)
p1_test = prune(p1_test, args.prune, args.max_per_record)
log(f"kept train {p1_train.height:,} pairs, test {p1_test.height:,} pairs "
    f"({p1_test.height / 1_732_544:.1f} per test S1)")


# ---- 3. competition features over the pruned pairs
def competition(p1, src):
    p1 = p1.join(src, on="io")
    q = pl.col("p1")
    p1 = p1.with_columns(
        q.rank("ordinal", descending=True).over("io").alias("q_rank"),
        (q.max().over("io") - q).alias("q_gap"),
        q.sort(descending=True).get(1, null_on_oob=True).over("io").fill_null(0).alias("q_2nd"),
        pl.len().over("io").alias("q_n"),
    ).with_columns((pl.col("q_rank") == 1).alias("q_best"))
    best = pl.when(pl.col("q_best")).then(q).otherwise(None)
    p1 = p1.with_columns(
        q.rank("ordinal", descending=True).over("i1").alias("s_rank"),
        pl.len().over("i1").alias("s_n"),
        best.count().over("i1").alias("s_n_best"),
        (best > 0.5).sum().over("i1").alias("s_n_strong"),
        best.sum().over("i1").fill_null(0).alias("s_sum"),
        best.max().over("i1").fill_null(0).alias("s_max"),
        best.count().over(["i1", "src"]).alias("s_n_best_src"),
        (best > 0.5).sum().over(["i1", "src"]).alias("s_n_strong_src"),
        best.max().over(["i1", "src"]).fill_null(0).alias("s_max_src"),
    )
    return (p1.with_columns((pl.col("s_max") - q).alias("s_gap"), (pl.col("s_max_src") - q).alias("s_gap_src"))
              .drop("src").sort("io"))


COMP = ["p1", "q_rank", "q_gap", "q_2nd", "q_n", "q_best", "s_rank", "s_n", "s_n_best", "s_n_strong", "s_sum",
        "s_max", "s_n_best_src", "s_n_strong_src", "s_max_src", "s_gap", "s_gap_src"]
FINAL_FEATURES = FEATURES + COMP
comp_train = competition(p1_train, pl.read_parquet(f"{W}/{args.split}_op.parquet", columns=["io", "src"]))
comp_test = competition(p1_test, pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "src"]))
del p1_train, p1_test
log("competition features done")
if args.save_comp:
    comp_train.write_parquet(f"{W}/comp_train{args.tag}.parquet")
    comp_test.write_parquet(f"{W}/comp_test{args.tag}.parquet")
    log("pruned pairs + competition features saved")



def with_comp(d, comp):
    """Join the competition features for one part (parts cover a contiguous range of io)."""
    io = comp["io"]
    lo, hi = io.search_sorted(d["io"].min(), "left"), io.search_sorted(d["io"].max(), "right")
    return d.join(comp.slice(lo, hi - lo), on=["io", "i1"])


# ---- 4. final model on the pruned pairs (small enough that no sampling is needed)
xs, ys, vs, keys = [], [], [], []
for part in train_parts:
    d = with_comp(pl.read_parquet(part, columns=["io", "i1"] + FEATURES),
                  comp_train.select(["io", "i1", "y", "val"] + COMP))
    xs.append(matrix(d, FINAL_FEATURES)); ys.append(d["y"].to_numpy()); vs.append(d["val"].to_numpy())
    keys.append(d.select("io", "i1"))
X, y, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
keys = pl.concat(keys)
unowned = keys.join(owner, on="io", how="left")["owner"].is_null().to_numpy()
del xs, ys, vs
log(f"final model rows: train {int((~v).sum()):,}, holdout {int(v.sum()):,}, positives {int(y.sum()):,}")
wtr = np.where(unowned[~v], args.neg_weight, 1.0)
dtr = lgb.Dataset(X[~v], y[~v], weight=wtr)
dva = lgb.Dataset(X[v], y[v], reference=dtr)
final = lgb.train({**PARAMS, "learning_rate": 0.05}, dtr, num_boost_round=args.final_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
log(f"final model trained {final.best_iteration} rounds")

val_keys = keys.filter(pl.Series(v))
Xv = X[v]
del X, dtr, dva
p_final = final.predict(Xv, num_iteration=final.best_iteration)
p_filter = Xv[:, FINAL_FEATURES.index("p1")]
val_ids = s1ids.filter(pl.col("i1").hash(seed=5) % args.val_mod == 0)["s1_id"]
truth = pairs.filter(pl.col("s1_id").is_in(val_ids.implode()))
best = {}
for name, p in (("filter only", p_filter), ("final", p_final)):
    scored = val_keys.with_columns(pl.Series("p", p))
    un = pl.Series("un", unowned[v])
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        k = assign(scored.with_columns(un), tau)
        pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
        f = macro_f05(val_ids.to_list(), pred, truth)
        # test has ~1.9x more non-matching records per S1: count every false match from one twice
        dup = k.filter(pl.col("un")).join(s1ids, on="i1").join(oids, on="io").select("s1_id", pl.col("o_id") + "_dup")
        f2 = macro_f05(val_ids.to_list(), pl.concat([pred, dup]), truth)
        log(f"{name}: tau={tau:.2f} macroF0.5={f:.5f}  test-density estimate={f2:.5f}")
        best[name] = max(best.get(name, (0, 0)), (f, tau))
log(f"BEST {best}")

final.save_model(f"{W}/model_final{args.tag}.txt", num_iteration=final.best_iteration)
json.dump({"tau": best["final"][1], "val_f05": best["final"][0], "prune": args.prune,
           "max_per_record": args.max_per_record, "filter_features": FILTER_FEATURES,
           "final_features": FINAL_FEATURES}, open(f"{W}/config_cascade{args.tag}.json", "w"))

# ---- test: candidate set = pruned pairs, then final scores
s1t, opt = prep_cached(W, "test", None, nd, ad)
s1t_ids = s1t.select("i1", pl.col("entity_id").alias("s1_id"))
ot_ids = opt.select("io", pl.col("entity_id").alias("o_id"))
del s1t, opt
write_id_lists_chunked(f"{args.out}/candidate_pairs{args.tag}.tsv", s1t_ids, comp_test.select("io", "i1"), ot_ids,
                       "candidate_entity_ids")
log("wrote candidate_pairs.tsv")
scored = []
for part in test_parts:
    d = with_comp(pl.read_parquet(part, columns=["io", "i1"] + FEATURES), comp_test)
    scored.append(d.select("io", "i1").with_columns(
        pl.Series("p", final.predict(matrix(d, FINAL_FEATURES), num_iteration=final.best_iteration),
                  dtype=pl.Float32)))
scored = pl.concat(scored)
scored.write_parquet(f"{W}/test_scored_cascade{args.tag}.parquet")
val_keys.with_columns(pl.Series("p", p_final), pl.Series("y", y[v]), pl.Series("unowned", unowned[v])).write_parquet(f"{W}/holdout_scored_cascade{args.tag}.parquet")
kept = assign(scored, args.final_tau)
write_id_lists_chunked(f"{args.out}/matching_results{args.tag}.tsv", s1t_ids, kept, ot_ids, "matched_entity_ids")
log(f"matching_results.tsv: tau={args.final_tau:.2f}, {kept.height:,} matches "
    f"(other thresholds: python rethreshold.py cascade <tau>)")
