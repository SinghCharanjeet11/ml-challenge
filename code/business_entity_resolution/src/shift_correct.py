"""Correct the cascade's probabilities for the train -> test distribution shift.

Test contains extra non-matching records concentrated in regions that are ambiguous in train
(US: same name, house number a few digits off; India: same name/address, different legal form;
France: same address, one business word swapped). A model trained on train gives them the
train odds of those regions, which are far too optimistic on test.

Label-free correction (covariate/label-shift style):
  test density t(x) = pi * h(x) + (1 - pi) * d(x)
    h = holdout (train-like) density of best-candidate pairs, d = extra decoy density.
  => P(match | x, test) = P(match | x, holdout) * pi / w(x),   w = t / h
w(x) comes from a domain classifier (test vs holdout, classes balanced, cross-fitted over the
target rows), pi from the density ratio in the clean region (p >= 0.98). Countries without a
labelled reference (France) are left unchanged.

  python shift_correct.py validate   # synthetic check: holdout + known decoys
  python shift_correct.py apply      # write test matching files from corrected scores
"""
import glob
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import assign
from features import FEATURES
from io_utils import gt_pairs, load_split, write_id_lists_chunked
from metric import macro_f05
from pipeline import log

W, OUT = "../../../work", "../../../output"
DOMAIN_FEATURES = FEATURES + ["p", "no_addr"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
MIN_P = 0.02          # records whose best candidate is below this are never matched; ignore them
N_REAL_TRAIN = 10_320_219


def best_pairs(scored):
    return scored.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)


def attach(best, parts):
    """Add the pair features of each (io, i1) from feature parts (contiguous io ranges)."""
    best = best.sort("io")
    io = best["io"]
    out = []
    for part in parts:
        d = pl.read_parquet(part, columns=["io", "i1"] + FEATURES)
        lo, hi = io.search_sorted(d["io"].min(), "left"), io.search_sorted(d["io"].max(), "right")
        if hi > lo:
            out.append(best.slice(lo, hi - lo).join(d, on=["io", "i1"]))
    return pl.concat(out)


def with_record_info(best, op_path):
    op = pl.read_parquet(op_path, columns=["io", "country", "ad"]).with_columns(
        pl.col("country").cast(pl.String), (pl.col("ad") == "").cast(pl.Int8).alias("no_addr")).drop("ad")
    return best.join(op, on="io")


def matrix(d):
    return d.select([pl.col(c).cast(pl.Float32) for c in DOMAIN_FEATURES]).to_numpy()


def density_ratio(ref, tgt, rounds=300):
    """w = t(x)/h(x) for every target row, from a class-balanced domain classifier.
    Target rows are cross-fitted (2 folds) so no row is scored by a model that saw it."""
    xr, xt = matrix(ref), matrix(tgt)
    fold = (tgt["io"].hash(seed=21) % 2).to_numpy()
    w = np.empty(len(xt))
    for k in (0, 1):
        tr = xt[fold != k]
        X = np.vstack([xr, tr])
        y = np.r_[np.zeros(len(xr)), np.ones(len(tr))]
        wt = np.r_[np.full(len(xr), len(tr) / len(xr)), np.ones(len(tr))]  # balance the classes
        m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=wt), num_boost_round=rounds)
        q = np.clip(m.predict(xt[fold == k]), 1e-4, 1 - 1e-4)
        w[fold == k] = q / (1 - q)
    return w


def corrected(ref, tgt):
    """tgt with p_adj = p * min(1, pi / w)."""
    w = density_ratio(ref, tgt)
    p = tgt["p"].to_numpy()
    clean = p >= 0.98
    pi = float(np.median(w[clean])) if clean.any() else 1.0
    adj = p * np.minimum(1.0, pi / w)
    log(f"  rows {len(p):,}  pi={pi:.3f}  share down-weighted >2x: {float((w > 2 * pi).mean()):.4f}")
    return tgt.with_columns(pl.Series("w", w), pl.Series("p_adj", adj))


def f05_at(best, col, tau, ids, s1ids, oids, truth):
    k = best.filter(pl.col(col) >= tau).join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    return macro_f05(ids, k, truth)


def validate():
    """Holdout of the synthetic-decoy split: reference = clean holdout entities of half A,
    target = holdout entities of half B with 6% of their synthetic decoys kept."""
    s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
    op = pl.read_parquet(f"{W}/train_aug2_op.parquet", columns=["io", "entity_id"])
    s1ids = s1.select("i1", pl.col("entity_id").alias("s1_id"))
    oids = op.select("io", pl.col("entity_id").alias("o_id"))
    parent = pl.read_parquet(f"{W}/train_aug2_parent.parquet")
    _, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
    pairs = gt_pairs(gt)
    owner = pairs.join(s1ids, on="s1_id").join(oids, on="o_id").select("io", pl.col("i1").alias("owner"))
    sc = pl.read_parquet(f"{W}/holdout_scored_cascade_train_aug2_oldmodel.parquet").select("io", "i1", "p", "y")
    sc = sc.filter((pl.col("io") < N_REAL_TRAIN) | ((pl.col("io").hash(seed=13) % 1000) < 60))
    best = with_record_info(best_pairs(sc), f"{W}/train_aug2_op.parquet")
    key = best.join(owner, on="io", how="left").join(parent, on="io", how="left").with_columns(
        (pl.coalesce("owner", "parent", "io").hash(seed=31) % 2).alias("half"))
    best = attach(key.filter(pl.col("p") >= MIN_P), sorted(glob.glob(f"{W}/feat_all_train_aug2/part_*.parquet")))
    ref = best.filter((pl.col("half") == 0) & (pl.col("io") < N_REAL_TRAIN))
    tgt = best.filter(pl.col("half") == 1)
    val_s1 = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0).with_columns((pl.col("i1").hash(seed=31) % 2).alias("half"))
    ids = val_s1.filter(pl.col("half") == 1)["s1_id"].to_list()
    truth = pairs.filter(pl.col("s1_id").is_in(ids))
    log(f"validate: reference {ref.height:,} clean rows, target {tgt.height:,} rows "
        f"({int((tgt['io'] >= N_REAL_TRAIN).sum()):,} synthetic decoys), {len(ids):,} target entities")
    parts = []
    for c in ("US", "India"):
        parts.append(corrected(ref.filter(pl.col("country") == c), tgt.filter(pl.col("country") == c)))
    tgt = pl.concat(parts)
    clean = tgt.filter(pl.col("io") < N_REAL_TRAIN)
    for tau in (0.2, 0.3, 0.5, 0.85):
        log(f"tau={tau}: raw={f05_at(tgt, 'p', tau, ids, s1ids, oids, truth):.5f}  "
            f"corrected={f05_at(tgt, 'p_adj', tau, ids, s1ids, oids, truth):.5f}  "
            f"clean(no decoys)={f05_at(clean, 'p', tau, ids, s1ids, oids, truth):.5f}")


def apply():
    """Real test: reference = train holdout, target = test US / India. France unchanged."""
    ref = with_record_info(pl.read_parquet(f"{W}/holdout_scored_cascade.parquet").select("io", "i1", "p"),
                           f"{W}/train_op.parquet")
    ref = attach(best_pairs(ref).filter(pl.col("p") >= MIN_P), sorted(glob.glob(f"{W}/feat_all_train/part_*.parquet")))
    tsc = pl.read_parquet(f"{W}/test_scored_cascade.parquet")
    tb = with_record_info(best_pairs(tsc), f"{W}/test_op.parquet")
    tgt = attach(tb.filter(pl.col("p") >= MIN_P), sorted(glob.glob(f"{W}/feat_all_test/part_*.parquet")))
    log(f"apply: reference {ref.height:,} holdout rows, target {tgt.height:,} test rows")
    parts = [tgt.filter(pl.col("country") == "France").with_columns(pl.lit(np.nan).alias("w"), pl.col("p").alias("p_adj"))]
    for c in ("US", "India"):
        parts.append(corrected(ref.filter(pl.col("country") == c), tgt.filter(pl.col("country") == c)))
    adj = pl.concat(parts).select("io", "i1", "p", "p_adj", "w", "country", "no_addr")
    adj.write_parquet(f"{W}/test_best_shift_corrected.parquet")
    s1 = pl.read_parquet(f"{W}/test_s1p.parquet", columns=["i1", "entity_id"]).rename({"entity_id": "s1_id"})
    op = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "entity_id"]).rename({"entity_id": "o_id"})
    fr = pl.col("country") == "France"
    for tau in (0.2, 0.3, 0.5):
        keep = adj.filter(pl.when(fr).then(pl.col("p") >= 0.85).otherwise(pl.col("p_adj") >= tau))
        write_id_lists_chunked(f"{OUT}/matching_results_shift_tau{tau:.2f}.tsv", s1, keep.select("io", "i1"), op,
                               "matched_entity_ids")
        per = keep.group_by("country").len().sort("country").rows()
        log(f"shift-corrected tau={tau}: {keep.height:,} matches {per}")
    base = adj.filter(pl.col("p") >= 0.85)
    log(f"current best (raw p>=0.85): {base.height:,} matches {base.group_by('country').len().sort('country').rows()}")


if __name__ == "__main__":
    validate() if sys.argv[1] == "validate" else apply()
