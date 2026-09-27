"""Independent evaluation + forensic diagnostics for the entity-resolution pipeline.

Does NOT change anything in the production pipeline. The metric here is written from the
challenge definition with plain Python sets and does not import the production metric.py.

Modes
  selftest                      unit tests of the metric (incl. the 0.714 worked example)
  score  --pred F --truth F     score any matching file against a ground-truth file
         [--entities F]         (optional list of Source 1 ids to restrict to)
  verify                        cross-check against the production metric on train ground
                                truth with random perturbations
  analyze                       full forensic analysis of the production model on its own
                                validation split (needs the pipeline's work/ caches and the v2
                                train candidates; run from code/business_entity_resolution/src)
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

# ----------------------------------------------------------------------------- metric


def entity_f05(g, p):
    """F0.5 for one Source 1 entity. g, p: sets of matched ids."""
    if not g:
        return 1.0 if not p else 0.0
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def evaluate(entities, pred, truth):
    """Full stats over `entities` (iterable of S1 ids). pred/truth: dict id -> set of ids.

    Macro precision is averaged over entities with a non-empty prediction, macro recall over
    entities with a non-empty truth (the challenge only defines the F0.5 average).
    """
    n = 0
    f_sum = 0.0
    p_sum = p_n = r_sum = r_n = 0
    tp_all = pred_all = true_all = 0
    true_empty = correct_empty = fp_empty = 0
    true_count_hist = defaultdict(int)
    for e in entities:
        g = truth.get(e, set())
        p = pred.get(e, set())
        n += 1
        f_sum += entity_f05(g, p)
        tp = len(g & p)
        tp_all += tp
        pred_all += len(p)
        true_all += len(g)
        if p:
            p_sum += tp / len(p)
            p_n += 1
        if g:
            r_sum += tp / len(g)
            r_n += 1
        else:
            true_empty += 1
            if p:
                fp_empty += 1
            else:
                correct_empty += 1
        true_count_hist[min(len(g), 3)] += 1
    return {
        "entities": n,
        "macro_f05": f_sum / n if n else float("nan"),
        "macro_precision": p_sum / p_n if p_n else float("nan"),
        "macro_recall": r_sum / r_n if r_n else float("nan"),
        "micro_precision": tp_all / pred_all if pred_all else float("nan"),
        "micro_recall": tp_all / true_all if true_all else float("nan"),
        "true_empty": true_empty,
        "correct_empty": correct_empty,
        "false_positive_empty": fp_empty,
        "true_1": true_count_hist[1],
        "true_2": true_count_hist[2],
        "true_3plus": true_count_hist[3],
        "avg_predicted": pred_all / n if n else float("nan"),
        "avg_true": true_all / n if n else float("nan"),
    }


def read_id_lists(path):
    """Read a two-column TSV (S1 id, comma-separated ids) with the header skipped."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, ids = line.partition("\t")
            out[s1] = {x for x in ids.split(",") if x}
    return out


def selftest():
    # Worked example from the problem statement.
    assert abs(entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}) - 0.714) < 1e-3
    assert entity_f05(set(), set()) == 1.0            # singleton, predicted empty
    assert entity_f05(set(), {"S2-1"}) == 0.0          # singleton, predicted something
    assert entity_f05({"S2-1"}, set()) == 0.0          # missed everything
    assert entity_f05({"S2-1"}, {"S2-2"}) == 0.0       # all wrong
    assert entity_f05({"S2-1"}, {"S2-1"}) == 1.0
    # precision 1, recall 0.5 -> 1.25*0.5/(0.25+0.5) = 0.8333
    assert abs(entity_f05({"a", "b"}, {"a"}) - 0.8333333) < 1e-6
    s = evaluate(["x", "y", "z"], {"x": {"a"}, "y": {"b"}}, {"x": {"a"}, "z": {"c"}})
    assert s["true_empty"] == 1 and s["false_positive_empty"] == 1 and abs(s["macro_f05"] - 1 / 3) < 1e-9
    print("selftest: all metric checks pass")


def verify(data_dir):
    """Our metric vs the production metric on perturbed copies of the train ground truth."""
    sys.path.insert(0, os.getcwd())
    import polars as pl
    from metric import macro_f05 as production_macro_f05

    truth = read_id_lists(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    ids = sorted(truth)
    rng = random.Random(0)
    all_o = [o for s in truth.values() for o in s]
    print(f"train ground truth: {len(ids):,} S1 entities, {len(all_o):,} links, "
          f"{sum(1 for s in truth.values() if not s):,} with no match")
    sample = rng.sample(ids, 200_000)
    cases = {"truth itself": {e: set(truth[e]) for e in sample},
             "all empty": {e: set() for e in sample}}
    noisy = {}
    for e in sample:
        p = {o for o in truth[e] if rng.random() > 0.2}
        if rng.random() < 0.15:
            p.add(rng.choice(all_o))
        noisy[e] = p
    cases["20% dropped + 15% random extra"] = noisy

    def to_pairs(d):
        rows = [(e, o) for e, s in d.items() for o in s]
        return pl.DataFrame(rows, schema={"s1_id": pl.String, "o_id": pl.String}, orient="row")

    truth_s = {e: truth[e] for e in sample}
    for name, pred in cases.items():
        ours = evaluate(sample, pred, truth_s)["macro_f05"]
        prod = production_macro_f05(sample, to_pairs(pred), to_pairs(truth_s))
        print(f"{name:32s} independent={ours:.6f} production={prod:.6f} diff={abs(ours - prod):.2e}")
        assert abs(ours - prod) < 1e-9, "metric mismatch"
    print("verify: production metric agrees with the independent implementation")


# ----------------------------------------------------------------------------- analysis

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FEATURE_GROUPS = {
    "name": ["nm_ratio", "nm_tset", "core_tsort", "core_tset", "core_partial", "core_jw", "concat_ratio",
             "concat_partial", "first_tok_eq", "core_eq", "len_core_o", "len_core_1", "extra_tok_o",
             "extra_tok_1", "extra_all_o", "extra_all_1", "legal_conflict", "legal_eq", "tfidf_sim"],
    "address": ["ad_tset", "ad_tsort", "ad_partial", "addr_missing"],
    "numeric/address-number": ["n_num_o", "n_num_1", "num_common", "num_jacc", "num0_eq", "hn0_logdiff",
                               "hn_min_logdiff", "hn_near", "hn0_prefix"],
    "blocking": ["block_score", "n_shared", "n_cand_1", "rank_block_1", "rank_block", "gap_block", "n_cand_o"],
    "cross-field / candidate context": ["combo", "rank_combo", "gap_combo", "gap_name", "gap_addr",
                                        "same_core_in_cands", "margin_2nd"],
    "record flags (source, script, web/alias)": ["src", "indic", "webname", "alias"],
}


def analyze(args):
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from rapidfuzz import fuzz, process
    from sklearn.metrics import average_precision_score

    sys.path.insert(0, os.getcwd())
    from decide import assign
    from io_utils import gt_pairs, load_split
    from pipeline import block_cached, featurise_chunks, prep_cached

    t0 = time.time()
    R = {}

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    W = args.work
    cfg = json.load(open(f"{W}/config.json"))
    booster = lgb.Booster(model_file=f"{W}/model.txt")
    feats = cfg["features"]
    tau = cfg["tau"]
    R["production_config"] = {k: cfg[k] for k in ("tau", "keep_k", "val_f05", "blocking") if k in cfg}
    R["model"] = {"type": "LightGBM binary classifier", "trees": booster.num_trees(), "features": len(feats)}

    nd = pl.read_parquet(f"{W}/name_dict.parquet")
    ad = pl.read_parquet(f"{W}/addr_dict.parquet")
    s1p, op = prep_cached(W, "train", None, nd, ad)
    cand = block_cached(W, "train", cfg.get("blocking", "v1"))
    log(f"loaded: s1={s1p.height:,} others={op.height:,} candidates={cand.height:,}")

    s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"), pl.col("country").cast(pl.String))
    oids = op.select("io", pl.col("entity_id").alias("o_id"), "src", "indic")
    _, _, gt = load_split(args.data, "train")
    truth_pairs = gt_pairs(gt).join(s1ids.select("i1", "s1_id"), on="s1_id").join(oids.select("io", "o_id"), on="o_id")
    owner = truth_pairs.select("io", pl.col("i1").alias("owner"))

    # ---- the production validation split, reproduced exactly (train_full.py)
    val_mask = pl.col("i1").hash(seed=5) % 10 == 0
    val_s1 = s1ids.filter(val_mask)
    qkey = pl.coalesce("owner", "io").hash(seed=5) % 10 == 0
    q = op.select("io").join(owner, on="io", how="left")
    val_io = q.filter(qkey)["io"]
    train_owned = q.filter(~qkey & pl.col("owner").is_not_null())
    leak = train_owned.join(val_s1.select(pl.col("i1").alias("owner")), on="owner", how="semi").height
    R["split"] = {
        "total_s1": s1p.height, "validation_s1": val_s1.height, "training_s1": s1p.height - val_s1.height,
        "strategy": "entity-level: S1 entity in validation iff hash(i1, seed=5) % 10 == 0; each S2/S3 "
                    "record follows its owner (unmatched records: hash(io, seed=5) % 10 == 0)",
        "seed": 5,
        "validation_queries": val_io.len(), "total_queries": op.height,
        "validation_owned_records_in_training": leak,
    }
    log(f"validation: {val_s1.height:,} S1 entities, {val_io.len():,} S2/S3 records, leaked={leak}")

    vt = truth_pairs.join(val_s1.select("i1"), on="i1", how="semi").join(oids.select("io", "src"), on="io")
    true_count = (val_s1.select("i1").join(vt.group_by("i1").len().rename({"len": "n_true"}), on="i1", how="left")
                  .with_columns(pl.col("n_true").fill_null(0)))

    # ---- 5. blocking recall
    in_cand = vt.join(cand.select("io", "i1"), on=["io", "i1"], how="semi").with_columns(pl.lit(True).alias("in_cand"))
    vt = vt.join(in_cand.select("io", "i1", "in_cand"), on=["io", "i1"], how="left").with_columns(
        pl.col("in_cand").fill_null(False)).join(val_s1.select("i1", "country"), on="i1")
    ent = vt.group_by("i1").agg(pl.col("in_cand").all().alias("all_in"), pl.len().alias("n_true"))
    per_s1 = cand.join(val_s1.select("i1"), on="i1", how="semi").group_by("i1").len()
    per_s1 = val_s1.select("i1", "country").join(per_s1, on="i1", how="left").with_columns(pl.col("len").fill_null(0))
    qs = per_s1["len"].to_numpy()
    blk = {
        "pair_recall": float(vt["in_cand"].mean()),
        "entity_all_match_recall": float(ent["all_in"].mean()),
        "candidates_per_s1": {"mean": float(qs.mean()), "median": float(np.median(qs)),
                              "p90": float(np.percentile(qs, 90)), "p95": float(np.percentile(qs, 95)),
                              "p99": float(np.percentile(qs, 99)), "max": int(qs.max())},
        "candidates_per_query_mean": cand.height / op.height,
        "by_country": {r["country"]: r["recall"] for r in vt.group_by("country").agg(
            pl.col("in_cand").mean().alias("recall")).iter_rows(named=True)},
        "by_source": {f"S{r['src']}": r["recall"] for r in vt.group_by("src").agg(
            pl.col("in_cand").mean().alias("recall")).iter_rows(named=True)},
    }
    bucket = pl.when(pl.col("n_true") >= 5).then(pl.lit("5+")).otherwise(pl.col("n_true").cast(pl.String))
    blk["by_true_count"] = {r["b"]: {"pair_recall": r["pr"], "all_match_recall": r["am"], "entities": r["n"]}
                            for r in vt.join(ent, on="i1").with_columns(bucket.alias("b")).group_by("b").agg(
                                pl.col("in_cand").mean().alias("pr"), pl.col("all_in").mean().alias("am"),
                                pl.col("i1").n_unique().alias("n")).iter_rows(named=True)}
    blk["by_true_count"]["0"] = {"pair_recall": None, "all_match_recall": None,
                                 "entities": int((true_count["n_true"] == 0).sum())}
    R["blocking"] = blk
    log(f"blocking pair recall {blk['pair_recall']:.4f}")

    # ---- score the validation pairs with the production model
    parts = []
    for f in featurise_chunks(cand, s1p, op, keep_k=cfg.get("keep_k", 10) or 0, queries=val_io):
        f = f.with_columns(pl.Series("p", booster.predict(f.select(feats).to_numpy()), dtype=pl.Float32))
        parts.append(f)
    sc = pl.concat(parts).join(owner, on="io", how="left").with_columns(
        (pl.col("i1") == pl.col("owner")).fill_null(False).alias("y")).join(oids.select("io", "indic").rename(
            {"indic": "indic_o"}), on="io")
    del parts, cand
    log(f"scored {sc.height:,} validation pairs")

    s1_name = dict(zip(val_s1["i1"].to_list(), val_s1["s1_id"].to_list()))
    o_name = dict(zip(oids["io"].to_list(), oids["o_id"].to_list()))
    val_entities = val_s1["s1_id"].to_list()
    truth = defaultdict(set)
    for i1, io in vt.select("i1", "io").iter_rows():
        truth[s1_name[i1]].add(o_name[io])

    def preds_from(kept):
        d = defaultdict(set)
        for i1, io in kept.select("i1", "io").iter_rows():
            if i1 in s1_name:
                d[s1_name[i1]].add(o_name[io])
        return d

    base_kept = assign(sc.select("io", "i1", "p"), tau)
    base_pred = preds_from(base_kept)
    base = evaluate(val_entities, base_pred, truth)
    R["baseline"] = base
    log(f"reproduced validation macro F0.5 = {base['macro_f05']:.5f} (production reported {cfg.get('val_f05')})")

    # ---- 6. matcher inside the candidate pool
    y = sc["y"].to_numpy()
    p = sc["p"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(len(y), 5_000_000), replace=False)
    grid = []
    best_by_io = sc.select("io", "i1", "p", "y").sort(["io", "p"], descending=[False, True]).group_by(
        "io", maintain_order=True).head(1)
    for t in THRESHOLDS:
        raw = p >= t
        tp = int((raw & y).sum())
        kept = best_by_io.filter(pl.col("p") >= t)
        tp_a = int(kept["y"].sum())
        m = evaluate(val_entities, preds_from(kept), truth)
        grid.append({"threshold": t, "raw_pair_precision": tp / max(int(raw.sum()), 1),
                     "raw_pair_recall": tp / max(int(y.sum()), 1),
                     "assigned_pair_precision": tp_a / max(kept.height, 1),
                     "assigned_pair_recall": tp_a / vt.height,
                     "macro_f05": m["macro_f05"], "avg_predicted": m["avg_predicted"]})
    R["matcher"] = {"pairs_scored": int(len(y)), "positives_in_pool": int(y.sum()),
                    "average_precision": float(average_precision_score(y[idx], p[idx])), "threshold_grid": grid}
    log("threshold grid done")

    # ---- 7. ranking quality (S1-centric and record-centric)
    ent_rank = (sc.join(val_s1.select("i1"), on="i1", how="semi")
                  .with_columns(pl.col("p").rank("ordinal", descending=True).over("i1").alias("r")))
    has_true = true_count.filter(pl.col("n_true") > 0)
    first_true = ent_rank.filter(pl.col("y")).group_by("i1").agg(pl.col("r").min().alias("first"),
                                                                 pl.col("r").max().alias("last"),
                                                                 pl.len().alias("n_true_in_pool"))
    rk = has_true.join(first_true, on="i1", how="left")
    rank = {f"top{k}_true_rate": float(((rk["first"] <= k).fill_null(False)).mean()) for k in (1, 3, 5, 10, 20)}
    rank["mrr"] = float((1.0 / rk["first"].cast(pl.Float64)).fill_null(0.0).mean())
    multi = rk.filter(pl.col("n_true") >= 2)
    for k in (3, 5, 10, 20):
        rank[f"multi_all_true_in_top{k}"] = float(
            ((pl.Series(multi["n_true_in_pool"] == multi["n_true"]).fill_null(False))
             & (multi["last"] <= k).fill_null(False)).mean())
    rec_rank = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("io").alias("r")).filter(pl.col("y"))
    rank["record_centric_true_s1_is_top1"] = float((rec_rank["r"] == 1).sum() / vt.height)
    rank["record_centric_true_s1_in_top3"] = float((rec_rank["r"] <= 3).sum() / vt.height)
    R["ranking"] = rank
    log("ranking done")

    # ---- per-pair view of errors
    pred_set = base_kept.select("io", "i1").with_columns(pl.lit(True).alias("pred"))
    sc = sc.join(pred_set, on=["io", "i1"], how="left").with_columns(pl.col("pred").fill_null(False))
    core_freq = s1p.group_by(["country", "core"]).len().rename({"len": "core_freq"})
    s1_ctx = s1p.select("i1", "country", "core").join(core_freq, on=["country", "core"]).select(
        "i1", pl.col("country").cast(pl.String), "core_freq")
    sc = sc.join(s1_ctx, on="i1", how="left")

    def pattern(df):
        both_nums = (pl.col("n_num_o") > 0) & (pl.col("n_num_1") > 0)
        return df.with_columns(
            pl.when(pl.col("addr_missing") > 0).then(pl.lit("missing address"))
            .when((pl.col("ad_tset") >= 90) & (pl.col("core_tsort") < 60)).then(pl.lit("same address / different business"))
            .when((pl.col("core_tsort") >= 85) & (pl.col("ad_tset") >= 70) & both_nums & (pl.col("num0_eq") == 0))
            .then(pl.lit("same name+street, different house number"))
            .when(pl.col("legal_conflict") > 0).then(pl.lit("legal-form confusion"))
            .when((pl.col("core_tsort") >= 70) & ((pl.col("extra_tok_o") > 0) | (pl.col("extra_tok_1") > 0)))
            .then(pl.lit("extra/missing word in name (branch or franchise variant)"))
            .when(pl.col("core_freq") >= 3).then(pl.lit("common business name"))
            .when((pl.col("core_tsort") >= 85) & (pl.col("ad_tset") < 60)).then(pl.lit("similar name / different address"))
            .when(pl.col("indic_o") > 0).then(pl.lit("transliteration"))
            .when((pl.col("webname") > 0) | (pl.col("alias") > 0)).then(pl.lit("website / dba alias name"))
            .when((pl.col("core_tsort") >= 90) & (pl.col("ad_tset") >= 90)).then(pl.lit("near-identical record (duplicate/decoy)"))
            .when(pl.col("core_tsort") < 60).then(pl.lit("name variation (abbreviation / typo)"))
            .otherwise(pl.lit("other")).alias("pattern"))

    def describe(df, n_total):
        cols = ["core_tsort", "nm_ratio", "ad_tset", "num0_eq", "legal_eq", "core_jw", "n_shared", "tfidf_sim", "p"]
        return {"count": n_total,
                "medians": {c: float(df[c].median()) for c in cols},
                "patterns": {r["pattern"]: round(r["len"] / df.height, 4) for r in
                             df.group_by("pattern").len().sort("len", descending=True).iter_rows(named=True)}}

    # ---- 8. false positives
    fp = pattern(sc.filter(pl.col("pred") & ~pl.col("y")))
    fp_info = describe(fp, fp.height)
    fp_info["from_unmatched_records"] = float(fp["owner"].is_null().mean())
    fp_info["owned_record_to_wrong_s1"] = float(fp["owner"].is_not_null().mean())
    fp_info["country_agreement"] = "always (blocking is per country)"
    fp_info["by_source"] = {f"S{r['src']}": round(r["len"] / fp.height, 4)
                            for r in fp.group_by(pl.col("src").cast(pl.Int8)).len().iter_rows(named=True)}
    fp_info["score_bands"] = {b: float(((fp["p"] >= lo) & (fp["p"] < hi)).mean())
                              for b, lo, hi in (("0.2-0.3", 0.2, 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-0.8", 0.5, 0.8),
                                                ("0.8-1.0", 0.8, 1.01))}
    R["false_positives"] = fp_info
    log(f"false positives: {fp.height:,}")

    # ---- 9. false negatives (every true pair of a validation entity that was not predicted)
    vt2 = vt.join(pred_set, on=["io", "i1"], how="left").with_columns(pl.col("pred").fill_null(False))
    fn_all = vt2.filter(~pl.col("pred"))
    scored_true = sc.filter(pl.col("y")).select("io", "i1", "p")
    top = sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1).select(
        "io", pl.col("i1").alias("top_i1"), pl.col("p").alias("top_p"))
    fn = fn_all.join(scored_true, on=["io", "i1"], how="left").join(top, on="io", how="left").with_columns(
        pl.when(~pl.col("in_cand")).then(pl.lit("never generated as candidate"))
        .when(pl.col("top_i1") != pl.col("i1")).then(pl.lit("candidate, but another S1 scored higher (one-owner rule)"))
        .when(pl.col("p") < 0.05).then(pl.lit("candidate ranked first, low model score (<0.05)"))
        .otherwise(pl.lit("candidate ranked first, score just below threshold")).alias("stage"))
    fn_info = {"count": fn.height, "share_of_true_pairs": fn.height / vt.height,
               "stages": {r["stage"]: round(r["len"] / fn.height, 4)
                          for r in fn.group_by("stage").len().sort("len", descending=True).iter_rows(named=True)},
               "by_source": {f"S{r['src']}": round(r["len"] / fn.height, 4)
                             for r in fn.group_by("src").len().iter_rows(named=True)},
               "by_country": {r["country"]: round(r["len"] / fn.height, 4)
                              for r in fn.group_by("country").len().iter_rows(named=True)}}
    fn_pool = pattern(fn.filter(pl.col("in_cand")).select("io", "i1", "stage").join(sc, on=["io", "i1"]))
    fn_info["in_pool_patterns"] = describe(fn_pool, fn_pool.height)["patterns"]
    # blocking misses: similarity on the prepared text directly
    miss = fn.filter(~pl.col("in_cand")).select("io", "i1").join(
        s1p.select("i1", "core", "ad"), on="i1").join(op.select("io", "core", "ad", "indic"), on="io", suffix="_o")
    if miss.height:
        ns = process.cpdist(miss["core"].to_list(), miss["core_o"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)
        as_ = process.cpdist(miss["ad"].to_list(), miss["ad_o"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        m2 = miss.with_columns(pl.Series("ns", ns), pl.Series("as", as_)).with_columns(
            pl.when(pl.col("ad_o") == "").then(pl.lit("missing address"))
            .when(pl.col("indic")).then(pl.lit("transliteration (Indic script)"))
            .when((pl.col("ns") < 60) & (pl.col("as") < 60)).then(pl.lit("name and address both very different"))
            .when(pl.col("ns") < 60).then(pl.lit("name variation (abbreviation / typo / alias)"))
            .when(pl.col("as") < 60).then(pl.lit("address variation"))
            .otherwise(pl.lit("similar text but crowded out of top-k")).alias("why"))
        fn_info["blocking_miss_patterns"] = {r["why"]: round(r["len"] / m2.height, 4) for r in
                                             m2.group_by("why").len().sort("len", descending=True).iter_rows(named=True)}
    R["false_negatives"] = fn_info
    log(f"false negatives: {fn.height:,}")

    # ---- 10. multi-match analysis
    per_ent = []
    for e in val_entities:
        g, pr = truth.get(e, set()), base_pred.get(e, set())
        tp = len(g & pr)
        per_ent.append((e, len(g), len(pr), tp, entity_f05(g, pr)))
    pe = pl.DataFrame(per_ent, schema=["s1_id", "n_true", "n_pred", "tp", "f05"], orient="row")
    pe = pe.join(val_s1.select("s1_id", "country"), on="s1_id")
    b = pl.when(pl.col("n_true") >= 5).then(pl.lit("5+")).otherwise(pl.col("n_true").cast(pl.String)).alias("bucket")
    mm = {}
    for r in pe.with_columns(b).group_by("bucket").agg(
            pl.len().alias("entities"), pl.col("f05").mean().alias("macro_f05"),
            (pl.col("tp").sum() / pl.col("n_pred").sum()).alias("precision"),
            (pl.col("tp").sum() / pl.col("n_true").sum()).alias("recall"),
            pl.col("n_pred").mean().alias("avg_pred"), pl.col("n_true").mean().alias("avg_true"),
            ((pl.col("tp") == pl.col("n_true")) & (pl.col("n_true") > 0)).mean().alias("all_match_recall"),
            (pl.col("n_pred") == 0).mean().alias("predicted_empty")).sort("bucket").iter_rows(named=True):
        mm[r.pop("bucket")] = r
    R["multi_match"] = mm
    # share of total F0.5 loss by bucket
    loss = pe.with_columns(b, (1 - pl.col("f05")).alias("loss")).group_by("bucket").agg(pl.col("loss").sum())
    tot = float(loss["loss"].sum())
    R["loss_share_by_true_count"] = {r["bucket"]: round(r["loss"] / tot, 4) for r in loss.iter_rows(named=True)}

    # ---- 11/12. source and country
    def f05_micro(pp, rr):
        return 1.25 * pp * rr / (0.25 * pp + rr) if pp + rr else 0.0

    def oracle(fix):
        """macro F0.5 when the errors selected by fix(o_id, s1_id) are corrected."""
        pr = {}
        for e in val_entities:
            g, p_ = truth.get(e, set()), base_pred.get(e, set())
            p2 = {o for o in p_ if o in g or not fix(o, e)} | {o for o in g if fix(o, e)}
            pr[e] = p2
        return evaluate(val_entities, pr, truth)["macro_f05"]

    src_of = dict(zip(oids["o_id"].to_list(), oids["src"].to_list()))
    country_of = dict(zip(val_s1["s1_id"].to_list(), val_s1["country"].to_list()))
    by_src = {}
    for s in (2, 3):
        t_s = vt2.filter(pl.col("src") == s)
        pred_s = sc.filter(pl.col("pred") & (pl.col("src") == s))
        pp = float(pred_s["y"].mean()) if pred_s.height else float("nan")
        rr = float(t_s["pred"].mean())
        by_src[f"S{s}"] = {"candidate_recall": float(t_s["in_cand"].mean()), "precision": pp, "recall": rr,
                           "f05_pairs": f05_micro(pp, rr), "false_positive_rate": 1 - pp, "false_negative_rate": 1 - rr,
                           "macro_f05_if_this_source_were_perfect": oracle(lambda o, e, s=s: src_of[o] == s)}
    R["by_source"] = by_src
    by_c = {}
    for c in sorted(pe["country"].unique().to_list()):
        sub = pe.filter(pl.col("country") == c)
        ents = sub["s1_id"].to_list()
        m = evaluate(ents, base_pred, truth)
        pc = per_s1.filter(pl.col("country") == c)["len"]
        by_c[c] = {"entities": len(ents), "candidate_recall": blk["by_country"].get(c), "macro_f05": m["macro_f05"],
                   "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
                   "empty_prediction_accuracy": m["correct_empty"] / max(m["true_empty"], 1),
                   "avg_candidates_per_s1": float(pc.mean()), "avg_predictions": m["avg_predicted"],
                   "macro_f05_if_this_country_were_perfect": oracle(lambda o, e, c=c: country_of[e] == c)}
    R["by_country"] = by_c
    log("source/country done")

    # ---- 14 (evidence). oracle decomposition of the remaining loss
    pool_true = defaultdict(set)
    for i1, io in sc.filter(pl.col("y")).select("i1", "io").iter_rows():
        if i1 in s1_name:
            pool_true[s1_name[i1]].add(o_name[io])
    fp_owner = {o_name[io]: ow for io, ow in fp.select("io", "owner").iter_rows()}
    dec = {
        "baseline": base["macro_f05"],
        "perfect_decisions_on_current_candidates": evaluate(val_entities, pool_true, truth)["macro_f05"],
        "remove_all_false_positives": oracle(lambda o, e: o not in truth.get(e, set())),
        "remove_false_positives_from_unmatched_records": oracle(
            lambda o, e: o not in truth.get(e, set()) and o in fp_owner and fp_owner[o] is None),
        "add_all_missed_true_pairs_in_pool": oracle(lambda o, e: o in truth.get(e, set()) and o in pool_true.get(e, set())),
        "add_missed_true_pairs_outside_pool (blocking)": oracle(
            lambda o, e: o in truth.get(e, set()) and o not in pool_true.get(e, set())),
    }
    R["oracle_decomposition"] = dec
    log("oracle decomposition done")

    # ---- 13. feature importance + permutation ablation on a subset of validation entities
    gain = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
    split = dict(zip(booster.feature_name(), booster.feature_importance("split")))
    names = booster.feature_name()
    tot_gain = sum(gain.values())
    R["feature_importance"] = sorted(
        [{"feature": feats[int(n.split("_")[1])] if n.startswith("Column_") else n,
          "gain_share": gain[n] / tot_gain, "splits": int(split[n])} for n in names],
        key=lambda r: -r["gain_share"])
    sub_s1 = val_s1.filter(pl.col("i1").hash(seed=9) % 4 == 0)
    sub = sc.filter(pl.col("owner").is_in(sub_s1["i1"].implode())
                    | (pl.col("owner").is_null() & (pl.col("io").hash(seed=9) % 4 == 0)))
    sub_ents = sub_s1["s1_id"].to_list()
    Xs = sub.select(feats).to_numpy()
    keys_sub = sub.select("io", "i1")

    def sub_f05(pvals):
        return evaluate(sub_ents, preds_from(assign(keys_sub.with_columns(pl.Series("p", pvals)), tau)), truth)["macro_f05"]

    base_sub = sub_f05(booster.predict(Xs))
    abl = {"subset_entities": len(sub_ents), "baseline_macro_f05": base_sub}
    for g, cols in FEATURE_GROUPS.items():
        Xp = Xs.copy()
        perm = rng.permutation(len(Xp))
        for c in cols:
            if c in feats:
                j = feats.index(c)
                Xp[:, j] = Xs[perm, j]
        abl[g] = {"macro_f05_when_shuffled": sub_f05(booster.predict(Xp))}
        abl[g]["drop"] = base_sub - abl[g]["macro_f05_when_shuffled"]
        log(f"ablation {g}: drop {abl[g]['drop']:.4f}")
    abl["country/state"] = "country is deliberately not a model feature; no state/postal features exist"
    R["ablation_permutation"] = abl

    # ---- test-vs-train distribution (no labels involved)
    s1t, opt = prep_cached(W, "test", None, nd, ad)
    dist = {}
    for name, a, b_ in (("train", s1p, op), ("test", s1t, opt)):
        dist[name] = {r["country"]: {"s1": r["s1"], "records": r["rec"], "records_per_s1": r["rec"] / r["s1"]}
                      for r in a.group_by(pl.col("country").cast(pl.String)).len().rename({"len": "s1"}).join(
                          b_.group_by(pl.col("country").cast(pl.String)).len().rename({"len": "rec"}),
                          on="country").iter_rows(named=True)}
    dist["train_links_per_s1"] = truth_pairs.height / s1p.height
    dist["train_share_of_records_matched"] = truth_pairs.height / op.height
    R["distribution"] = dist

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, "w"), indent=1, default=str)
    log(f"wrote {args.out}")

    g02 = next(r for r in grid if abs(r["threshold"] - tau) < 1e-9)
    print("\nCURRENT_LOCAL_F05 =", round(base["macro_f05"], 5))
    print("VALIDATION_ENTITIES =", len(val_entities))
    print("CANDIDATE_PAIR_RECALL =", round(blk["pair_recall"], 5))
    print("ALL_MATCH_RECALL =", round(blk["entity_all_match_recall"], 5))
    for k in (1, 5, 10):
        print(f"TOP{k}_TRUE_RATE =", round(rank[f"top{k}_true_rate"], 5))
    print("PRECISION =", round(base["micro_precision"], 5), "(micro)")
    print("RECALL =", round(base["micro_recall"], 5), "(micro)")
    print("F05 =", round(base["macro_f05"], 5), f"(assigned-pair F0.5 at tau={tau}: "
          f"{f05_micro(g02['assigned_pair_precision'], g02['assigned_pair_recall']):.5f})")
    print("SOURCE2_F05 =", round(by_src["S2"]["f05_pairs"], 5), "(pair level)")
    print("SOURCE3_F05 =", round(by_src["S3"]["f05_pairs"], 5), "(pair level)")
    print("US_F05 =", round(by_c.get("US", {}).get("macro_f05", float("nan")), 5))
    print("INDIA_F05 =", round(by_c.get("India", {}).get("macro_f05", float("nan")), 5))
    print("SINGLE_MATCH_F05 =", round(mm["1"]["macro_f05"], 5))
    multi_f = pe.filter(pl.col("n_true") >= 2)["f05"].mean()
    print("MULTI_MATCH_F05 =", round(multi_f, 5))
    print("TRUE_EMPTY_ACCURACY =", round(base["correct_empty"] / max(base["true_empty"], 1), 5))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--truth", required=True)
    s.add_argument("--entities")
    v = sub.add_parser("verify")
    v.add_argument("--data", default="../../../dataset/student_resource/dataset")
    a = sub.add_parser("analyze")
    a.add_argument("--data", default="../../../dataset/student_resource/dataset")
    a.add_argument("--work", default="../../../work")
    a.add_argument("--out", default="../../../reports/forensic_results.json")
    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "score":
        pred, truth = read_id_lists(args.pred), read_id_lists(args.truth)
        ents = ([l.strip() for l in open(args.entities, encoding="utf-8") if l.strip()] if args.entities
                else sorted(truth))
        for k, val in evaluate(ents, pred, truth).items():
            print(f"{k:22s} {val:.6f}" if isinstance(val, float) else f"{k:22s} {val}")
    elif args.cmd == "verify":
        verify(args.data)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
