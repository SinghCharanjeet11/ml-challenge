"""Transformer cross-encoder for the pairs the LightGBM cascade is unsure about.

A small pretrained cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2, Apache-2.0, 22.7M
parameters) reads "S1 name | address" and "record name | address" together and is fine-tuned to
say whether they are the same business. It only scores pairs in the uncertain band; its score is
then combined with the cascade probability by a small model fitted on the corrected holdout.

Stages (run in order, each saves its output in work/):
  python cross_encoder.py pairs     # train pairs (anchored folds) with out-of-fold filter probability
  python cross_encoder.py train     # fine-tune on training-fold pairs
  python cross_encoder.py score     # score uncertain holdout + test pairs
  python cross_encoder.py combine   # stack with the cascade on the holdout, write test files
"""
import glob
import json
import math
import sys
import time

import numpy as np
import polars as pl

W, OUT = "../../../work", "../../../output"
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BAND = (0.02, 0.99)          # cascade probabilities worth a second opinion
MAX_LEN = 96


def log(m, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {m}", flush=True)


def texts(split):
    s1 = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1", "nm", "ad"]).select(
        "i1", (pl.col("nm") + " | " + pl.col("ad")).alias("t1"))
    op = pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "nm", "ad"]).select(
        "io", (pl.col("nm") + " | " + pl.col("ad")).alias("t2"))
    return s1, op


def stage_pairs():
    import lightgbm as lgb
    from io_utils import gt_pairs, load_split
    cfg = json.load(open(f"{W}/config_cascade_anc.json"))
    feats = cfg["filter_features"]
    s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
    op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
    _, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
    owner = (gt_pairs(gt).join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
             .join(op.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", pl.col("i1").alias("owner")))
    cand = pl.read_parquet(f"{W}/train_cand_v2.parquet", columns=["io", "i1", "block_score"])
    anchor = (cand.sort(["io", "block_score"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
                  .select("io", pl.col("i1").alias("anchor")))
    del cand
    models = [lgb.Booster(model_file=f"{W}/model_filter_{k}_anc.txt") for k in (0, 1)]
    out = []
    for part in sorted(glob.glob(f"{W}/feat_all_train/part_*.parquet")):
        d = pl.read_parquet(part, columns=["io", "i1"] + feats).join(owner, on="io", how="left").join(anchor, on="io", how="left")
        key = pl.coalesce("owner", "anchor", "io")
        d = d.with_columns((pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
                           (key.hash(seed=7) % 2).alias("fold"), (key.hash(seed=5) % 10 == 0).alias("val"))
        for k in (0, 1):
            dk = d.filter(pl.col("fold") == k)
            x = dk.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            out.append(dk.select("io", "i1", "y", "val").with_columns(pl.Series("p1", models[k].predict(x), dtype=pl.Float32)))
    p = pl.concat(out)
    p = p.filter((pl.col("p1") >= cfg["prune"]) & (pl.col("p1").rank("ordinal", descending=True).over("io") <= cfg["max_per_record"]))
    p.write_parquet(f"{W}/ce_train_pairs.parquet")
    log(f"pruned train pairs {p.height:,}; training fold {int((~p['val']).sum()):,}")


def load_model(path=BASE):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(16)
    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)


def stage_train(n_band=260_000, n_high=50_000, n_low=40_000, batch=64, lr=3e-5):
    import torch
    p = pl.read_parquet(f"{W}/ce_train_pairs.parquet").filter(~pl.col("val"))
    band = p.filter((pl.col("p1") >= BAND[0]) & (pl.col("p1") < BAND[1]))
    sel = pl.concat([band.sample(min(n_band, band.height), seed=1),
                     p.filter(pl.col("p1") >= BAND[1]).sample(n_high, seed=2),
                     p.filter(pl.col("p1") < BAND[0]).sample(n_low, seed=3)]).sample(fraction=1.0, shuffle=True, seed=4)
    s1, op = texts("train")
    sel = sel.join(s1, on="i1").join(op, on="io")
    log(f"training pairs {sel.height:,} (band {min(n_band, band.height):,}), positives {sel['y'].mean():.3f}")
    tok, model = load_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(sel.height / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    t1, t2, y = sel["t1"].to_list(), sel["t2"].to_list(), sel["y"].cast(pl.Float32).to_numpy()
    for s in range(steps):
        a, b = s * batch, (s + 1) * batch
        enc = tok(t1[a:b], t2[a:b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        loss = lossf(model(**enc).logits.squeeze(-1), torch.tensor(y[a:b]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if s % 500 == 0:
            log(f"step {s}/{steps} loss {loss.item():.4f}")
    model.save_pretrained(f"{W}/ce_model")
    tok.save_pretrained(f"{W}/ce_model")
    log("model saved")


def predict(tok, model, t1, t2, batch=512):
    import torch
    model.eval()
    out = np.empty(len(t1), dtype=np.float32)
    with torch.inference_mode():
        for a in range(0, len(t1), batch):
            enc = tok(t1[a:a + batch], t2[a:a + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            out[a:a + batch] = model(**enc).logits.squeeze(-1).numpy()
            if (a // batch) % 400 == 0:
                log(f"  scored {a + batch:,}/{len(t1):,}")
    return out


def stage_score():
    tok, model = load_model(f"{W}/ce_model")
    for name, f, split in (("holdout", "holdout_scored_cascade_anc.parquet", "train"),
                           ("test", "test_scored_cascade_anc.parquet", "test")):
        sc = pl.read_parquet(f"{W}/{f}")
        band = sc.filter((pl.col("p") >= BAND[0]) & (pl.col("p") < BAND[1])).select("io", "i1")
        s1, op = texts(split)
        band = band.join(s1, on="i1").join(op, on="io")
        log(f"{name}: scoring {band.height:,} pairs")
        ce = predict(tok, model, band["t1"].to_list(), band["t2"].to_list())
        band.select("io", "i1").with_columns(pl.Series("ce", ce)).write_parquet(f"{W}/ce_scores_{name}.parquet")
        log(f"{name} done")


def stage_combine():
    import lightgbm as lgb
    from decide import assign
    from io_utils import gt_pairs, load_split, write_id_lists_chunked
    from metric import macro_f05
    s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
    op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
    s1ids = s1.select("i1", pl.col("entity_id").alias("s1_id"))
    oids = op.select("io", pl.col("entity_id").alias("o_id"))
    _, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
    val = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)
    ids = val["s1_id"].to_list()
    truth = gt_pairs(gt).join(val, on="s1_id", how="semi")

    def prep(sc, ce):
        d = sc.join(ce, on=["io", "i1"], how="left")
        # record-level context of the cross-encoder: best ce among the record's candidates
        d = d.with_columns(pl.col("ce").max().over("io").alias("ce_rec_max"))
        return d.with_columns(pl.col("ce").is_not_null().alias("scored"),
                              (pl.col("ce") - pl.col("ce_rec_max")).alias("ce_gap"))

    h = prep(pl.read_parquet(f"{W}/holdout_scored_cascade_anc.parquet"), pl.read_parquet(f"{W}/ce_scores_holdout.parquet"))
    F = ["p", "ce", "ce_gap"]
    hb = h.filter(pl.col("scored"))
    fold = (hb["i1"].hash(seed=41) % 2).to_numpy()
    X, y = hb.select(F).to_numpy(), hb["y"].to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1)
    oof = np.empty(len(y))
    for k in (0, 1):
        m = lgb.train(params, lgb.Dataset(X[fold != k], y[fold != k]), 300)
        oof[fold == k] = m.predict(X[fold == k])
    hc = h.join(hb.select("io", "i1").with_columns(pl.Series("p_comb", oof)), on=["io", "i1"], how="left").with_columns(
        pl.coalesce("p_comb", "p").alias("p_comb"))

    def f05(d, col, tau, dup):
        k = assign(d.select("io", "i1", pl.col(col).alias("p"), "unowned"), tau)
        pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
        if dup:
            pred = pl.concat([pred, k.filter(pl.col("unowned")).join(s1ids, on="i1").join(oids, on="io").select(
                "s1_id", pl.col("o_id") + "_dup")])
        return macro_f05(ids, pred, truth)

    from sklearn.metrics import roc_auc_score
    log(f"holdout band pairs {hb.height:,}: AUC cascade p {roc_auc_score(y, hb['p']):.4f}  ce {roc_auc_score(y, hb['ce']):.4f}  "
        f"combined(oof) {roc_auc_score(y, oof):.4f}")
    res = {}
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        a, b = f05(hc, "p", tau, True), f05(hc, "p_comb", tau, True)
        res[tau] = (a, b)
        log(f"tau={tau}: test-density estimate  cascade {a:.5f}  +cross-encoder {b:.5f}  (plain holdout: "
            f"{f05(hc, 'p', tau, False):.5f} -> {f05(hc, 'p_comb', tau, False):.5f})")
    # final combiner on the whole holdout band, applied to test
    m = lgb.train(params, lgb.Dataset(X, y), 300)
    t = prep(pl.read_parquet(f"{W}/test_scored_cascade_anc.parquet"), pl.read_parquet(f"{W}/ce_scores_test.parquet"))
    tb = t.filter(pl.col("scored"))
    t = t.join(tb.select("io", "i1").with_columns(pl.Series("p_comb", m.predict(tb.select(F).to_numpy()))),
               on=["io", "i1"], how="left").with_columns(pl.coalesce("p_comb", "p").alias("p_comb"))
    t.select("io", "i1", pl.col("p_comb").alias("p")).write_parquet(f"{W}/test_scored_CE.parquet")
    b = t.sort(["io", "p_comb"], descending=[False, True]).group_by("io", maintain_order=True).head(1)["p_comb"].sort(descending=True)
    log("test matches by threshold: " + " ".join(f"{x}:{int((b >= x).sum()):,}" for x in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9))
        + f" | p at 5,762,771th match: {float(b[5762770]):.4f}")


if __name__ == "__main__":
    {"pairs": stage_pairs, "train": stage_train, "score": stage_score, "combine": stage_combine}[sys.argv[1]]()
