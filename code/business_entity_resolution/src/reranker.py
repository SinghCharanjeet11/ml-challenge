"""Multilingual cross-encoder over the whole candidate set.

BAAI/bge-reranker-base (MIT, XLM-RoBERTa base, 278M parameters) reads "S1 name | address" and
"record name | address" together. Unlike the small English cross-encoder of our earlier submission it
reads the original scripts (Devanagari etc., French accents), so we feed it the raw text; Indic
names also get our transliteration appended.

Two models are cross-fitted: every record gets a fold from the S1 entity it belongs to (or its
top blocking candidate when it matches nothing), model k trains on fold k, and every pair (train
and test alike) is scored by the model of the other fold. The train scores are therefore out of
fold and can be used as features by the final LightGBM (stage3.py).

  python reranker.py text train|test      # model input text per record                  (CPU)
  python reranker.py pairs                # training pairs from blocking + fold per record (CPU)
  python reranker.py fit 0|1              # fine-tune one model per fold                   (GPU)
  python reranker.py score train|test     # score the pruned candidate pairs               (GPU)
"""
import argparse
import math
import time

import numpy as np
import polars as pl

W, DATA = "../../../work", "../../../dataset/student_resource/dataset"
BASE = "BAAI/bge-reranker-base"
MAX_LEN = 128


def log(m, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {m}", flush=True)


def stage_text(split):
    from io_utils import load_split
    s1, oth, _ = load_split(DATA, split)

    def clean(c):
        return pl.col(c).fill_null("").str.replace_all(r"\s+", " ").str.strip_chars()

    s1p = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1", "entity_id", "nm", "ad"])
    s1p.join(s1, on="entity_id").select(
        "i1", (clean("business_name") + " | " + clean("business_address")).alias("t"),
        (pl.col("nm") + " | " + pl.col("ad")).alias("tn")).write_parquet(f"{W}/{split}_rr_s1.parquet")
    op = pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "entity_id", "nm", "ad", "indic"])
    name = pl.when(pl.col("indic")).then(clean("business_name") + " / " + pl.col("nm")).otherwise(clean("business_name"))
    op = op.join(oth, on="entity_id").select(
        "io", (name + " | " + clean("business_address")).alias("t"), (pl.col("nm") + " | " + pl.col("ad")).alias("tn"))
    op.write_parquet(f"{W}/{split}_rr_op.parquet")
    log(f"{split}: text for {s1p.height:,} S1 entities and {op.height:,} records")


def stage_pairs(version, splits):
    """Per record: the true link (if any) and its top blocking candidates as negatives.

    Also writes the fold of every train / test record. The fold follows the S1 entity the
    record belongs to, else its top blocking candidate, so a record and its decoys share a fold."""
    from io_utils import gt_pairs, load_split
    s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
    op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
    _, _, gt = load_split(DATA, "train")
    owner = (gt_pairs(gt).join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
             .join(op.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", pl.col("i1").alias("owner")))
    del s1, op, gt
    for split in splits:
        folds, negs = [], []
        # v1 = key blocking only (no tfidf pass): negatives come from the key ranking alone
        path = f"{W}/{split}_cand.parquet" if version == "v1" else f"{W}/{split}_cand_{version}.parquet"
        for j in range(4):   # quarters of the records keep memory down
            c = pl.scan_parquet(path).filter(pl.col("io") % 4 == j)
            c = c.select("io", "i1", "block_score", pl.col("tfidf_sim") if version != "v1" else pl.lit(0.0).alias("tfidf_sim")).collect()
            c = c.with_columns(pl.col("block_score").rank("ordinal", descending=True).over("io").alias("rb"),
                               (pl.col("tfidf_sim").rank("ordinal", descending=True).over("io") if version != "v1"
                                else pl.lit(99, dtype=pl.UInt32)).alias("rt"))
            anchor = c.filter(pl.col("rb") == 1).select("io", pl.col("i1").alias("anchor"))
            if split == "train":
                anchor = anchor.join(owner, on="io", how="left").select(
                    "io", pl.coalesce("owner", "anchor").alias("anchor"))
            folds.append(anchor.select("io", (pl.col("anchor").hash(seed=13) % 2).cast(pl.Int8).alias("cefold")))
            if split == "train":
                c = c.join(owner, on="io", how="left").filter(pl.col("i1") != pl.col("owner").fill_null(2**32 - 1))
                negs.append(c.filter((pl.col("rb") <= 3) | (pl.col("rt") <= 2)).select(
                    "io", "i1", ((pl.col("rb") == 1) | (pl.col("rt") == 1)).alias("hard")))
            del c
        folds = pl.concat(folds)
        folds.write_parquet(f"{W}/{split}_rr_fold.parquet")
        if split == "train":
            pos = owner.select("io", pl.col("owner").alias("i1"), pl.lit(True).alias("hard"))
            p = pl.concat([pos.with_columns(pl.lit(True).alias("y")),
                           pl.concat(negs).with_columns(pl.lit(False).alias("y"))]).join(folds, on="io")
            p.write_parquet(f"{W}/rr_train_pairs.parquet")
            log(f"train pairs: {int(p['y'].sum()):,} positive, {int((~p['y']).sum()):,} negative "
                f"({int((~p['y'] & p['hard']).sum()):,} hardest)")
        log(f"{split}: folds for {folds.height:,} records")


def texts(split, col="t"):
    return (pl.read_parquet(f"{W}/{split}_rr_s1.parquet", columns=["i1", col]).rename({col: "t1"}),
            pl.read_parquet(f"{W}/{split}_rr_op.parquet", columns=["io", col]).rename({col: "t2"}))


def sample_pairs(fold, n_pos, n_neg, seed):
    """Positives and negatives of one fold; the hardest negatives (top of a blocking ranking) make
    up two thirds of the negatives."""
    p = pl.read_parquet(f"{W}/rr_train_pairs.parquet").filter(pl.col("cefold") == fold)
    pos, neg = p.filter(pl.col("y")), p.filter(~pl.col("y"))
    hard, rest = neg.filter(pl.col("hard")), neg.filter(~pl.col("hard"))
    n_hard = min(hard.height, 2 * n_neg // 3)
    return pl.concat([pos.sample(min(n_pos, pos.height), seed=seed), hard.sample(n_hard, seed=seed + 1),
                      rest.sample(min(rest.height, n_neg - n_hard), seed=seed + 2)]).select("io", "i1", "y")


def encode(tok, t1, t2):
    return tok(t1, t2, truncation=True, max_length=MAX_LEN)["input_ids"]


def pad(ids, pad_id):
    n = max(len(x) for x in ids)
    a = np.full((len(ids), n), pad_id, dtype=np.int64)
    for i, x in enumerate(ids):
        a[i, :len(x)] = x
    return a


def length_batches(ids, batch, rng):
    """Shuffled batches of similar length (less padding)."""
    order = rng.permutation(len(ids))
    out = []
    for a in range(0, len(order), batch * 50):
        chunk = order[a:a + batch * 50]
        chunk = chunk[np.argsort([len(ids[i]) for i in chunk], kind="stable")]
        out += [chunk[b:b + batch] for b in range(0, len(chunk), batch)]
    rng.shuffle(out)
    return out


def predict(tok, model, t1, t2, batch=512):
    import torch
    model.eval()
    out = np.empty(len(t1), dtype=np.float32)
    step = 200_000
    for a in range(0, len(t1), step):
        ids = encode(tok, t1[a:a + step], t2[a:a + step])
        order = np.argsort([len(x) for x in ids], kind="stable")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, len(order), batch):
                idx = order[b:b + batch]
                x = torch.from_numpy(pad([ids[i] for i in idx], tok.pad_token_id)).cuda()
                out[a + idx] = model(input_ids=x, attention_mask=(x != tok.pad_token_id).long()).logits[:, 0].float().cpu().numpy()
        log(f"  scored {min(a + step, len(t1)):,}/{len(t1):,}")
    return out


def stage_fit(fold, n_pos, n_neg, text, tag, batch=128, lr=3e-5, check=False, init=None, seed=0):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(fold + seed)
    sel = sample_pairs(fold, n_pos, n_neg, seed=10 * fold + seed)
    s1t, opt = texts("train", text)
    sel = sel.join(s1t, on="i1").join(opt, on="io")
    tok = AutoTokenizer.from_pretrained(BASE)
    # init: continue from an earlier fit of this fold on a fresh sample (different seed)
    model = AutoModelForSequenceClassification.from_pretrained(init or BASE, num_labels=1).cuda()
    ids = encode(tok, sel["t1"].to_list(), sel["t2"].to_list())
    y = sel["y"].cast(pl.Float32).to_numpy()
    lens = np.array([len(x) for x in ids])
    log(f"fold {fold}: {len(y):,} pairs, {y.mean():.3f} positive; tokens median {int(np.median(lens))}, "
        f"p99 {int(np.percentile(lens, 99))}, truncated {(lens >= MAX_LEN).mean():.4f}")
    batches = length_batches(ids, batch, np.random.default_rng(fold + seed))
    opt_ = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(opt_, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t, run = time.time(), 0.0
    for s, idx in enumerate(batches):
        x = torch.from_numpy(pad([ids[i] for i in idx], tok.pad_token_id)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(input_ids=x, attention_mask=(x != tok.pad_token_id).long()).logits[:, 0]
        loss = lossf(logit.float(), torch.from_numpy(y[idx]).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_.step(); sched.step(); opt_.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if s % 500 == 0 or s == steps - 1:
            log(f"step {s}/{steps} loss {run:.4f} ({(s + 1) * batch / (time.time() - t):.0f} pairs/s)")
    model.save_pretrained(f"{W}/rr_model_{fold}{tag}")
    tok.save_pretrained(f"{W}/rr_model_{fold}{tag}")
    log("model saved")
    if check:   # quick look on pairs of the other fold, drawn the same way
        from sklearn.metrics import log_loss, roc_auc_score
        ev = sample_pairs(1 - fold, 40_000, 60_000, seed=99).join(s1t, on="i1").join(opt, on="io")
        ye = ev["y"].to_numpy()
        z = predict(tok, model, ev["t1"].to_list(), ev["t2"].to_list())
        pr = 1 / (1 + np.exp(-z))
        hard = (~ev["y"]).to_numpy()
        log(f"CHECK {text}{tag}: AUC {roc_auc_score(ye, z):.5f}  logloss {log_loss(ye, pr):.5f}  "
            f"errors at 0.5: {int(((pr >= 0.5) != ye).sum())} of {len(ye):,} (false matches {int(((pr >= 0.5) & hard).sum())})")


def stage_score(split, tag):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    pairs = pl.read_parquet(f"{W}/rr_pruned_{split}.parquet").join(
        pl.read_parquet(f"{W}/{split}_rr_fold.parquet"), on="io", how="left").with_columns(pl.col("cefold").fill_null(0))
    s1t, opt = texts(split)
    out = []
    for k in (0, 1):
        d = pairs.filter(pl.col("cefold") != k).join(s1t, on="i1").join(opt, on="io")
        path = f"{W}/rr_model_{k}{tag}"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).cuda()
        log(f"{split}: model {k} scores {d.height:,} pairs")
        out.append(d.select("io", "i1").with_columns(pl.Series("ce", predict(tok, model, d["t1"].to_list(), d["t2"].to_list()))))
        del model
    pl.concat(out).write_parquet(f"{W}/rr_scores_{split}{tag}.parquet")
    log(f"{split} done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["text", "pairs", "fit", "score"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n_pos", type=int, default=700_000)
    ap.add_argument("--n_neg", type=int, default=1_000_000)
    ap.add_argument("--text", default="t", help="t = raw text, tn = our normalised text")
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true", help="after fitting, score a sample of the other fold")
    ap.add_argument("--cand", default="v2", help="candidate file the training pairs come from (v1 or v2)")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--init", default=None, help="continue from this model directory")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "text":
        stage_text(a.arg)
    elif a.stage == "pairs":
        stage_pairs(a.cand, a.splits.split(","))
    elif a.stage == "fit":
        stage_fit(int(a.arg), a.n_pos, a.n_neg, a.text, a.tag, lr=a.lr, check=a.check, init=a.init, seed=a.seed)
    else:
        stage_score(a.arg, a.tag)
