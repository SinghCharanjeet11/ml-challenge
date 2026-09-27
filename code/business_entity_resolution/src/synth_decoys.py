"""Synthetic decoy records for training, built only from the training data itself.

Test has about 2.3 non-matching S2/S3 records per Source 1 entity, train about 1.2, and a model
tuned on train is overconfident on test (its leaderboard-optimal threshold is 0.85 vs 0.2 on the
train holdout). To train and validate under test-like conditions we add ~1.1 decoys per S1
entity. Each decoy copies a real matching record of that entity (or the S1 record when it has
none) and changes it the way the test's near-miss records look:

- house number shifted by a few (or a digit dropped)
- one name word swapped for another business word from the same country
- a made-up name (recombined chunks of real name words) at the same address
- an extra word appended to the name, usually with a shifted house number
- no address, with a swapped or extra word

Decoys are labelled as matching nothing. Output (split "train_aug"):
  work/train_aug_s1p.parquet  (same S1 table)
  work/train_aug_op.parquet   (real + synthetic S2/S3, synthetic io after the real ones)
  work/train_aug_parent.parquet (io, parent i1) for the synthetic records
  work/train_aug_cand_v2.parquet (real v2 candidates + candidates of the synthetic records)
"""
import re
import shutil

import numpy as np
import polars as pl

from blocking import generate_candidates
from features import prepare
from io_utils import gt_pairs, load_split
from normalize import LEGAL, add_clean_columns
from pipeline import BLOCK_PARAMS, KEEP_O, V2_KEY_TOPK, V2_TFIDF_TOPN, log
from tfidf_blocking import tfidf_candidates
from translit import transliterate_addresses, transliterate_names

W = "../../../work"
DATA = "../../../dataset/student_resource/dataset"
RATE = 1.1
SPLIT = "train_aug2"
rng = np.random.default_rng(42)

s1_raw, oth_raw, gt = load_split(DATA, "train")
s1p = pl.read_parquet(f"{W}/train_s1p.parquet")
op = pl.read_parquet(f"{W}/train_op.parquet")
truth = gt_pairs(gt)

# ---- word pools from the dataset (per country): business words and name chunks
words = (s1_raw.select("country", pl.col("business_name").str.to_lowercase()
                       .str.replace_all(r"[^a-z ]", " ").str.split(" ").alias("w"))
         .explode("w").filter((pl.col("w").str.len_chars() >= 4) & ~pl.col("w").is_in(list(LEGAL))))
top = (words.group_by("country", "w").len().sort("len", descending=True)
            .group_by("country", maintain_order=True).head(3000))
pool = {c: top.filter(pl.col("country") == c)["w"].to_list() for c in top["country"].unique().to_list()}
chunks = {c: [w[i:i + 3] for w in ws for i in range(0, len(w) - 2, 3)] for c, ws in pool.items()}

# ---- choose parents and bases
n_s1 = s1_raw.height
k = rng.poisson(RATE, n_s1)
parents = pl.DataFrame({"s1_id": s1_raw["entity_id"], "k": k}).filter(pl.col("k") > 0)
base_true = (truth.join(oth_raw.select(pl.col("entity_id").alias("o_id"), "business_name", "business_address"), on="o_id")
                  .group_by("s1_id").agg(pl.col("business_name"), pl.col("business_address")))
parents = (parents
                  .join(s1_raw.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1_name"),
                                      pl.col("business_address").alias("s1_addr"), "country"), on="s1_id"))
log(f"parents: {parents.height:,} S1 entities, decoys to make: {int(parents['k'].sum()):,}")

NUM = re.compile(r"\d+")


def shift_number(addr):
    m = NUM.search(addr)
    if not m:
        return None
    n = m.group()
    if len(n) >= 3 and rng.random() < 0.3:
        new = n[:-1]
    else:
        new = str(max(1, int(n[:9]) + int(rng.choice([-1, 1])) * int(rng.choice([1, 2, 3, 4, 5, 10, 20]))))
    return addr[:m.start()] + new + addr[m.end():]


def swap_word(name, country):
    ws = name.split()
    idx = [i for i, w in enumerate(ws) if len(w) >= 3 and w.lower().strip(".,&") not in LEGAL]
    if not idx:
        return None
    i = int(rng.choice(idx))
    ws[i] = str(rng.choice(pool[country]))
    return " ".join(ws)


def made_up(name, country):
    ch = chunks[country]
    word = "".join(str(rng.choice(ch)) for _ in range(int(rng.integers(2, 4))))
    tail = [w for w in name.split() if w.lower().strip(".,") in LEGAL]
    return " ".join([word] + tail[-1:])


def extra_word(name, country):
    return f"{name} {rng.choice(pool[country])}"


ABBR = [("street", "st"), ("road", "rd"), ("avenue", "ave"), ("drive", "dr"), ("lane", "ln"), ("boulevard", "blvd"),
        ("north", "n"), ("south", "s"), ("east", "e"), ("west", "w"), ("rue", "r"), ("nagar", "ngr"), ("sector", "sec")]
LEGAL_VARIANTS = [("private limited", "pvt ltd"), ("limited", "ltd"), ("incorporated", "inc"), ("company", "co"),
                  ("corporation", "corp"), ("& ", "and ")]


def typo(text):
    if len(text) < 5:
        return text
    i = int(rng.integers(1, len(text) - 2))
    op = rng.random()
    if op < 0.33:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if op < 0.66:
        return text[:i] + text[i + 1:]
    return text[:i] + text[i] + text[i:]


def noisy_name(name):
    n = name.lower()
    for a, b in LEGAL_VARIANTS:
        if rng.random() < 0.5:
            n = n.replace(a, b) if a in n else n.replace(b, a) if rng.random() < 0.3 else n
    if rng.random() < 0.2:
        n = typo(n)
    return n


def noisy_addr(addr):
    a = addr.lower()
    for full, short in ABBR:
        if rng.random() < 0.5:
            a = a.replace(f" {full} ", f" {short} ") if f" {full} " in a else a.replace(f" {short} ", f" {full} ")
    parts = [x.strip() for x in a.split(",")]
    if len(parts) > 2 and rng.random() < 0.3:
        parts = parts[:-1]                       # drop the last component (state / region)
    if rng.random() < 0.1:
        parts.insert(int(rng.integers(len(parts) + 1)), "null")
    if rng.random() < 0.15:
        parts = parts[1:] + parts[:1]            # component reordering
    a = ", ".join(parts)
    if rng.random() < 0.2:
        a = typo(a)
    return a


rows = []
for r in parents.iter_rows(named=True):
    for _ in range(r["k"]):
        name, addr, c = noisy_name(r["s1_name"]), noisy_addr(r["s1_addr"]), r["country"]
        t = rng.random()
        if t < 0.35:
            new_a = shift_number(addr)
            nn, na = (name, new_a) if new_a else (swap_word(name, c), addr)
        elif t < 0.60:
            nn, na = swap_word(name, c), addr
        elif t < 0.75:
            nn, na = made_up(name, c), addr
        elif t < 0.85:
            nn, na = extra_word(name, c), shift_number(addr) or addr
        elif t < 0.95:
            nn, na = swap_word(name, c) or extra_word(name, c), ""
        else:
            nn, na = extra_word(name, c), ""
        if nn is None:
            nn = extra_word(name, c)
        rows.append((r["s1_id"], nn, na, c))
syn = pl.DataFrame(rows, schema={"parent_id": pl.String, "business_name": pl.String,
                                 "business_address": pl.String, "country": pl.String}, orient="row")
src = rng.choice([2, 3], syn.height)
syn = syn.with_columns(pl.Series("src", src, dtype=pl.Int8),
                       (pl.lit("S") + pl.Series(src).cast(pl.String) + "-SYN" + pl.int_range(pl.len()).cast(pl.String)
                        .str.zfill(8)).alias("entity_id"))
log(f"synthetic records: {syn.height:,}")

# ---- prepare them like real records, append after the real ones
nd = pl.read_parquet(f"{W}/name_dict.parquet")
ad = pl.read_parquet(f"{W}/addr_dict.parquet")
part = syn.select("entity_id", "business_name", "business_address", "country", "src").with_row_index("io", offset=op.height)
part = transliterate_addresses(transliterate_names(part, nd), ad)
synp = prepare(add_clean_columns(part)).select(KEEP_O).with_columns(pl.col("country").cast(op.schema["country"]))
synp = synp.select(op.columns).cast(op.schema)
parent = syn.select(pl.col("parent_id").alias("s1_id")).with_row_index("io", offset=op.height).join(
    s1p.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id").select("io", pl.col("i1").alias("parent"))
parent = parent.with_columns(pl.col("io").cast(op.schema["io"]))

# ---- candidates for the synthetic records only (same v2 blocking as production)
s1n = s1p.select("i1", "country", "nm", "ad", "core")
qn = synp.select("io", "country", "nm", "ad", "core")
keys = generate_candidates(s1n, qn, **{**BLOCK_PARAMS, "top_k": V2_KEY_TOPK})
tf = tfidf_candidates(s1n, qn, top_n=V2_TFIDF_TOPN)
real = pl.read_parquet(f"{W}/train_cand_v2.parquet")
new = (keys.join(tf, on=["io", "i1"], how="full", coalesce=True)
           .with_columns(pl.col("block_score").fill_null(0.0), pl.col("n_shared").fill_null(0),
                         pl.col("tfidf_sim").fill_null(0.0))
           .select(real.columns).cast(real.schema))
log(f"synthetic candidates: {new.height:,}; parent retrieved for "
    f"{new.join(parent.rename({'parent': 'i1'}), on=['io', 'i1'], how='semi').select('io').n_unique() / synp.height:.3f} of decoys")

shutil.copy(f"{W}/train_s1p.parquet", f"{W}/{SPLIT}_s1p.parquet")
pl.concat([op, synp]).write_parquet(f"{W}/{SPLIT}_op.parquet")
parent.write_parquet(f"{W}/{SPLIT}_parent.parquet")
pl.concat([real, new]).write_parquet(f"{W}/{SPLIT}_cand_v2.parquet")
log(f"{SPLIT} written")
