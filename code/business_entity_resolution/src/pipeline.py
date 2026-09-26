"""Shared steps used by train and predict: prep, cached blocking, featurising."""
import os
import time

import numpy as np
import polars as pl

from blocking import generate_candidates
from features import FEATURES, build_features, prepare
from io_utils import gt_pairs, load_split
from normalize import add_clean_columns
from translit import learn_address_dictionary, learn_dictionary, transliterate_addresses, transliterate_names

BLOCK_PARAMS = dict(top_k=10, df_cap=1000, max_keys=24, budget=10_000, query_chunk=250_000,
                    chunk_rows=5_000_000)
# v2 blocking = key blocking with a bigger top_k + the char 3-gram tfidf pass (needs more RAM).
V2_KEY_TOPK = 20
V2_TFIDF_TOPN = 10
# Columns kept after preparation (everything else is dropped to save memory).
KEEP_S1 = ["i1", "entity_id", "country", "nm", "ad", "core", "legal", "adc", "nums", "num0"]
KEEP_O = KEEP_S1[1:] + ["io", "src", "indic", "webname", "alias"]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def learn_dictionaries(data_dir):
    s1, oth, gt = load_split(data_dir, "train")
    pairs = gt_pairs(gt)
    return learn_dictionary(s1, oth, pairs), learn_address_dictionary(s1, oth, pairs)


def prep_records(s1, oth, name_dict, addr_dict):
    """Add integer ids, transliterate Source 2/3 text, and add normalised columns."""
    s1 = prepare(add_clean_columns(s1.with_row_index("i1"))).select(KEEP_S1)
    parts = []
    for start in range(0, oth.height, 2_000_000):
        part = oth.slice(start, 2_000_000).with_row_index("io", offset=start)
        part = transliterate_addresses(transliterate_names(part, name_dict), addr_dict)
        parts.append(prepare(add_clean_columns(part)).select(KEEP_O))
    oth = pl.concat(parts)
    cat = pl.Enum(sorted(set(s1["country"].unique()) | set(oth["country"].unique())))
    return s1.with_columns(pl.col("country").cast(cat)), oth.with_columns(pl.col("country").cast(cat))


def top_k_with_s1_context(cand, keep_k):
    """Top keep_k per query + S1-side context over the whole candidate table (so it doesn't
    depend on sampling or chunking).
    """
    if keep_k:
        cand = (cand.sort(["io", "block_score"], descending=[False, True])
                    .group_by("io", maintain_order=True).head(keep_k))
    return cand.with_columns(
        pl.len().over("i1").alias("n_cand_1"),
        pl.col("block_score").rank("ordinal", descending=True).over("i1").alias("rank_block_1"))


def featurise_chunks(cand, s1p, op, keep_k=10, chunk_queries=400_000, queries=None):
    """Yields feature frames ~400k queries at a time. `queries` limits it to a subset."""
    cand = top_k_with_s1_context(cand, keep_k)
    if queries is not None:
        cand = cand.filter(pl.col("io").is_in(queries.implode()))
    cand = cand.sort("io")
    # Row offset where each chunk of `chunk_queries` queries starts.
    starts = cand.select((pl.col("io") != pl.col("io").shift()).fill_null(True).arg_true())["io"]
    bounds = starts.gather_every(chunk_queries).to_list() + [cand.height]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        f = build_features(cand.slice(lo, hi - lo), s1p, op)
        yield f.select(["io", "i1"] + [pl.col(x).cast(pl.Float32) for x in FEATURES])


def featurise(cand, s1p, op, keep_k=10, chunk_queries=400_000, queries=None):
    return pl.concat(featurise_chunks(cand, s1p, op, keep_k, chunk_queries, queries))


def prep_cached(work, split, load, name_dict, addr_dict):
    """prep_records, cached as parquet in work/."""
    p1, po = f"{work}/{split}_s1p.parquet", f"{work}/{split}_op.parquet"
    if not (os.path.exists(p1) and os.path.exists(po)):
        s1, oth = load()
        s1p, op = prep_records(s1, oth, name_dict, addr_dict)
        del s1, oth
        s1p.write_parquet(p1)
        op.write_parquet(po)
        del s1p, op
    return pl.read_parquet(p1), pl.read_parquet(po)


def block_cached(work, split, version="v1"):
    """Load (or build) the candidates for a prepared split. Only loads the columns blocking needs.

    v1: key blocking top 10. v2: key blocking top V2_KEY_TOPK merged with the tfidf top
    V2_TFIDF_TOPN (pairs found by only one pass get 0 for the other pass's score).
    """
    path = f"{work}/{split}_cand.parquet" if version == "v1" else f"{work}/{split}_cand_{version}.parquet"
    if not os.path.exists(path):
        cols = ["country", "nm", "ad"]
        s1n = pl.read_parquet(f"{work}/{split}_s1p.parquet", columns=["i1"] + cols + ["core"])
        on = pl.read_parquet(f"{work}/{split}_op.parquet", columns=["io"] + cols + ["core"])
        if version == "v1":
            cand = generate_candidates(s1n, on, **BLOCK_PARAMS)
        else:
            from tfidf_blocking import tfidf_candidates
            keys = generate_candidates(s1n, on, **{**BLOCK_PARAMS, "top_k": V2_KEY_TOPK})
            tf = tfidf_candidates(s1n, on, top_n=V2_TFIDF_TOPN)
            cand = (keys.join(tf, on=["io", "i1"], how="full", coalesce=True)
                        .with_columns(pl.col("block_score").fill_null(0.0), pl.col("n_shared").fill_null(0),
                                      pl.col("tfidf_sim").fill_null(0.0)))
        cand.write_parquet(path)
    return pl.read_parquet(path)
