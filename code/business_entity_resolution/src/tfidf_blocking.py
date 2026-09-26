"""Second blocking pass: char 3-gram TF-IDF on the core name, per country.

The key-based blocking needs at least one whole word in common, so typos and merged words
("sharmaa traders", "srinivasa" vs "shrinivas") slip through. Here every S2/S3 core name is
compared to the S1 core names of its country by cosine similarity of char 3-grams and we
keep the top_n. Trigrams that appear in more than `max_df` of the S1 names are dropped,
otherwise the sparse product gets far too dense.
"""
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _text(df):
    return df.select(pl.when(pl.col("core") != "").then(pl.col("core")).otherwise(pl.col("nm"))
                       .str.replace_all(" ", "_").alias("t"))["t"].to_list()


def tfidf_candidates(s1, others, top_n=10, max_df=0.02, min_sim=0.3, query_chunk=500_000,
                     n_threads=16, verbose=True):
    """(io, i1, tfidf_sim) for the top_n S1 names of each S2/S3 record (same country)."""
    out = []
    for country in s1["country"].unique().sort().to_list():
        a = s1.filter(pl.col("country") == country)
        b = others.filter(pl.col("country") == country)
        if b.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=max_df,
                              sublinear_tf=True, dtype=np.float32)
        s1_t = vec.fit_transform(_text(a)).T.tocsr()
        i1 = a["i1"].to_numpy()
        n = 0
        for start in range(0, b.height, query_chunk):
            part = b.slice(start, query_chunk)
            q = vec.transform(_text(part))
            m = sp_matmul_topn(q, s1_t, top_n=top_n, threshold=min_sim, n_threads=n_threads).tocoo()
            out.append(pl.DataFrame({"io": part["io"].to_numpy()[m.row], "i1": i1[m.col],
                                     "tfidf_sim": m.data.astype(np.float32)}))
            n += m.nnz
        if verbose:
            print(f"  tfidf[{country}]: s1={a.height:,} queries={b.height:,} cands={n:,}", flush=True)
    return pl.concat(out)
