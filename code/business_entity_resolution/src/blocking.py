"""Candidate generation: every Source 2/3 record retrieves its top-K Source 1 records.

Records are indexed by hashed keys: single name tokens, name-token pairs (order-free),
a concatenated core name, address tokens, adjacent address bigrams, and name-token x
address-token combinations. Keys are only compared within the same country label, keys
that are too common on the Source 1 side are skipped, and each candidate is scored by the
summed IDF of the keys it shares with the query.
"""
import polars as pl

from normalize import FILLER, LEGAL, addr_tokens, name_tokens

MAX_NAME_TOKENS = 4
MAX_ADDR_TOKENS = 10


def record_keys(df, id_col, slice_rows=250_000):
    """Long table (id, key:u64) of hashed blocking keys; `df` must have nm/ad columns.

    Built in slices so the intermediate string-key tables stay small."""
    return pl.concat([_record_keys(df.slice(s, slice_rows), id_col)
                      for s in range(0, max(df.height, 1), slice_rows)])


def _record_keys(df, id_col):
    nt = name_tokens(df, id_col).with_columns(pl.int_range(pl.len()).over(id_col).alias("pos"))
    core = nt.filter(~pl.col("tok").is_in(list(LEGAL | FILLER)))
    # Fall back to all tokens when a name is nothing but legal/filler words.
    core = pl.concat([core, nt.join(core.select(id_col).unique(), on=id_col, how="anti")])
    core = core.sort([id_col, "pos"]).with_columns(pl.int_range(pl.len()).over(id_col).alias("pos"))
    core_top = core.filter(pl.col("pos") < MAX_NAME_TOKENS)

    at = (addr_tokens(df, id_col).with_columns(pl.int_range(pl.len()).over(id_col).alias("pos"))
            .filter(pl.col("pos") < 2 * MAX_ADDR_TOKENS))
    at_key = at.filter((pl.col("tok").str.len_chars() >= 3) | pl.col("tok").str.contains(r"^\d+$"))
    at_key = at_key.unique([id_col, "tok"]).filter(pl.int_range(pl.len()).over(id_col) < MAX_ADDR_TOKENS)

    nn = (core_top.join(core_top, on=id_col, suffix="_b").filter(pl.col("tok") < pl.col("tok_b"))
                  .select(id_col, (pl.lit("nn|") + pl.col("tok") + "_" + pl.col("tok_b")).alias("key")))
    bigram = (at.join(at.with_columns(pl.col("pos") - 1), on=[id_col, "pos"], suffix="_nx")
                .select(id_col, (pl.lit("aa|") + pl.col("tok") + "_" + pl.col("tok_nx")).alias("key")))
    na = (core_top.select(id_col, "tok").join(at_key.select(id_col, pl.col("tok").alias("atok")), on=id_col)
                  .select(id_col, (pl.lit("na|") + pl.col("tok") + "@" + pl.col("atok")).alias("key")))
    concat = (core.group_by(id_col, maintain_order=True).agg(pl.col("tok").str.join(""))
                  .filter(pl.col("tok").str.len_chars() >= 5)
                  .select(id_col, (pl.lit("c|") + pl.col("tok")).alias("key")))
    keys = pl.concat([
        core_top.select(id_col, (pl.lit("n|") + pl.col("tok")).alias("key")),
        at_key.select(id_col, (pl.lit("a|") + pl.col("tok")).alias("key")),
        nn, bigram, na, concat,
    ])
    return keys.select(id_col, pl.col("key").hash(seed=17).alias("key")).unique()


def _s1_index(k1, n_s1, df_cap):
    dfk = k1.group_by("key").len().rename({"len": "df"})
    usable = (dfk.filter(pl.col("df") <= df_cap)
                 .with_columns((pl.lit(n_s1, pl.Float64) / pl.col("df")).log().cast(pl.Float32).alias("idf")))
    return usable, k1.join(usable.select("key"), on="key", how="semi")


def _candidates_one_country(usable, k1, ko, top_k, max_keys, budget, chunk_rows):
    ko = (ko.join(usable, on="key")
            .sort(["io", "df"])
            .with_columns(pl.col("df").cum_sum().over("io").alias("cum_df"),
                          pl.int_range(pl.len()).over("io").alias("rank"))
            .filter((pl.col("rank") < max_keys) & ((pl.col("cum_df") - pl.col("df") < budget) | (pl.col("rank") == 0)))
            .select("io", "key", "idf", "df"))
    per_q = (ko.group_by("io").agg(pl.col("df").sum().alias("w")).sort("io")
               .with_columns((pl.col("w").cum_sum() // chunk_rows).alias("chunk")))
    total = int(per_q["w"].sum()) if per_q.height else 0
    ko = ko.join(per_q.select("io", "chunk"), on="io")
    parts = []
    for _, part in ko.group_by(["chunk"], maintain_order=True):
        cand = (part.select("io", "key", "idf").join(k1, on="key")
                    .group_by(["io", "i1"]).agg(pl.col("idf").sum().alias("block_score"),
                                                 pl.len().cast(pl.Int16).alias("n_shared")))
        # A filter copies the kept rows; group_by().head() would return a view that pins the
        # whole grouped table in memory for every chunk (~60 MB each, tens of GB in total).
        parts.append(cand.filter(pl.col("block_score").rank("ordinal", descending=True).over("io") <= top_k))
    return (pl.concat(parts) if parts else None), total


def generate_candidates(s1, others, top_k=20, df_cap=2000, max_keys=24, budget=20_000,
                        chunk_rows=60_000_000, query_chunk=1_000_000, verbose=True):
    """Return (io, i1, block_score, n_shared) candidate pairs using integer row ids.

    Runs one country label at a time (whatever labels exist in the data); queries are
    processed in chunks of `query_chunk` records against a Source 1 index built once.
    """
    out = []
    for country in s1["country"].unique().sort().to_list():
        a = s1.filter(pl.col("country") == country)
        b = others.filter(pl.col("country") == country)
        if b.height == 0:
            continue
        usable, k1 = _s1_index(record_keys(a, "i1"), a.height, df_cap)
        total = n_cand = 0
        for start in range(0, b.height, query_chunk):
            part = b.slice(start, query_chunk)
            cand, t = _candidates_one_country(usable, k1, record_keys(part, "io"), top_k, max_keys,
                                              budget, chunk_rows)
            total += t
            if cand is not None:
                out.append(cand)
                n_cand += cand.height
        if verbose:
            print(f"  blocking[{country}]: s1={a.height:,} queries={b.height:,} join={total:,} "
                  f"cands={n_cand:,}", flush=True)
    return pl.concat(out)
