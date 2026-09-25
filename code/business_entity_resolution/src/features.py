"""Pairwise features for (Source 2/3 record, Source 1 candidate) pairs."""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

from normalize import ADDR_CANON, FILLER, IN_STATES, LEGAL, NAME_CANON, US_STATES
from translit import INDIC

ADDR_MAP = {**ADDR_CANON, **US_STATES, **IN_STATES}


def _deleet(e):
    has_both = e.str.contains(r"[a-z]") & e.str.contains(r"[0-9]") & ~e.str.contains(r"^[0-9]+[a-z]{1,2}$")
    fixed = (e.str.replace_all("0", "o").str.replace_all("1", "l").str.replace_all("3", "e")
             .str.replace_all("5", "s").str.replace_all("4", "a").str.replace_all("7", "t"))
    return pl.when(has_both).then(fixed).otherwise(e)


def prepare(df):
    """Add the per-record columns the pair features need (df must already have nm/ad)."""
    toks = pl.col("nm").str.split(" ").list.eval(_deleet(pl.element()).replace(NAME_CANON))
    core = toks.list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL | FILLER)) & (pl.element() != "")))
    legal = toks.list.eval(pl.element().filter(pl.element().is_in(list(LEGAL - {"&", "the", "l", "p", "c"}))))
    atoks = pl.col("ad").str.split(" ").list.eval(pl.element().replace(ADDR_MAP))
    return df.with_columns(
        core.list.join(" ").alias("core"),
        legal.list.unique().list.sort().list.join(" ").alias("legal"),
        atoks.list.join(" ").alias("adc"),
        pl.col("ad").str.extract_all(r"\d+").list.unique().alias("nums"),
        pl.col("ad").str.extract(r"(\d+)").alias("num0"),
        pl.col("business_name").str.contains(INDIC).alias("indic"),
        pl.col("business_name").str.contains(r"(?i)\.com|\.in\b|^[#@]").alias("webname"),
        pl.col("business_name").str.contains(r"(?i)\b(dba|fka|aka|a/k/a|d/b/a|f/k/a)\b").alias("alias"),
    )


def _pairwise(scorer, a, b, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def build_features(cand, s1p, op):
    """cand: (io, i1, block_score, n_shared, n_cand_1, rank_block_1). s1p/op: prepared frames."""
    s1_cols = ["i1", "nm", "core", "legal", "adc", "nums", "num0"]
    o_cols = ["io", "nm", "core", "legal", "adc", "nums", "num0", "src", "indic", "webname", "alias"]
    df = (cand.join(op.select(o_cols), on="io")
              .join(s1p.select(s1_cols), on="i1", suffix="_1"))

    nm_o, nm_1 = df["nm"].to_list(), df["nm_1"].to_list()
    co_o, co_1 = df["core"].to_list(), df["core_1"].to_list()
    ad_o, ad_1 = df["adc"].to_list(), df["adc_1"].to_list()
    cc_o = [c.replace(" ", "") for c in co_o]
    cc_1 = [c.replace(" ", "") for c in co_1]
    f = {
        "nm_ratio": _pairwise(fuzz.ratio, nm_o, nm_1),
        "nm_tset": _pairwise(fuzz.token_set_ratio, nm_o, nm_1),
        "core_tsort": _pairwise(fuzz.token_sort_ratio, co_o, co_1),
        "core_tset": _pairwise(fuzz.token_set_ratio, co_o, co_1),
        "core_partial": _pairwise(fuzz.partial_ratio, co_o, co_1),
        "core_jw": _pairwise(distance.JaroWinkler.normalized_similarity, co_o, co_1),
        "concat_ratio": _pairwise(fuzz.ratio, cc_o, cc_1),
        "concat_partial": _pairwise(fuzz.partial_ratio, cc_o, cc_1),
        "ad_tset": _pairwise(fuzz.token_set_ratio, ad_o, ad_1),
        "ad_tsort": _pairwise(fuzz.token_sort_ratio, ad_o, ad_1),
        "ad_partial": _pairwise(fuzz.partial_ratio, ad_o, ad_1),
    }
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])

    inter = pl.col("nums").list.set_intersection("nums_1").list.len()
    union = pl.col("nums").list.set_union("nums_1").list.len()
    first_o = pl.col("core").str.split(" ").list.first()
    first_1 = pl.col("core_1").str.split(" ").list.first()
    df = df.with_columns(
        pl.col("nums").list.len().alias("n_num_o"),
        pl.col("nums_1").list.len().alias("n_num_1"),
        inter.alias("num_common"),
        (inter / pl.max_horizontal(union, 1)).alias("num_jacc"),
        (pl.col("num0") == pl.col("num0_1")).fill_null(False).alias("num0_eq"),
        (first_o == first_1).fill_null(False).alias("first_tok_eq"),
        (pl.col("core") == pl.col("core_1")).alias("core_eq"),
        ((pl.col("legal") != "") & (pl.col("legal_1") != "") & (pl.col("legal") != pl.col("legal_1"))).alias("legal_conflict"),
        (pl.col("legal") == pl.col("legal_1")).alias("legal_eq"),
        (pl.col("adc") == "").alias("addr_missing"),
        pl.col("core").str.len_chars().alias("len_core_o"),
        pl.col("core_1").str.len_chars().alias("len_core_1"),
    )

    # House-number distance. Near-duplicate distractors typically sit a few numbers away
    # (6705 vs 6708) while true matches mostly repeat the number exactly.
    num = lambda c, j: pl.col(c).list.get(j, null_on_oob=True).str.slice(0, 9).cast(pl.Float64)
    diffs = [(num("nums", a) - num("nums_1", b)).abs() for a in range(3) for b in range(3)]
    n0, n1 = pl.col("num0"), pl.col("num0_1")
    d0 = (n0.str.slice(0, 9).cast(pl.Float64) - n1.str.slice(0, 9).cast(pl.Float64)).abs()
    min_diff = pl.min_horizontal(diffs)
    df = df.with_columns(
        d0.log1p().alias("hn0_logdiff"),
        min_diff.log1p().alias("hn_min_logdiff"),
        ((min_diff > 0) & (min_diff <= 20)).fill_null(False).alias("hn_near"),
        ((n0 != n1) & (n0.str.starts_with(n1) | n1.str.starts_with(n0))).fill_null(False).alias("hn0_prefix"),
        pl.col("core").str.split(" ").list.set_difference(pl.col("core_1").str.split(" ")).list.len().alias("extra_tok_o"),
        pl.col("core_1").str.split(" ").list.set_difference(pl.col("core").str.split(" ")).list.len().alias("extra_tok_1"),
        pl.col("nm").str.split(" ").list.set_difference(pl.col("nm_1").str.split(" ")).list.len().alias("extra_all_o"),
        pl.col("nm_1").str.split(" ").list.set_difference(pl.col("nm").str.split(" ")).list.len().alias("extra_all_1"),
    )

    # Context: how this candidate compares with the other candidates of the same query,
    # and how crowded the Source 1 record is.
    df = df.with_columns(
        (pl.col("core_tsort") + pl.col("ad_tset")).alias("combo"),
    )
    grp_o = [
        pl.len().over("io").alias("n_cand_o"),
        pl.col("block_score").rank("ordinal", descending=True).over("io").alias("rank_block"),
        (pl.col("block_score").max().over("io") - pl.col("block_score")).alias("gap_block"),
        pl.col("combo").rank("ordinal", descending=True).over("io").alias("rank_combo"),
        (pl.col("combo").max().over("io") - pl.col("combo")).alias("gap_combo"),
        (pl.col("core_tsort").max().over("io") - pl.col("core_tsort")).alias("gap_name"),
        (pl.col("ad_tset").max().over("io") - pl.col("ad_tset")).alias("gap_addr"),
        (pl.col("core_1") == pl.col("core_1")).sum().over(["io", "core_1"]).alias("same_core_in_cands"),
    ]
    df = df.with_columns(grp_o)
    # Second-best gap: how far ahead the best candidate is of the runner-up for this query.
    second = (df.select("io", "combo").sort(["io", "combo"], descending=[False, True])
                .group_by("io", maintain_order=True).agg(pl.col("combo").get(1, null_on_oob=True).alias("combo_2nd")))
    df = df.join(second, on="io", how="left").with_columns(
        (pl.col("combo") - pl.col("combo_2nd").fill_null(0)).alias("margin_2nd"))
    return df


FEATURES = [
    "block_score", "n_shared", "nm_ratio", "nm_tset", "core_tsort", "core_tset", "core_partial", "core_jw",
    "concat_ratio", "concat_partial", "ad_tset", "ad_tsort", "ad_partial", "n_num_o", "n_num_1", "num_common",
    "num_jacc", "num0_eq", "first_tok_eq", "core_eq", "legal_conflict", "legal_eq", "addr_missing",
    "len_core_o", "len_core_1", "src", "indic", "webname", "alias", "combo", "n_cand_o", "rank_block",
    "gap_block", "rank_combo", "gap_combo", "gap_name", "gap_addr", "same_core_in_cands", "n_cand_1",
    "rank_block_1", "margin_2nd", "hn0_logdiff", "hn_min_logdiff", "hn_near", "hn0_prefix",
    "extra_tok_o", "extra_tok_1", "extra_all_o", "extra_all_1",
]
