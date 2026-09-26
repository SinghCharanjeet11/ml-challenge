"""Indic -> Latin word dictionary learned from the train matches.

S2/S3 often write an English name phonetically in an Indian script, word for word
("Modern Agro Private Limited" -> 4 Gujarati words). When both names have the same number of
words we line them up by position and keep the most common Latin word for each Indic word.
"""
import polars as pl

INDIC = r"[ऀ-෿]"  # Devanagari .. Malayalam blocks (covers all nine scripts seen)


def _words(col):
    return (pl.col(col).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
            .str.strip_chars().str.split(" "))


def learn_dictionary(s1, others, pairs, min_count=2, min_share=0.5):
    """(src_tok, dst_tok) table: Indic word -> Latin word."""
    p = (pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1")), on="s1_id")
              .join(others.select(pl.col("entity_id").alias("o_id"), pl.col("business_name").alias("n2")), on="o_id")
              .filter(pl.col("n2").str.contains(INDIC))
              .select(_words("n1").alias("w1"), _words("n2").alias("w2"))
              .filter(pl.col("w1").list.len() == pl.col("w2").list.len()))
    aligned = (p.explode(["w1", "w2"])
                .filter(pl.col("w2").str.contains(INDIC) & ~pl.col("w1").str.contains(INDIC)))
    counts = aligned.group_by(["w2", "w1"]).len()
    tot = counts.group_by("w2").agg(pl.col("len").sum().alias("tot"))
    best = (counts.sort("len", descending=True).group_by("w2", maintain_order=True).head(1)
                  .join(tot, on="w2")
                  .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share)))
    return best.select(pl.col("w2").alias("src_tok"), pl.col("w1").alias("dst_tok"))


def learn_address_dictionary(s1, others, pairs, min_count=5, min_share=0.6):
    """Map Indic address components (mostly state names) to the Latin last component of the S1 address."""
    comp = lambda c: pl.col(c).str.split(",").list.eval(pl.element().str.strip_chars().str.to_lowercase())
    p = (pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), comp("business_address").alias("a1")), on="s1_id")
              .join(others.select(pl.col("entity_id").alias("o_id"), comp("business_address").alias("a2")), on="o_id")
              .select(pl.col("a1").list.last().alias("dst_tok"), pl.col("a2"))
              .explode("a2").filter(pl.col("a2").str.contains(INDIC)))
    counts = p.group_by(["a2", "dst_tok"]).len()
    tot = counts.group_by("a2").agg(pl.col("len").sum().alias("tot"))
    best = (counts.sort("len", descending=True).group_by("a2", maintain_order=True).head(1)
                  .join(tot, on="a2")
                  .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share)))
    return best.select(pl.col("a2").alias("src_tok"), pl.col("dst_tok"))


def transliterate_addresses(df, dictionary, col="business_address", out="addr_lat"):
    """Replace whole Indic address components with their learned Latin form."""
    parts = (df.select(pl.int_range(pl.len()).alias("_r"),
                       pl.col(col).str.split(",").list.eval(pl.element().str.strip_chars()).alias("c"))
               .explode("c").with_row_index("_o")
               .with_columns(pl.col("c").str.to_lowercase().alias("_k")))
    parts = (parts.join(dictionary, left_on="_k", right_on="src_tok", how="left")
                  .with_columns(pl.coalesce("dst_tok", "c").fill_null("").alias("c")).sort("_o")
                  .group_by("_r", maintain_order=True).agg(pl.col("c").str.join(", ")))
    return df.with_columns(parts.sort("_r")["c"].alias(out))


def transliterate_names(df, dictionary, col="business_name", out="name_lat"):
    """Replace known Indic words in `col` with their Latin form; unknown words are kept."""
    words = (df.select(pl.int_range(pl.len()).alias("_r"), _words(col).alias("w"))
               .with_columns(pl.col("w").list.eval(pl.element()).alias("w"))
               .explode("w").with_row_index("_o"))
    words = (words.join(dictionary, left_on="w", right_on="src_tok", how="left")
                  .with_columns(pl.coalesce("dst_tok", "w").alias("w")).sort("_o")
                  .group_by("_r", maintain_order=True).agg(pl.col("w").str.join(" ")))
    return df.with_columns(words.sort("_r")["w"].alias(out))
