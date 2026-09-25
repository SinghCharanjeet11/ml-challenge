"""Loading source files and writing submission files."""
import os

import polars as pl


def read_tsv(path):
    """Read a challenge TSV with every column as a string.

    Quote handling is off (some names contain literal quote characters) and empty
    fields stay as empty strings, so "NA" or "null" business names survive intact.
    """
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    df = df.with_columns(pl.all().fill_null(""))
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f) - 1
    assert df.height == n_lines, f"{path}: parsed {df.height} rows but file has {n_lines}"
    return df


def load_split(data_dir, split):
    """Return (s1, others, ground_truth or None) for 'train' or 'test'.

    `others` stacks Source 2 and Source 3 with an integer `src` column.
    """
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv")).with_columns(pl.lit(2, pl.Int8).alias("src"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv")).with_columns(pl.lit(3, pl.Int8).alias("src"))
    gt_path = os.path.join(d, f"{split}_ground_truth.tsv")
    gt = read_tsv(gt_path) if os.path.exists(gt_path) else None
    return s1, pl.concat([s2, s3]), gt


def gt_pairs(gt):
    """Explode the ground truth into (s1_id, o_id) string pairs."""
    return (gt.filter(pl.col("matched_entity_ids") != "")
              .with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "o_id"}))


def write_id_lists(path, s1_ids, pairs, id_col, header_col):
    """Write one row per S1 id with a comma-joined list of matched ids (LF endings, no quoting)."""
    lists = pairs.group_by("s1_id").agg(pl.col(id_col).unique().sort().str.join(","))
    out = (pl.DataFrame({"s1_id": s1_ids}).join(lists, on="s1_id", how="left")
             .with_columns(pl.col(id_col).fill_null("")))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1, ids in out.iter_rows():
            f.write(f"{s1}\t{ids}\n")


def write_id_lists_chunked(path, s1_ids, pairs, o_ids, header_col, chunk=200_000):
    """Low-memory write_id_lists for integer-id pairs.

    s1_ids: (i1, s1_id) in output order with i1 = row number; pairs: (i1, io);
    o_ids: (io, o_id). Rows are written in slices of `chunk` Source 1 entities so only one
    slice of pairs is ever joined to its string ids.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pairs = pairs.select("i1", "io").sort("i1")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for lo in range(0, s1_ids.height, chunk):
            ids = s1_ids.slice(lo, chunk)
            part = pairs.filter(pl.col("i1").is_between(lo, lo + chunk - 1))
            lists = (part.join(o_ids, on="io").group_by("i1")
                         .agg(pl.col("o_id").unique().sort().str.join(",")))
            out = ids.join(lists, on="i1", how="left").sort("i1").with_columns(pl.col("o_id").fill_null(""))
            for s1, o in out.select("s1_id", "o_id").iter_rows():
                f.write(f"{s1}\t{o}\n")
