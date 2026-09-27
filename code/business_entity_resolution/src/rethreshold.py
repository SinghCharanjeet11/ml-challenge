"""Rewrite a matching file from saved test scores at other thresholds (no re-scoring).

    python rethreshold.py cascade 0.85,0.9          # one threshold for every record
    python rethreshold.py cascade 0.85 --no_addr 0.2  # records without an address use their own

Uses work/test_scored_<model>.parquet and writes output/matching_results_<model>_tau<t>[...].tsv.

Why a separate threshold for records without an address: in train, 4.4% of matched S2/S3
records have no address but only 0.3% of unmatched ones do, and test shows the same mix. The
extra non-matching records in test therefore almost all carry an address, so the strict test
threshold is only needed for records that have one.
"""
import argparse

import polars as pl

from decide import assign
from io_utils import write_id_lists_chunked

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("taus")
ap.add_argument("--no_addr", type=float, default=None, help="threshold for records without an address")
args = ap.parse_args()

W, OUT = "../../../work", "../../../output"
scored = pl.read_parquet(f"{W}/test_scored_{args.model}.parquet")
s1 = pl.read_parquet(f"{W}/test_s1p.parquet", columns=["i1", "entity_id"]).rename({"entity_id": "s1_id"})
op = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "entity_id", "ad"])
no_addr = op.filter(pl.col("ad") == "").select("io")
op = op.select("io", pl.col("entity_id").alias("o_id"))
for tau in [float(t) for t in args.taus.split(",")]:
    kept = assign(scored, 0.0 if args.no_addr is not None else tau)
    suffix = f"tau{tau:.2f}"
    if args.no_addr is not None:
        flag = kept.join(no_addr.with_columns(pl.lit(True).alias("na")), on="io", how="left")
        kept = flag.filter(pl.when(pl.col("na").is_not_null()).then(pl.col("p") >= args.no_addr)
                           .otherwise(pl.col("p") >= tau)).drop("na")
        suffix += f"_noaddr{args.no_addr:.2f}"
    write_id_lists_chunked(f"{OUT}/matching_results_{args.model}_{suffix}.tsv", s1, kept, op, "matched_entity_ids")
    print(f"{suffix}: {kept.height:,} matches", flush=True)
