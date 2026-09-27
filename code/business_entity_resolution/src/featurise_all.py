"""Features for every candidate pair of a split, written to work/feat_all_<split>/ in parts.

Needs `block_split.py <split> --blocking v2` first.
"""
import argparse
import os
import shutil

import polars as pl

from pipeline import block_cached, featurise_chunks, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test"])
ap.add_argument("--work", default="../../../work")
args = ap.parse_args()

nd = pl.read_parquet(f"{args.work}/name_dict.parquet")
ad = pl.read_parquet(f"{args.work}/addr_dict.parquet")
s1p, op = prep_cached(args.work, args.split, None, nd, ad)
cand = block_cached(args.work, args.split, "v2")
log(f"{args.split}: {cand.height:,} candidate pairs")

out_dir = f"{args.work}/feat_all_{args.split}"
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir)
for n, f in enumerate(featurise_chunks(cand, s1p, op, keep_k=0)):
    f.write_parquet(f"{out_dir}/part_{n:04d}.parquet")
    if n % 5 == 0:
        log(f"chunk {n} done")
log("all features written")
