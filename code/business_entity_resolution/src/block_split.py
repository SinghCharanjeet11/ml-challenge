"""Prepare and block one split in its own process, caching the results in `work`.

Blocking is the memory peak of the pipeline, so it runs before train.py / predict.py
(which then load the cached frames) rather than alongside their other data.
Needs work/name_dict.parquet and work/addr_dict.parquet (written by learn_dicts below).
"""
import argparse
import os

import polars as pl

from io_utils import load_split
from pipeline import block_cached, learn_dictionaries, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test"])
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
args = ap.parse_args()
os.makedirs(args.work, exist_ok=True)

nd_path, ad_path = f"{args.work}/name_dict.parquet", f"{args.work}/addr_dict.parquet"
if not (os.path.exists(nd_path) and os.path.exists(ad_path)):
    nd, ad = learn_dictionaries(args.data)
    nd.write_parquet(nd_path)
    ad.write_parquet(ad_path)
nd, ad = pl.read_parquet(nd_path), pl.read_parquet(ad_path)
prep_cached(args.work, args.split, lambda: load_split(args.data, args.split)[:2], nd, ad)
log("prepared")
cand = block_cached(args.work, args.split)
log(f"{args.split} candidates: {cand.height:,}")
