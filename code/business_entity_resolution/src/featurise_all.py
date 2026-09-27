"""Features for every candidate pair of a split, written to work/feat_all_<split>/ in parts.

Needs `block_split.py <split> --blocking v2` first. With --score, each part is also scored with
the current model (work/model.txt) and written with a `p0` column, and for test we write one
matching file per threshold in --taus (used to probe the leaderboard).
"""
import argparse
import json
import os
import shutil

import lightgbm as lgb
import polars as pl

from decide import assign
from io_utils import write_id_lists_chunked
from pipeline import block_cached, featurise_chunks, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test", "train_aug", "train_aug2"])
ap.add_argument("--work", default="../../../work")
ap.add_argument("--out", default="../../../output")
ap.add_argument("--score", action="store_true")
ap.add_argument("--taus", default="0.2,0.3,0.4,0.5")
args = ap.parse_args()

nd = pl.read_parquet(f"{args.work}/name_dict.parquet")
ad = pl.read_parquet(f"{args.work}/addr_dict.parquet")
s1p, op = prep_cached(args.work, args.split, None, nd, ad)
cand = block_cached(args.work, args.split, "v2")
log(f"{args.split}: {cand.height:,} candidate pairs")

if args.score:
    cfg = json.load(open(f"{args.work}/config.json"))
    booster = lgb.Booster(model_file=f"{args.work}/model.txt")

out_dir = f"{args.work}/feat_all_{args.split}"
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir)
scored = []
for n, f in enumerate(featurise_chunks(cand, s1p, op, keep_k=0)):
    if args.score:
        p = booster.predict(f.select(cfg["features"]).to_numpy())
        f = f.with_columns(pl.Series("p0", p, dtype=pl.Float32))
        scored.append(f.select("io", "i1", "p0"))
    f.write_parquet(f"{out_dir}/part_{n:04d}.parquet")
    if n % 5 == 0:
        log(f"chunk {n} done")
del cand
log("all features written")

if args.score and args.split == "test":
    scored = pl.concat(scored).rename({"p0": "p"})
    scored.write_parquet(f"{args.work}/test_scored_v2.parquet")
    s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
    oids = op.select("io", pl.col("entity_id").alias("o_id"))
    for tau in [float(t) for t in args.taus.split(",")]:
        kept = assign(scored, tau)
        write_id_lists_chunked(f"{args.out}/matching_results_tau{tau:.2f}.tsv", s1ids, kept, oids,
                               "matched_entity_ids")
        log(f"tau={tau:.2f}: {kept.height:,} matches written")
