"""Run the trained pipeline on the test split and write both submission files.

Outputs (tab-separated, LF line endings, one row per test Source 1 entity):
  output/candidate_pairs.tsv  - the exact candidate set the model scores (top-K per record)
  output/matching_results.tsv - final matches after assignment + threshold
"""
import argparse
import json

import lightgbm as lgb
import polars as pl

from decide import assign
from features import FEATURES
from io_utils import load_split, write_id_lists_chunked
from pipeline import block_cached, featurise_chunks, log, prep_cached

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../../dataset/student_resource/dataset")
ap.add_argument("--work", default="../../../work")
ap.add_argument("--out", default="../../../output")
ap.add_argument("--tau", type=float, default=None)
args = ap.parse_args()

cfg = json.load(open(f"{args.work}/config.json"))
tau = args.tau if args.tau is not None else cfg["tau"]
nd = pl.read_parquet(f"{args.work}/name_dict.parquet")
ad = pl.read_parquet(f"{args.work}/addr_dict.parquet")
booster = lgb.Booster(model_file=f"{args.work}/model.txt")

prep_cached(args.work, "test", lambda: load_split(args.data, "test")[:2], nd, ad)
cand = block_cached(args.work, "test")
s1p, op = prep_cached(args.work, "test", None, nd, ad)
log(f"test: s1={s1p.height:,} others={op.height:,} countries={sorted(s1p['country'].unique().to_list())}")
log(f"candidates: {cand.height:,}")

# Score chunk by chunk and keep only (io, i1, p): the full feature table for ~100M
# test pairs would need ~20 GB.
scored = []
for n, feat in enumerate(featurise_chunks(cand, s1p, op, keep_k=cfg["keep_k"])):
    p = booster.predict(feat.select(FEATURES).to_numpy())
    scored.append(feat.select("io", "i1").with_columns(pl.Series("p", p, dtype=pl.Float32)))
    if n % 5 == 0:
        log(f"scored chunk {n}: {sum(s.height for s in scored):,} pairs so far")
del cand
scored = pl.concat(scored)
log(f"scored {scored.height:,} pairs")

s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
del s1p, op
assert (s1ids["i1"] == pl.int_range(s1ids.height, eager=True)).all()

write_id_lists_chunked(f"{args.out}/candidate_pairs.tsv", s1ids, scored, oids, "candidate_entity_ids")
log("wrote candidate_pairs.tsv")
kept = assign(scored, tau)
write_id_lists_chunked(f"{args.out}/matching_results.tsv", s1ids, kept, oids, "matched_entity_ids")
matches = kept.select("i1")
all_s1 = s1ids["s1_id"]
n_matched = matches["i1"].n_unique()
log(f"tau={tau}: {matches.height:,} matched pairs; {n_matched:,} of {all_s1.len():,} S1 entities have a match "
    f"({1 - n_matched / all_s1.len():.1%} predicted singletons)")
