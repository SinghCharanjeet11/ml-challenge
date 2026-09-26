"""How many true links do we gain with a bigger top_k and/or the tfidf pass? (train only)"""
import sys
import time

import polars as pl

from io_utils import gt_pairs, load_split
from pipeline import BLOCK_PARAMS
from blocking import generate_candidates
from tfidf_blocking import tfidf_candidates

W = "../../../work"
DATA = "../../../dataset/student_resource/dataset"
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 20

s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id", "country", "nm", "ad", "core"])
op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id", "country", "nm", "ad", "core"])
_, _, gt = load_split(DATA, "train")
truth = (gt_pairs(gt).join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
                     .join(op.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", "i1"))
print(f"true links {truth.height:,}", flush=True)

t = time.time()
params = {**BLOCK_PARAMS, "top_k": TOPK}
kc = generate_candidates(s1, op, **params)
kc = kc.with_columns(pl.col("block_score").rank("ordinal", descending=True).over("io").alias("r"))
kc.write_parquet(f"{W}/train_cand_top{TOPK}.parquet")
print(f"key blocking top{TOPK}: {kc.height:,} pairs, {time.time() - t:.0f}s", flush=True)

t = time.time()
tc = tfidf_candidates(s1, op)
tc.write_parquet(f"{W}/train_tfidf.parquet")
print(f"tfidf: {tc.height:,} pairs, {time.time() - t:.0f}s", flush=True)


def recall(c):
    return truth.join(c.select("io", "i1").unique(), on=["io", "i1"], how="semi").height / truth.height


k10 = kc.filter(pl.col("r") <= 10)
for name, c in [("keys top10", k10), (f"keys top{TOPK}", kc),
                ("tfidf top10", tc), ("tfidf top5", tc.filter(pl.col("tfidf_sim").rank("ordinal", descending=True).over("io") <= 5)),
                ("keys top10 + tfidf top10", pl.concat([k10.select("io", "i1"), tc.select("io", "i1")])),
                (f"keys top{TOPK} + tfidf top10", pl.concat([kc.select("io", "i1"), tc.select("io", "i1")]))]:
    n = c.select("io", "i1").unique().height
    print(f"{name:28s} recall={recall(c):.4f} pairs={n:,} ({n / op.height:.1f}/query)", flush=True)
