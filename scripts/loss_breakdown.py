"""Where does the corrected (anchored) holdout lose score? Oracle fixes on a saved holdout.
    python loss_breakdown.py holdout_scored_cascade_anc.parquet 0.6"""
import sys

import polars as pl

sys.path.insert(0, ".")
from decide import assign  # noqa: E402
from io_utils import gt_pairs, load_split  # noqa: E402
from metric import macro_f05  # noqa: E402

W = "../../../work"
sc = pl.read_parquet(f"{W}/{sys.argv[1]}")
tau = float(sys.argv[2])
s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id", "ad"])
s1ids = s1.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
val = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)
ids = val["s1_id"].to_list()
truth = gt_pairs(gt).join(val, on="s1_id", how="semi")
T = truth.join(s1ids, on="s1_id").join(oids, on="o_id").select("io", "i1")


def score(pairs):
    return macro_f05(ids, pairs.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id"), truth)


k = assign(sc, tau)
base = score(k.select("io", "i1"))
fp = k.filter(~pl.col("y"))
un = pl.col("unowned")
print(f"tau={tau}: base {base:.5f}; predictions {k.height:,}; false {fp.height:,} "
      f"(non-matching records {fp.filter(un).height:,}, owned->wrong S1 {fp.filter(~un).height:,})")
print("gain if no FP from non-matching records:", round(score(k.filter(~(~pl.col('y') & un)).select('io', 'i1')) - base, 5))
print("gain if no FP owned->wrong S1          :", round(score(k.filter(~(~pl.col('y') & ~un)).select('io', 'i1')) - base, 5))
pool_true = sc.filter(pl.col("y")).select("io", "i1")
miss = pool_true.join(k.select("io", "i1"), on=["io", "i1"], how="anti")
print(f"gain if all true pairs in the pool were predicted ({miss.height:,}):",
      round(score(pl.concat([k.select("io", "i1"), miss])) - base, 5))
lost = T.join(pool_true, on=["io", "i1"], how="anti")
print(f"gain if pruning/blocking lost nothing ({lost.height:,} of {T.height:,}):",
      round(score(pl.concat([k.select("io", "i1"), lost])) - base, 5))
na = op.filter(pl.col("ad") == "").select("io")
print("no-address share: of FP", round(fp.join(na, on="io", how="semi").height / fp.height, 3),
      "| of missed-in-pool", round(miss.join(na, on="io", how="semi").height / max(miss.height, 1), 3))
print("FP by probability band:", fp.select(pl.col("p").cut([0.7, 0.85, 0.95, 0.99]).alias("b")).group_by("b").len().sort("b").rows())
