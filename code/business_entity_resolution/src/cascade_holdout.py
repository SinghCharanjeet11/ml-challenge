"""Holdout predictions of the saved cascade (filter + final model) and per-segment thresholds.

Recomputes the filter probability for every train pair with the saved fold models (each pair
scored by the fold model that did not train on it), prunes and builds competition features the
same way as stage2.py, scores the 10% holdout with work/model_final.txt and reports the best
threshold separately for records with and without an address.
Writes work/holdout_scored_cascade.parquet (io, i1, p, y, no_addr).
"""
import glob
import json
import sys

import lightgbm as lgb
import polars as pl

from decide import assign
from io_utils import gt_pairs, load_split
from metric import macro_f05
from pipeline import log

W = "../../../work"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
cfg = json.load(open(f"{W}/config_cascade.json"))
FILTER, FINAL = cfg["filter_features"], cfg["final_features"]
COMP = FINAL[FINAL.index("p1"):]

s1p = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
op = pl.read_parquet(f"{W}/{SPLIT}_op.parquet", columns=["io", "entity_id", "src", "ad"])
s1ids = s1p.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
pairs = gt_pairs(gt)
owner = pairs.join(s1ids, on="s1_id").join(oids, on="o_id").select("io", pl.col("i1").alias("owner"))
parent = (pl.read_parquet(f"{W}/{SPLIT}_parent.parquet") if SPLIT != "train"
          else pl.DataFrame(schema={"io": pl.UInt32, "parent": pl.UInt32}))
models = [lgb.Booster(model_file=f"{W}/model_filter_{k}.txt") for k in (0, 1)]
final = lgb.Booster(model_file=f"{W}/model_final.txt")
parts = sorted(glob.glob(f"{W}/feat_all_{SPLIT}/part_*.parquet"))


def matrix(d, cols):
    return d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


p1 = []
for part in parts:
    d = pl.read_parquet(part, columns=["io", "i1"] + FILTER).join(owner, on="io", how="left").join(parent, on="io", how="left")
    key = pl.coalesce("owner", "parent", "io")
    d = d.with_columns((pl.col("i1") == pl.col("owner")).fill_null(False).alias("y"),
                       (key.hash(seed=7) % 2).alias("fold"), (key.hash(seed=5) % 10 == 0).alias("val"))
    for k in (0, 1):
        dk = d.filter(pl.col("fold") == k)  # fold k was predicted by the model trained on the other fold
        p1.append(dk.select("io", "i1", "y", "val").with_columns(
            pl.Series("p1", models[k].predict(matrix(dk, FILTER)), dtype=pl.Float32)))
p1 = pl.concat(p1)
p1 = p1.filter((pl.col("p1") >= cfg["prune"])
               & (pl.col("p1").rank("ordinal", descending=True).over("io") <= cfg["max_per_record"]))
log(f"pruned train pairs: {p1.height:,}")

q = pl.col("p1")
c = p1.join(op.select("io", "src"), on="io").with_columns(
    q.rank("ordinal", descending=True).over("io").alias("q_rank"),
    (q.max().over("io") - q).alias("q_gap"),
    q.sort(descending=True).get(1, null_on_oob=True).over("io").fill_null(0).alias("q_2nd"),
    pl.len().over("io").alias("q_n")).with_columns((pl.col("q_rank") == 1).alias("q_best"))
best = pl.when(pl.col("q_best")).then(q).otherwise(None)
c = c.with_columns(
    q.rank("ordinal", descending=True).over("i1").alias("s_rank"), pl.len().over("i1").alias("s_n"),
    best.count().over("i1").alias("s_n_best"), (best > 0.5).sum().over("i1").alias("s_n_strong"),
    best.sum().over("i1").fill_null(0).alias("s_sum"), best.max().over("i1").fill_null(0).alias("s_max"),
    best.count().over(["i1", "src"]).alias("s_n_best_src"),
    (best > 0.5).sum().over(["i1", "src"]).alias("s_n_strong_src"),
    best.max().over(["i1", "src"]).fill_null(0).alias("s_max_src"))
c = c.with_columns((pl.col("s_max") - q).alias("s_gap"), (pl.col("s_max_src") - q).alias("s_gap_src"))
hold = c.filter(pl.col("val")).drop("src").sort("io")
log(f"holdout pairs: {hold.height:,}")

scored = []
for part in parts:
    d = pl.read_parquet(part, columns=["io", "i1"] + FINAL[:FINAL.index("p1")])
    io = hold["io"]
    lo, hi = io.search_sorted(d["io"].min(), "left"), io.search_sorted(d["io"].max(), "right")
    d = d.join(hold.slice(lo, hi - lo), on=["io", "i1"])
    if d.height:
        scored.append(d.select("io", "i1", "y").with_columns(pl.Series("p", final.predict(matrix(d, FINAL)))))
sc = pl.concat(scored).join(op.select("io", (pl.col("ad") == "").alias("no_addr")), on="io")
sc.write_parquet(f"{W}/holdout_scored_cascade_{SPLIT}_oldmodel.parquet")

val_ids = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)
truth = pairs.join(val_ids, on="s1_id", how="semi")
ids = val_ids["s1_id"].to_list()


def f05(t_addr, t_noaddr):
    b = sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
    k = b.filter(pl.when(pl.col("no_addr")).then(pl.col("p") >= t_noaddr).otherwise(pl.col("p") >= t_addr))
    pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
    return macro_f05(ids, pred, truth)


log(f"holdout records without address: {int(sc.filter(pl.col('y'))['no_addr'].sum()):,} true pairs")
for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
    log(f"single tau={t}: {f05(t, t):.5f}")
for tn in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.85):
    log(f"tau_addr=0.2 tau_noaddr={tn}: {f05(0.2, tn):.5f}   |   tau_addr=0.85 tau_noaddr={tn}: {f05(0.85, tn):.5f}")
