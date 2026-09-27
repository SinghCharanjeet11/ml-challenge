"""How many matches should the new model submit? Read-only.

Both models are scored on the same corrected holdout. For each, find the number of matches that
maximises the test-density estimate, and apply the ratio to the reference submission's count
(5,762,921 matches, LB 0.9808). Also prints both models at equal match counts.

    python ../../../scripts/count_shift.py cascade_f RR      # run from src/
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, ".")
from decide import assign  # noqa: E402
from io_utils import gt_pairs, load_split  # noqa: E402
from metric import macro_f05  # noqa: E402

W = "../../../work"
REF = 5_762_921
base, new = sys.argv[1], sys.argv[2]
s1ids = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"]).rename({"entity_id": "s1_id"})
oids = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"]).rename({"entity_id": "o_id"})
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
val_ids = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)["s1_id"]
truth = gt_pairs(gt).filter(pl.col("s1_id").is_in(val_ids.implode()))
ids = val_ids.to_list()


def curve(name):
    h = pl.read_parquet(f"{W}/holdout_scored_{name}.parquet")
    best = assign(h.select("io", "i1", "p", "unowned"), 0.0)
    ps = np.sort(best["p"].to_numpy())[::-1]
    out = {}
    for q in np.arange(0.74, 0.8451, 0.0025):       # match count as a share of the records
        n = int(q * len(ps))
        tau = float(ps[min(n, len(ps) - 1)])
        k = best.filter(pl.col("p") >= tau)
        pred = k.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
        dup = k.filter(pl.col("unowned")).join(s1ids, on="i1").join(oids, on="io").select("s1_id", pl.col("o_id") + "_dup")
        out[round(q, 4)] = (k.height, tau, macro_f05(ids, pred, truth), macro_f05(ids, pl.concat([pred, dup]), truth))
    return out


cb, cn = curve(base), curve(new)
print(f"{'share':>6} | {base}: matches tau F0.5 test-density | {new}: matches tau F0.5 test-density")
for q in cb:
    b, n = cb[q], cn[q]
    print(f"{q:6.4f} | {b[0]:9,} {b[1]:.4f} {b[2]:.5f} {b[3]:.5f} | {n[0]:9,} {n[1]:.4f} {n[2]:.5f} {n[3]:.5f}")
ob = max(cb.values(), key=lambda r: r[3])
on = max(cn.values(), key=lambda r: r[3])
print(f"best test-density: {base} {ob[3]:.5f} at {ob[0]:,} matches; {new} {on[3]:.5f} at {on[0]:,} matches")
print(f"count ratio {on[0] / ob[0]:.4f} -> test matches {int(REF * on[0] / ob[0]):,} (reference {REF:,})")
