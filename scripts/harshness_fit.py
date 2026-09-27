"""Calibrate test harshness: count every false match coming from a non-matching record k times
(k ~ how many more such records test has, and how much harder) and see where the corrected
holdout's threshold curve peaks. Pick k so the reference model's peak matches the leaderboard
(old cascade: best at tau 0.85), then use that k for the new models.
    python harshness_fit.py holdout_scored_cascade_anc.parquet [more files...]"""
import sys

import polars as pl

sys.path.insert(0, ".")
from decide import assign  # noqa: E402
from io_utils import gt_pairs, load_split  # noqa: E402
from metric import macro_f05  # noqa: E402

W = "../../../work"
s1 = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
s1ids = s1.select("i1", pl.col("entity_id").alias("s1_id"))
oids = op.select("io", pl.col("entity_id").alias("o_id"))
_, _, gt = load_split("../../../dataset/student_resource/dataset", "train")
val = s1ids.filter(pl.col("i1").hash(seed=5) % 10 == 0)
ids = val["s1_id"].to_list()
truth = gt_pairs(gt).join(val, on="s1_id", how="semi")
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
for f in sys.argv[1:]:
    sc = pl.read_parquet(f"{W}/{f}")
    print(f"\n== {f}")
    for k in (2, 3, 4, 6, 8):
        row = []
        for tau in TAUS:
            kept = assign(sc, tau)
            pred = kept.join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
            fps = kept.filter(pl.col("unowned")).join(s1ids, on="i1").join(oids, on="io").select("s1_id", "o_id")
            extra = [fps.with_columns(pl.col("o_id") + f"_d{j}") for j in range(k - 1)]
            row.append(macro_f05(ids, pl.concat([pred] + extra), truth))
        best = TAUS[max(range(len(TAUS)), key=lambda i: row[i])]
        print(f"k={k}: " + "  ".join(f"{t}:{v:.4f}" for t, v in zip(TAUS, row)) + f"   -> best tau {best}")
