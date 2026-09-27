"""Print holdout records the cascade is unsure about (0.2 <= p < 0.85, record has an address):
real matches vs non-matches, with the raw S1 and record text. Read-only."""
import sys

import polars as pl

sys.path.insert(0, ".")
from io_utils import load_split  # noqa: E402

W = "../../../work"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 28
s1r, oth, _ = load_split("../../../dataset/student_resource/dataset", "train")
s1p = pl.read_parquet(f"{W}/train_s1p.parquet", columns=["i1", "entity_id"])
op = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "entity_id"])
h = pl.read_parquet(f"{W}/holdout_midband_profile.parquet").filter((pl.col("p") >= 0.2) & (pl.col("p") < 0.85))
h = (h.join(s1p, on="i1")
      .join(s1r.select("entity_id", pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")),
            on="entity_id").drop("entity_id")
      .join(op, on="io").join(oth.select("entity_id", "business_name", "business_address"), on="entity_id"))
for lab, flt in (("TRUE: real match, model unsure", pl.col("y")), ("FALSE: not a match, model unsure", ~pl.col("y"))):
    print(f"\n######## {lab}")
    for r in h.filter(flt).sample(N, seed=7).iter_rows(named=True):
        print(f"[{r['country']} p={r['p']:.2f}] S1 : {r['s1_name']} | {r['s1_addr']}")
        print(f"               rec: {r['business_name']} | {r['business_address']}")
