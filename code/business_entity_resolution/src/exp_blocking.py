"""Experiment: blocking recall / size on a slice of the training data."""
import sys
import time

import polars as pl

from blocking import generate_candidates
from io_utils import gt_pairs, load_split
from normalize import add_clean_columns
from translit import (INDIC, learn_address_dictionary, learn_dictionary, transliterate_addresses,
                      transliterate_names)

DATA = "../../../dataset/student_resource/dataset"
COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "India"
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
CONFIGS = [(2000, 20, 24, 20000), (1000, 20, 24, 10000)]

t = time.time()
s1, oth, gt = load_split(DATA, "train")
pairs = gt_pairs(gt)
nd, ad = learn_dictionary(s1, oth, pairs), learn_address_dictionary(s1, oth, pairs)
s1 = s1.filter(pl.col("country") == COUNTRY).with_row_index("i1")
oth = oth.filter(pl.col("country") == COUNTRY).sample(NQ, seed=0).with_row_index("io")
oth = transliterate_addresses(transliterate_names(oth, nd), ad)
s1, oth = add_clean_columns(s1), add_clean_columns(oth)
print(f"prep {time.time()-t:.1f}s  s1={s1.height} queries={oth.height}", flush=True)

truth = (pairs.join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
              .join(oth.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", "i1"))
cat = oth.select("io", pl.col("business_name").str.contains(INDIC).alias("indic"),
                 (pl.col("business_address") == "").alias("no_addr"))
truth = truth.join(cat, on="io")
print("true links among queries:", truth.height)

for cap, k, mk, bud in CONFIGS:
    t = time.time()
    cand = generate_candidates(s1, oth, top_k=k, df_cap=cap, max_keys=mk, budget=bud)
    rk = cand.with_columns(pl.int_range(pl.len()).over("io").alias("rank"))
    r = truth.join(rk.select("io", "i1", "rank"), on=["io", "i1"], how="left")
    summ = r.group_by(["indic", "no_addr"]).agg(pl.len().alias("n"), pl.col("rank").is_not_null().mean().alias("recall"),
                                                (pl.col("rank") == 0).mean().alias("top1")).sort(["indic", "no_addr"])
    print(f"cap={cap} k={k} max_keys={mk} budget={bud}: {time.time()-t:.1f}s cands={cand.height} "
          f"({cand.height/oth.height:.1f}/q) recall={r['rank'].is_not_null().mean():.4f} top1={(r['rank']==0).mean():.4f}")
    print(summ, flush=True)
