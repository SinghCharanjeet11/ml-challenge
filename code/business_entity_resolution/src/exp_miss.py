import sys, polars as pl
from blocking import generate_candidates, record_keys
from io_utils import gt_pairs, load_split
from normalize import add_clean_columns
DATA = "../../../dataset/student_resource/dataset"
s1, oth, gt = load_split(DATA, "train")
s1 = add_clean_columns(s1.filter(pl.col("country") == "India").with_row_index("i1"))
oth = add_clean_columns(oth.filter(pl.col("country") == "India").sample(40000, seed=0).with_row_index("io"))
truth = (gt_pairs(gt).join(s1.select("i1", pl.col("entity_id").alias("s1_id")), on="s1_id")
         .join(oth.select("io", pl.col("entity_id").alias("o_id")), on="o_id").select("io", "i1"))
cand = generate_candidates(s1, oth, top_k=15, df_cap=3000, max_keys=12, budget=5000)
miss = truth.join(cand, on=["io", "i1"], how="anti")
print("recall", 1 - miss.height / truth.height, "misses", miss.height)
k1 = record_keys(s1, "i1"); ko = record_keys(oth, "io")
shared = miss.join(ko, on="io").join(k1, on=["i1", "key"], how="semi").group_by(["io","i1"]).agg(pl.col("key"))
miss = miss.join(shared, on=["io","i1"], how="left")
# is true owner present but outranked?
rank = cand.with_columns(pl.int_range(pl.len()).over("io").alias("r")).group_by("io").agg(pl.col("block_score").max().alias("best"))
m = (miss.join(s1.select("i1","business_name","business_address"), on="i1")
         .join(oth.select("io",pl.col("business_name").alias("oname"),pl.col("business_address").alias("oaddr"),"src"), on="io"))
print("misses with zero shared keys:", m.filter(pl.col("key").is_null()).height)
pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
for r in m.head(40).iter_rows(named=True):
    print(f"S1: {r['business_name']} | {r['business_address']}\n O{r['src']}: {r['oname']} | {r['oaddr']}\n   shared={r['key']}\n")
