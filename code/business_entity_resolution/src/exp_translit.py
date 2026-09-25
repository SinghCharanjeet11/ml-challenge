import polars as pl, time
from io_utils import gt_pairs, load_split
from translit import learn_dictionary, _words, INDIC, transliterate_names
DATA = "../../../dataset/student_resource/dataset"
t = time.time()
s1, oth, gt = load_split(DATA, "train")
d = learn_dictionary(s1, oth, gt_pairs(gt))
print("dict size", d.height, f"{time.time()-t:.0f}s")
print(d.sample(15, seed=1))
_, toth, _ = load_split(DATA, "test")
tw = toth.select(_words("business_name").alias("w")).explode("w").filter(pl.col("w").str.contains(INDIC))
cov = tw.join(d, left_on="w", right_on="src_tok", how="semi").height / tw.height
print("test Indic word tokens:", tw.height, " covered by dict:", round(cov, 4))
unk = tw.join(d, left_on="w", right_on="src_tok", how="anti").group_by("w").len().sort("len", descending=True)
print("distinct unknown:", unk.height); print(unk.head(15))
x = transliterate_names(toth.filter(pl.col("business_name").str.contains(INDIC)).head(12), d)
for a, b in x.select("business_name", "name_lat").iter_rows(): print(a, "=>", b)
d.write_parquet("../artifacts_translit_dict.parquet")
