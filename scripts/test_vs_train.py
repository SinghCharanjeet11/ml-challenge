"""Where does test differ from train? Label-free comparison of the cascade's behaviour.

Compares, test vs the 10% train holdout (same cascade model, same pruned candidate sets):
- best-candidate probability bands per record, by country and source
- how close the runner-up candidate is (second-best probability)
- S1 "twins": S1 entities sharing core name (and street) with another S1 entity
Read-only; run from code/business_entity_resolution/src on the AWS box.
"""
import polars as pl

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(60)

BANDS = [0.05, 0.2, 0.5, 0.85, 0.95]
LABELS = ["<0.05", "0.05-0.2", "0.2-0.5", "0.5-0.85", "0.85-0.95", ">=0.95"]


def per_record(sc, op):
    s = sc.sort(["io", "p"], descending=[False, True])
    top = s.group_by("io", maintain_order=True).agg(pl.col("p").first().alias("p1"),
                                                    pl.col("p").get(1, null_on_oob=True).alias("p2"),
                                                    pl.col("i1").first().alias("i1"))
    return top.with_columns(pl.col("p2").fill_null(0.0)).join(op, on="io")


def bands(df, name):
    return (df.with_columns(pl.col("p1").cut(BANDS, labels=LABELS).alias("band"))
              .group_by("band").len().with_columns((pl.col("len") / pl.col("len").sum()).round(4).alias(name))
              .select("band", name))


op_t = pl.read_parquet(f"{W}/test_op.parquet", columns=["io", "country", "src", "ad"]).with_columns(
    pl.col("country").cast(pl.String), (pl.col("ad") == "").alias("no_addr")).drop("ad")
op_r = pl.read_parquet(f"{W}/train_op.parquet", columns=["io", "country", "src", "ad"]).with_columns(
    pl.col("country").cast(pl.String), (pl.col("ad") == "").alias("no_addr")).drop("ad")
test = per_record(pl.read_parquet(f"{W}/test_scored_cascade.parquet"), op_t)
hold = per_record(pl.read_parquet(f"{W}/holdout_scored_cascade.parquet").select("io", "i1", "p"), op_r)
# the holdout only holds 10% of train records; normalise by records that reached the final model
print(f"records reaching the final model: test {test.height:,} / {op_t.height:,}   holdout {hold.height:,}")

print("\n== best-candidate probability bands (share of records that reached the final model)")
out = bands(hold, "holdout")
for c in ("US", "India", "France"):
    out = out.join(bands(test.filter(pl.col("country") == c), f"test_{c}"), on="band", how="full", coalesce=True)
for c in ("US", "India"):
    out = out.join(bands(hold.filter(pl.col("country") == c), f"hold_{c}"), on="band", how="full", coalesce=True)
print(out.sort("band"))

print("\n== same, by source and address presence (test vs holdout, share in 0.2-0.85)")
mid = (pl.col("p1") >= 0.2) & (pl.col("p1") < 0.85)
for name, d in (("holdout", hold), ("test", test)):
    print(name, d.group_by("src", "no_addr").agg(mid.mean().round(4).alias("mid_share"), pl.len()).sort("src", "no_addr").rows())

print("\n== runner-up closeness among confident-ish records (p1 >= 0.2): share with p2 >= 0.1 / >= 0.3")
for name, d in (("holdout", hold), ("test", test)):
    x = d.filter(pl.col("p1") >= 0.2)
    print(name, float((x["p2"] >= 0.1).mean()).__round__(4), float((x["p2"] >= 0.3).mean()).__round__(4),
          "| mid-band records with p2>=0.1:", float(x.filter(mid)["p2"].ge(0.1).mean()).__round__(4))

print("\n== S1 twins: share of S1 entities whose core name appears on another S1 entity (same country)")
for split in ("train", "test"):
    s1 = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1", "country", "core", "adc", "num0"]).with_columns(
        pl.col("country").cast(pl.String),
        pl.col("adc").str.replace_all(r"\d+", "").str.replace_all(r"\s+", " ").str.strip_chars().alias("street"))
    s1 = s1.filter(pl.col("core") != "")
    n_core = s1.group_by("country", "core").len().rename({"len": "n_core"})
    n_street = s1.group_by("country", "core", "street").len().rename({"len": "n_street"})
    x = s1.join(n_core, on=["country", "core"]).join(n_street, on=["country", "core", "street"])
    print(split, x.group_by("country").agg((pl.col("n_core") >= 2).mean().round(4).alias("same_name"),
                                          (pl.col("n_street") >= 2).mean().round(4).alias("same_name_same_street"),
                                          pl.len()).sort("country").rows())

print("\n== S1 entities that are the best candidate of many records (possible magnets), records per S1 among p1>=0.2")
for name, d in (("holdout", hold), ("test", test)):
    k = d.filter(pl.col("p1") >= 0.2).group_by("i1").len()
    print(name, k.select(pl.col("len").mean().round(3).alias("mean"), pl.col("len").quantile(0.99).alias("p99"),
                         (pl.col("len") >= 8).mean().round(4).alias("share>=8")).rows())
