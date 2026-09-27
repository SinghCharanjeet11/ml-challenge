"""Feature profile of 'unsure' records (best candidate p in [0.2, 0.85), record has an address):
test vs holdout-true vs holdout-false. The test excess over a holdout-like mix is the decoy
signature. Read-only; run from code/business_entity_resolution/src on the AWS box."""
import glob

import polars as pl

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(80)
F = ["num0_eq", "hn_near", "hn0_prefix", "hn_min_logdiff", "n_num_o", "n_num_1", "num_jacc", "core_eq", "core_tsort",
     "nm_ratio", "core_jw", "ad_tset", "ad_partial", "extra_tok_o", "extra_tok_1", "extra_all_o", "legal_conflict",
     "legal_eq", "tfidf_sim", "n_shared", "block_score", "webname", "alias", "indic", "src", "n_cand_o", "margin_2nd",
     "len_core_o", "len_core_1", "first_tok_eq"]


def best_mid(sc, op):
    b = sc.sort(["io", "p"], descending=[False, True]).group_by("io", maintain_order=True).head(1)
    return b.join(op, on="io").filter(~pl.col("no_addr"))


def attach(best, parts):
    best = best.sort("io")
    io = best["io"]
    out = []
    for part in parts:
        d = pl.read_parquet(part, columns=["io", "i1"] + F)
        lo, hi = io.search_sorted(d["io"].min(), "left"), io.search_sorted(d["io"].max(), "right")
        if hi > lo:
            out.append(best.slice(lo, hi - lo).join(d, on=["io", "i1"]))
    return pl.concat(out)


def op_of(split):
    return pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io", "country", "ad"]).with_columns(
        pl.col("country").cast(pl.String), (pl.col("ad") == "").alias("no_addr")).drop("ad")


test = attach(best_mid(pl.read_parquet(f"{W}/test_scored_cascade.parquet"), op_of("test")),
              sorted(glob.glob(f"{W}/feat_all_test/part_*.parquet")))
hold = attach(best_mid(pl.read_parquet(f"{W}/holdout_scored_cascade.parquet"), op_of("train")),
              sorted(glob.glob(f"{W}/feat_all_train/part_*.parquet")))
band = [(0.2, 0.5), (0.5, 0.85), (0.85, 0.95)]


def summary(d, name):
    return d.select([pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).mean().round(3).alias(c) for c in F]).with_columns(
        pl.lit(name).alias("group"))


rows = []
for lo, hi in band:
    m = (pl.col("p") >= lo) & (pl.col("p") < hi)
    rows += [summary(hold.filter(m & pl.col("y")), f"hold_true {lo}-{hi}"),
             summary(hold.filter(m & ~pl.col("y")), f"hold_false {lo}-{hi}")]
    for c in ("US", "India", "France"):
        rows.append(summary(test.filter(m & (pl.col("country") == c)), f"test_{c} {lo}-{hi}"))
t = pl.concat(rows).select(["group", "n"] + F)
with pl.Config(tbl_cols=40):
    for chunk in (F[:10], F[10:20], F[20:]):
        print(t.select(["group", "n"] + chunk))
# precision of the holdout mid band (how many of the holdout's unsure records are real)
for lo, hi in band:
    m = (pl.col("p") >= lo) & (pl.col("p") < hi)
    x = hold.filter(m)
    print(f"holdout {lo}-{hi}: {x.height:,} records with address, true share {float(x['y'].mean()):.3f}")
test.write_parquet(f"{W}/test_midband_profile.parquet")
hold.write_parquet(f"{W}/holdout_midband_profile.parquet")
