"""Do the new word features separate real from false among the holdout's unsure records, and how
often do test's unsure records look like replacements? Read-only."""
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from word_features import WORD_FEATURES, word_features  # noqa: E402

W = "../../../work"
pl.Config.set_tbl_width_chars(250)
TXT = ["core", "legal", "num0"]


def with_text(d, split):
    s1 = pl.read_parquet(f"{W}/{split}_s1p.parquet", columns=["i1"] + TXT).rename({c: f"{c}_1" for c in TXT})
    op = pl.read_parquet(f"{W}/{split}_op.parquet", columns=["io"] + TXT)
    return d.join(op, on="io").join(s1, on="i1")


mid = (pl.col("p") >= 0.2) & (pl.col("p") < 0.85)
hold = with_text(pl.read_parquet(f"{W}/holdout_midband_profile.parquet").filter(mid), "train")
hold = pl.concat([hold, word_features(hold)], how="horizontal")
test = with_text(pl.read_parquet(f"{W}/test_midband_profile.parquet").filter(mid).sample(150_000, seed=1), "test")
test = pl.concat([test, word_features(test)], how="horizontal")

y = hold["y"].to_numpy()
print(f"holdout unsure records: {len(y):,} (true share {y.mean():.3f})")
print(f"model p alone, AUC true-vs-false inside the band: {roc_auc_score(y, hold['p'].to_numpy()):.3f}")
for c in WORD_FEATURES:
    x = hold[c].to_numpy()
    auc = roc_auc_score(y, x)
    print(f"  {c:20s} AUC {max(auc, 1 - auc):.3f}  mean true {x[y].mean():7.3f}  false {x[~y].mean():7.3f}  "
          f"test US {test.filter(pl.col('country') == 'US')[c].mean():7.3f}  India {test.filter(pl.col('country') == 'India')[c].mean():7.3f}  "
          f"France {test.filter(pl.col('country') == 'France')[c].mean():7.3f}")
