"""Token-level name/number features that tell a misspelt real record from a decoy.

A real S2/S3 record usually keeps every word of the S1 name (possibly misspelt) and may ADD
generic words ("Center", "Authority", "Dr", ...). A decoy typically REPLACES a business word
("Forestry" -> "Services", "musique" -> "pharmacie") or the legal form. The existing features
count missing/extra words but cannot tell a typo from a replacement; these can.

word_features(df) expects columns core, core_1 (record / S1 core names), legal, legal_1,
num0, num0_1 and returns the feature columns below (computed in parallel).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

WORD_FEATURES = ["w_n_missing", "w_n_added", "w_miss_best_min", "w_miss_best_mean", "w_n_replaced", "w_n_new",
                 "w_replace", "w_new_only", "legal_missing", "legal_added", "num_same_len", "num_digit_edits",
                 "num_one_digit_sub", "num_last_digit_only"]


def _one(co, c1, lo, l1, no, n1):
    b, a = set(co.split()), set(c1.split())
    missing, added = a - b, b - a
    bests = [max((fuzz.ratio(m, x) for x in b), default=0.0) for m in missing]
    replaced = sum(1 for s in bests if s < 70)
    new = sum(1 for e in added if max((fuzz.ratio(e, x) for x in a), default=0.0) < 70)
    lo_s, l1_s = set(lo.split()) if lo else set(), set(l1.split()) if l1 else set()
    if no and n1:
        same = len(no) == len(n1)
        edits = Levenshtein.distance(no, n1)
        diffpos = [i for i in range(len(no)) if no[i] != n1[i]] if same else []
        one_sub = same and len(diffpos) == 1
        last_only = one_sub and diffpos[0] == len(no) - 1
    else:
        same, edits, one_sub, last_only = False, -1, False, False
    return (len(missing), len(added), min(bests) if bests else 100.0, float(np.mean(bests)) if bests else 100.0,
            replaced, new, int(replaced > 0 and new > 0), int(replaced == 0 and new > 0),
            len(l1_s - lo_s), len(lo_s - l1_s), int(same), edits, int(one_sub), int(last_only))


def _chunk(args):
    return [_one(*r) for r in zip(*args)]


def word_features(df, workers=16, chunk=200_000):
    cols = [df[c].fill_null("").to_list() for c in ("core", "core_1", "legal", "legal_1", "num0", "num0_1")]
    jobs = [[c[i:i + chunk] for c in cols] for i in range(0, df.height, chunk)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_chunk, jobs) for r in part]
    return pl.DataFrame(rows, schema={c: pl.Float32 for c in WORD_FEATURES}, orient="row")
