"""Turn pair probabilities into final matches.

1. Assignment: every Source 2/3 record belongs to at most one Source 1 entity (true in all
   7.6M training links), so each record keeps only its highest-probability candidate.
2. Threshold: the kept pair must have probability >= `tau`.
3. Expected-F0.5 pruning: for each Source 1 entity, sort its assigned records by probability
   and keep the prefix that maximises expected F0.5 under the model's probabilities.
"""
import numpy as np
import polars as pl


def assign(scored, tau):
    """scored: (io, i1, p). Returns the (io, i1, p) pairs kept after assignment + threshold."""
    best = (scored.sort(["io", "p"], descending=[False, True])
                  .group_by("io", maintain_order=True).head(1))
    return best.filter(pl.col("p") >= tau)


def expected_f05_prefix(kept, prior_singleton=0.0):
    """Keep, per i1, the probability-sorted prefix with the highest expected F0.5.

    Expected F0.5 of predicting the top-n records is approximated by plugging the expected
    true-positive count sum(p[:n]) and the expected number of true links
    (sum of all assigned p) into the F0.5 formula.
    """
    kept = kept.sort(["i1", "p"], descending=[False, True])
    i1 = kept["i1"].to_numpy()
    p = kept["p"].to_numpy().astype(np.float64)
    keep = np.zeros(len(p), dtype=bool)
    starts = np.flatnonzero(np.r_[True, i1[1:] != i1[:-1]])
    ends = np.r_[starts[1:], len(p)]
    for s, e in zip(starts, ends):
        ps = p[s:e]
        tp = np.cumsum(ps)
        n = np.arange(1, e - s + 1)
        total = ps.sum() + prior_singleton
        prec, rec = tp / n, tp / max(total, 1e-9)
        f = 1.25 * prec * rec / (0.25 * prec + rec + 1e-12)
        keep[s:s + int(np.argmax(f)) + 1] = True
    return kept.filter(pl.Series(keep))
