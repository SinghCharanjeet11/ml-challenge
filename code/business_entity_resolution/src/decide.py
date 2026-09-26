"""Final decision: each S2/S3 record goes to at most one S1 (its best candidate), and only
if the probability clears tau.
"""
import polars as pl


def assign(scored, tau):
    """scored: (io, i1, p). Returns the (io, i1, p) pairs kept after assignment + threshold."""
    best = (scored.sort(["io", "p"], descending=[False, True])
                  .group_by("io", maintain_order=True).head(1))
    return best.filter(pl.col("p") >= tau)

