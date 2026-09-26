"""Macro F0.5 per S1 entity, same as the leaderboard."""
import polars as pl


def macro_f05(s1_ids, pred_pairs, true_pairs):
    """pred_pairs / true_pairs have columns (s1_id, o_id).

    Empty truth + empty prediction = 1.0, one side empty = 0.0.
    """
    base = pl.DataFrame({"s1_id": pl.Series(s1_ids, dtype=pl.String)})
    pred = pred_pairs.select("s1_id", "o_id").unique()
    true = true_pairs.select("s1_id", "o_id").unique()
    tp = pred.join(true, on=["s1_id", "o_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pred.group_by("s1_id").len().rename({"len": "npred"})
    ntrue = true.group_by("s1_id").len().rename({"len": "ntrue"})
    df = (base.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
              .join(ntrue, on="s1_id", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("npred")
    r = pl.col("tp") / pl.col("ntrue")
    f = (pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    return float(df.select(f.mean()).item())


if __name__ == "__main__":
    # Worked example from the problem statement: expect 0.714.
    pred = pl.DataFrame({"s1_id": ["S1-1"] * 3, "o_id": ["S2-47", "S2-193", "S3-812"]})
    true = pl.DataFrame({"s1_id": ["S1-1"] * 2, "o_id": ["S2-47", "S3-812"]})
    assert abs(macro_f05(["S1-1"], pred, true) - 0.7142857) < 1e-6
    # Singleton predicted empty = 1.0, singleton with a match = 0.0, missed entity = 0.0.
    assert macro_f05(["S1-9"], pred.head(0), true.head(0)) == 1.0
    assert macro_f05(["S1-9"], pl.DataFrame({"s1_id": ["S1-9"], "o_id": ["S2-1"]}), true.head(0)) == 0.0
    assert macro_f05(["S1-1"], pred.head(0), true) == 0.0
    print("metric tests pass")
