
import pandas as pd


def add_rank_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Adds rank columns for each score column in cols.
    Rank 1 = highest score.
    """
    out = df.copy()
    for c in cols:
        out[f"{c}_rank"] = out[c].rank(method="min", ascending=False).astype(int)
    return out


def classify_rank_shifts(
    df: pd.DataFrame,
    supportive_rank_col: str,
    suppressive_rank_col: str,
    stable_threshold: int = 20,
    shift_threshold: int = 200,
) -> pd.DataFrame:
    """
    Uses delta = supportive_rank - suppressive_rank.
    Negative delta => improves under supportive regime (amplifier).
    Positive delta => worsens under supportive regime (inhibitor).
    """
    out = df.copy()
    out["delta_rank"] = out[supportive_rank_col] - out[suppressive_rank_col]

    def label(d):
        if abs(d) <= stable_threshold:
            return "Stable Actor"
        if d <= -shift_threshold:
            return "Power Amplifier"
        if d >= shift_threshold:
            return "Power Inhibitor"
        # moderate change: still informative
        return "Moderate Shift"

    out["role_class"] = out["delta_rank"].apply(label)
    return out
