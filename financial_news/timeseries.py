"""Shared time-series helpers for deterministic signal modules.

Used by both analysis.py (news-volume baseline) and price.py (return baseline).
"""

import pandas as pd


def winsorize(values: list[float], k: float, two_sided: bool) -> list[float]:
    """Cap outliers in *values* at median + k * MAD, and at median - k * MAD too
    when two_sided.

    If MAD is 0 (no dispersion around the median, common with sparse, mostly-zero
    baselines), values are returned unchanged rather than collapsing every point
    on the capped side down to the median.
    """
    if not values:
        return values
    series = pd.Series(values, dtype=float)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return values
    lower = median - k * mad if two_sided else None
    return series.clip(lower=lower, upper=median + k * mad).tolist()


def ewma_mean_std(values: list[float], span: int) -> tuple[float, float]:
    """Return the exponentially weighted mean and std over values (adjust=True)."""
    if not values:
        return 0.0, 0.0
    ewm = pd.Series(values, dtype=float).ewm(span=span, adjust=True)
    mean = float(ewm.mean().iloc[-1])
    std = ewm.std().iloc[-1]
    return mean, 0.0 if pd.isna(std) else float(std)
