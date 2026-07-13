"""Shared time-series helpers for deterministic signal modules.

Used by both analysis.py (news-volume baseline) and price.py (return baseline).
"""

import pandas as pd


def ewma_mean_std(values: list[float], span: int) -> tuple[float, float]:
    """Return the exponentially weighted mean and std over values (adjust=True)."""
    if not values:
        return 0.0, 0.0
    ewm = pd.Series(values, dtype=float).ewm(span=span, adjust=True)
    mean = float(ewm.mean().iloc[-1])
    std = ewm.std().iloc[-1]
    return mean, 0.0 if pd.isna(std) else float(std)
