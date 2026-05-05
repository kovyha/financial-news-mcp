# Z-score Iteration — Summary

Audience: Developers

## Overview
This document tracks the evolution of the z-score signal methodology across iterations.

---

## Iteration 2 — EWM baseline, calendar-day window, config-driven thresholds (2026-05-04)

### What changed
- Replaced the 7-day simple mean/std baseline with a 30-day **exponentially weighted mean (EWM)** baseline.
- Baseline uses a **30 calendar-day window**. News is not gated by exchange hours; weekends and holidays are valid signal days and are included.
- EWM is computed using `pd.Series.ewm(span=30, adjust=True)` from pandas, which handles the reliability-weight bias correction internally.
- Classification thresholds (`threshold_elevated`, `threshold_unusual`) and the baseline window (`baseline_days`) are now sourced from `config.toml` via `AnalysisConfig` rather than hardcoded in `analysis.py`.
- `fetch_news` signature changed from `(symbol, days: int)` to `(symbol, from_date: date, to_date: date)` for explicit date control.

### Files touched
- `financial_news/analysis.py` — added `_ewma_mean_std()`, updated `compute_volume_stats()` and `fetch_news()`.
- `financial_news/config.py` — added `AnalysisConfig` dataclass and `[analysis]` section parsing in `load_config`.
- `financial_news/server.py` — output labels now use the configured `BASELINE_DAYS` value dynamically.
- `tests/test_config.py` — added 6 tests for the `[analysis]` config section.
- `tests/test_get_news_volume.py`, `tests/test_boundary.py` — updated for new `fetch_news` signature and label strings.

### Rationale
- A 7-day simple mean over at most 6 data points produces an unstable variance estimate (high noise at small n, fat-tailed t-distribution).
- EWM over 30 days gives a more stable baseline and naturally downweights older observations, so the signal adapts to gradual regime changes without false alarms.
- Calendar days are correct for news volume — unlike price, news breaks on weekends and holidays.

---

## Iteration 1 — Z-score normalization (2026-03-20)

### What changed
- Added NumPy as a project dependency.
- Updated code to compute means and standard deviations using `numpy` (`ddof=1` for sample std).
- Implemented `calculate_z_score()` and integrated z-score classification into the MCP flow.
- Baseline: 7-day rolling window, simple mean and std.

### Files touched
- `financial_news/server.py` — introduced `calculate_z_score()` and z-score classification.
- `pyproject.toml` — dependency entry for `numpy`.
- `tests/test_calculate_z_score.py` — unit coverage for z-score guard logic.
- `tests/test_get_news_volume.py` — classification-oriented tests for tool output.

### Rationale
- `numpy` was chosen because numeric operations benefit from NumPy's API and clarity.
- `ddof=1` was used explicitly to distinguish sample vs population std.
