# Z-score Iteration — Summary

Audience: Developers

## Overview
This document tracks the evolution of the z-score signal methodology across iterations.

---

## Iteration 3 — Winsorized baseline before EWM (2026-07-12)

### What changed
- Before computing the EWM mean/std, the 30-day `baseline_counts` window is now winsorized: any day above `median + winsorize_k * MAD` is capped to that value. One-sided (upper tail only) — article counts floor at 0, so a symmetric cap would rarely touch the low end.
- `winsorize_k` (default `5.19`) is a new `[analysis]` config field, validated `> 0`, following the same pattern as `z_score_cap`. The default is `3.5 / 0.6745`, matching the Iglewicz & Hoaglin modified z-score outlier threshold (`|M| > 3.5`, where `M = 0.6745*(x-median)/MAD`) expressed in raw MAD units instead of `0.6745`-scaled units — an earlier draft used `k=3.5` directly, which was a more aggressive threshold than the textbook rule it was meant to match.
- If MAD is 0 (common with sparse, mostly-zero baselines — no dispersion around the median), winsorization is a no-op rather than collapsing every above-median value down to the median.
- `baseline_counts` in `compute_volume_stats()`'s return value stays **raw** (unwinsorized) — only the values fed into `_ewma_mean_std()` are capped. Consumers that read `baseline_counts` (tests, monitor gauges via `mean`/`std`, not the counts array itself) are unaffected in shape.

### Files touched
- `financial_news/analysis.py` — added `_winsorize_upper()`, wired into `compute_volume_stats()` ahead of `_ewma_mean_std()`, added `capped_days` to the info log line.
- `financial_news/config.py` — added `winsorize_k` to `AnalysisConfig` and `load_config`.
- `config.example.toml` — documented the new `[analysis] winsorize_k` key.
- `tests/test_config.py` — added tests for `winsorize_k` defaults, loading, and validation.
- `tests/test_calculate_z_score.py` — added direct unit tests for `_winsorize_upper()`.
- `tests/test_get_news_volume.py` — added a wiring test confirming `compute_volume_stats()` calls `_winsorize_upper()` on the baseline before computing mean/std.

### Rationale
- A single outlier day sitting in the 30-day baseline (e.g. an old earnings/M&A headline burst) inflates the EWM std, which widens what counts as "normal" and can mask a genuinely unusual day today.
- A full modified z-score (median/MAD replacing mean/std entirely) was considered but rejected: pandas has no built-in exponentially-weighted median/MAD, so it would either require abandoning the EWM recency-weighting added in Iteration 2 (losing gradual-regime adaptation) or writing a nonstandard, unverified weighted-median implementation. Winsorizing the *input* to the existing EWM pipeline gets the outlier-robustness benefit without touching the EWM machinery or its guarantees.

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
