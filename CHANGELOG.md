# Changelog

## 2026-05-04 — EWM baseline, config-driven thresholds, calendar-day window (by kovyha)

- Replaced the 7-day simple mean/std baseline with a 30-day exponentially weighted mean (EWM) baseline using pandas `ewm(span=30, adjust=True)`.
- Baseline uses a 30 calendar-day window. News activity is not gated by exchange hours; weekends and holidays are included so all news days contribute equally to the baseline.
- Moved all hardcoded analysis constants out of `analysis.py` and into `config.py` / `config.toml`:
  - `baseline_days` (default: 30) — number of calendar days in the EWM window.
  - `threshold_elevated` (default: 2.0) — z-score above which volume is elevated.
  - `threshold_unusual` (default: 3.0) — z-score above which volume is unusual.
- Added `AnalysisConfig` dataclass to `config.py` with validation: unknown keys raise, `baseline_days` must be > 0, `threshold_elevated` must be < `threshold_unusual`.
- Updated `fetch_news` signature from `(symbol, days: int)` to `(symbol, from_date: date, to_date: date)` for explicit date control.
- Server output labels now reflect the configured baseline window dynamically (e.g. `Mean (30-day EWM)`).
- Added 6 new `test_config.py` tests covering the `[analysis]` config section (defaults, overrides, unknown keys, ordering violation, zero baseline).
- Updated `test_get_news_volume.py` and `test_boundary.py` for the new `fetch_news` signature and output label strings.
- Added `pandas>=2.0` as a project dependency; EWM is delegated to `pd.Series.ewm()` rather than a custom NumPy implementation.

## 2026-03-24 — Packaging, validation, documentation, and governance refinement (by kovyha)

- Refactored the project into an installable package structure under `financial_news/`.
- Added editable install support through `pyproject.toml` build-system configuration.
- Added focused test coverage for z-score logic and news-volume classification.
- Added clearer runtime handling for missing `FINNHUB_API_KEY`.
- Wrapped upstream Finnhub fetch failures in a stable application-level error message.
- Added a neutral no-data message for symbols with no recent and no baseline coverage.
- Expanded tests for startup error handling, upstream fetch failures, and no-data behavior.
- Added repository validation tooling with `ruff check`, `ruff format`, and documented `pytest` as part of the default workflow.
- Added a minimal GitHub Actions CI workflow for lint and test on pushes and pull requests.
- Added `financial_news/config.py`: Configuration system for runtime settings (logging directory, file size, level, etc.).
- Added `config.example.toml`: Example configuration file with inline documentation.
- Added `financial_news/diagnostic.py`: Diagnostic agent module for error log analysis and investigation.
- Added `.github/workflows/diagnostic.yaml`: Scheduled diagnostic workflow (disabled by default; runs daily or manually).
- Added `.github/CODEOWNERS`: Code ownership rules for protected files (AGENTS.md, SKILL.md, CI, locked deps).
- Added `MAINTAINERS.md`: Documentation of project maintainers and review responsibilities.
- Expanded logging to use timestamp-based rolling file handler with gzip compression (in `server.py`).
- Added comprehensive test coverage:
  - `test_boundary.py`: Tests enforcing deterministic/LLM boundary (no model inference in signal layer).
  - `test_config.py`: Tests for configuration loading, validation, defaults, and edge cases.
  - `test_logging.py`: Tests for rolling file handler, compression, and rollover behavior.
  - `test_diagnostic.py`: Tests for error log parsing and diagnostic reporting.
- Updated `docs/developer-infra.md` to document all modules and test files.
- Updated `docs/AGENTS.md` to include Diagnostic agent role with scope and constraints.
- Added and refined supporting documentation in `docs/engineering-standards.md`, `docs/SKILL.md`, and `README.md`.
- Maintained 95% code coverage threshold across all new code.

## 2026-03-20 — Z-score iteration (by kovyha)

- Introduced z-score normalization using `numpy` for numeric operations.
- Added `numpy` to project dependencies and installed via `uv`.
- Updated code to compute mean/std and apply z-score normalization (`server.py`).
- Updated `pyproject.toml` and `uv.lock` to record the dependency and resolved versions.
- Documentation: see [docs/z_score_iteration.md](docs/z_score_iteration.md)
