# Changelog

## 2026-11-04 — Second-model label check as a batch script

- New `python -m financial_news.evaluation cross-check`. A second model labels rows from the same blind inputs as the human labeller (return plus every headline and summary), one request per row through the Message Batches API, and the command lists rows where it disagrees with `eval/labels.json`. It prints an exact-input-token cost estimate and spends nothing without `--submit`, and refuses if the worst case exceeds `--max-usd` (default $10). Results go to `eval/cross_check.json`.
- The classification rule moves from `docs/evaluation.md` into `eval/classification_rule.md`, its only copy. The human labeller follows it and the `cross-check` prompt reads it, so the two can't drift apart. It now includes the after-close clarification (news after 4pm explains the next session) applied in the 2026-11-02 revision.
- New rule in `CLAUDE.md`: agents must not do bulk LLM work row by row inside their own session. The 2026-11-02 label check (43 rows, Fable 5.1) was done that way and cost £10.13 (about $13.50). A typical batch run over all 53 labelled rows is estimated at $3.74 on Fable 5.1 or $1.49 on Opus 5.5.

## 2026-11-02 — Evaluation judge moves to Opus 5.5; ground-truth labels revised

- The evaluation judge now uses the same Claude settings as the live agents (`[llm]`: `claude-opus-5-5`, effort `medium`, server-side fallback), so a change to the live model re-runs the evaluation gate. `max_tokens` rises from 300 to 16000 because Opus 5.5 always thinks and thinking counts toward the cap. The gate fingerprint records the full Claude request settings instead of the model name alone.
- kovyha revised 15 of the 43 scored labels in `eval/labels.json`, re-applying the classification rule strictly to rows flagged by a second-model check: 13 to `ambiguous`, 2 to `noise`. Scored rows fall from 43 to 30. Each revised row keeps its original note plus the reason. See "Label revision" in `docs/evaluation.md`, including why the review wasn't blind.
- On the revised labels the full system (Opus 5.5) scores F1 0.923 / accuracy 0.900, net 3 rows better than Opus 4.7. On the original labels it had failed the gate (net 5 worse). The component ablation and exploratory analyses are still on the original labels and are marked as such.

## 2026-11-02 — Evaluation gate in CI

- New `financial_news/eval_gate.py` (protected) and `eval/fingerprint.json`. The fingerprint records what the evaluation judge's verdicts depend on: its prompts, the code on the verdict path, the Claude model and token limit, the locked versions of `anthropic`, `transformers` and `torch`, the pinned finBERT revision, the article-selection settings, and the labels and candidates.
- New CI jobs. `eval-fingerprint` runs on every PR with no API calls. `eval-run` runs only when the fingerprint or stored verdicts change, waits for maintainer approval in the `evaluation` environment, and fails if verdicts got net worse by more than max(2, 2% of labeled rows), or if the committed verdicts don't reproduce.
- `eval/candidates.json` for CI comes from the private repo `kovyha/financial-news-eval-data`, checked against the committed checksum.

## 2026-11-02 — Enforce the deterministic/LLM boundary at runtime

- The MCP server and monitor now run inside a network sandbox (`financial_news/sandbox.py`, Anthropic's sandbox runtime) that allows only Finnhub, Yahoo Finance and the Grafana endpoint. Both refuse to start outside it. Launch with `python -m financial_news.sandbox {server,monitor}`.
- Sentiment models must be listed in `financial_news/approved_models.py`, pinned to a Hugging Face commit; the server loads finBERT offline. A failed model load now logs at ERROR.
- `tests/test_boundary.py` now follows imports from the entry points instead of a hand-kept module list, and also checks for local model runtimes, MCP sampling, dynamic imports, committed weights, and unclassified entry points.
- New `tests/test_sandbox_integration.py` and CI `sandbox` job try real escape routes against the sandbox on Linux.
- The protected-files check now runs on PRs, reads its list from `main` (`.github/protected-files.txt`), and is approved with the `protected-change-approved` label. CODEOWNERS points at `@kovyha`. `.claude/settings.json` denies agent pushes to `main`.
- Duplication removed: one `winsorize` in `timeseries.py` (one- or two-sided) replaces the copies in `analysis.py` and `price.py`; one `setup-sandbox` action for both workflows; protected-file and review-gate details now live only in `docs/SKILL.md`.
- Docs corrected: finBERT runs inside `get_news_volume`, so the claim is "no LLM inference", not "no model inference".

## 2026-11-02 — Move the briefing and diagnostic agents to Claude Opus 5.5

- The briefing and diagnostic agents moved from Claude Opus 4.7 to Opus 5.5, with effort set explicitly to medium and Anthropic's server-side fallback enabled. All three settings live in one `LLMConfig`, overridable from a new `[llm]` section in `config.toml`. `max_tokens` rises to 16000 for both (from 4096 in the briefing and 8192 in the diagnostic), which raises the cost ceiling per call. The evaluation harness stays on Opus 4.7 until it's re-run, so its published results remain accurate (see `docs/evaluation.md`).
- One Claude tool-use loop in `llm.py` replaces the copies in `briefing.py` and `diagnostic.py`, with shared test mocks in `tests/claude_mocks.py`.

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
