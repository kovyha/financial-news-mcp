# Developer Infra Notes

Audience: Developers

## Purpose
Provide a concise reference for the project's packaging, install, and test workflows so future work is reproducible and easy to follow.

For the canonical contributor workflow and validation commands, see `CONTRIBUTING.md`.

## High-level overview
- Project packaged as `financial_news` (a Python package directory). This lets you `import financial_news.server` and makes the code installable.
- Build/install tooling: we use `uv` as the project manager. Run `uv sync` to install the project and its dependencies. For finBERT sentiment scoring use `uv sync --group sentiment`.
- Tests use `pytest`. Test helper/setup is in `tests/conftest.py`.

## Key files
- `financial_news/` — package directory:
  - `server.py` — MCP server exposing tools (`get_news_volume`, `get_price_move`, `get_price_intraday`, `health_check`).
  - `config.py` — configuration loader for logging, analysis, and price settings (baseline window, classification thresholds).
  - `briefing.py` — daily briefing agent; runs the enrichment pipeline over watchlist stats (news + price), then calls Claude (with `get_news_headlines` and, for elevated/unusual tickers, `get_price_intraday`) to produce a plain-language briefing.
  - `enrichment.py` — centralised enrichment pipeline: finBERT scoring, confidence-threshold article selection, and neutral-article filtering for elevated/unusual tickers; used by both the MCP server and the briefing agent.
  - `sentiment.py` — finBERT sentiment scoring; deterministic preprocessing for the briefing agent. Requires the `sentiment` dep group (`uv sync --group sentiment`).
  - `diagnostic.py` — LLM-powered diagnostic agent; reads error logs then calls Claude to identify root cause and propose a fix.
  - `monitor.py` — daily monitoring agent; pushes news z-score/count/EWM mean and price z-score/return/EWM mean as OTel gauges to Grafana Cloud.
  - `price.py` — deterministic price-move signal: a volatility-normalized return z-score per ticker (via yfinance), mirroring `analysis.py`'s news-volume z-score so the two can be directly compared. Also provides `fetch_intraday_prices`, a same-day-only attribution drill-down (not part of the anomaly-detection signal), fetched on demand only for tickers already flagged elevated/unusual.
  - `timeseries.py` — shared EWM mean/std helper used by both `analysis.py` and `price.py`.
  - `__init__.py` — package initialization.
- `tests/` — pytest test files and fixtures:
  - `test_calculate_z_score.py` — unit tests for z-score calculation.
  - `test_get_news_volume.py` — integration tests for news-volume detection.
  - `test_price.py` — unit tests for the price-move z-score, two-sided winsorization, and `compute_price_stats` (mocked yfinance).
  - `test_get_price_move.py` — integration tests for the `get_price_move` MCP tool.
  - `test_get_price_intraday.py` — integration tests for the `get_price_intraday` MCP tool.
  - `test_e2e_price.py` — end-to-end tests against the live yfinance API; skipped unless `RUN_PRICE_E2E=1` (`uv run pytest -m price`).
  - `test_boundary.py` — tests enforcing the deterministic/LLM boundary.
  - `test_config.py` — tests for the configuration system.
  - `test_logging.py` — tests for logging and rollover behavior.
  - `test_diagnostic.py` — tests for the diagnostic agent.
  - `test_monitor.py` — tests for the monitoring agent (run(), gauge values for both news and price, OTel wiring, independent failure handling).
  - `conftest.py` — pytest configuration and fixtures.
- `docs/` — documentation for iterations and developer notes.
- `.github/workflows/` — GitHub Actions CI/CD workflows:
  - `ci.yaml` — baseline CI (`ruff check` + `pytest` on push and pull request, protected-file checks); caches `ProsusAI/finbert` model weights and pre-downloads them before the test step so sentiment tests run without a live HuggingFace download.
  - `monitor.yaml` — daily run at 12:00 UTC (8am ET, pre-market); runs the monitor step (OTel gauges to Grafana Cloud), then the briefing step (finBERT sentiment + Claude analysis + email), and invokes the LLM diagnostic agent on failure. Disabled by default until secrets are configured.
- `.github/CODEOWNERS` — code ownership and review requirements for governance files.
- `config.example.toml` — example configuration file (copy to `config.toml` to customize).
- `README.md` — public project overview and usage entry point.
- `CONTRIBUTING.md` — canonical contributor workflow and validation steps.
- `MAINTAINERS.md` — project maintainers and their responsibilities.
- `CHANGELOG.md` — change history across iterations.
- `pyproject.toml` — project metadata, dependencies, and `build-system` config for setuptools. The `diagnostic` dependency group (`anthropic`) is separate from the main runtime deps to keep the MCP server's install surface free of LLM packages. `yfinance` (price-move history/quotes) is a main runtime dependency, not an optional group — it's required for `get_price_move` to function at all, unlike the optional sentiment/diagnostic enhancements.
- `docs/engineering-standards.md` — code quality and review expectations for humans and agents.
- `uv.lock` — locked dependency versions (managed by `uv`).

## Environment variable overrides (no config.toml required)

These env vars are read by `load_config` when no `config.toml` is present — useful in CI where the file is gitignored.

| Env var | Overrides | Example |
|---|---|---|
| `LOG_LEVEL` | `[logging] level` | `DEBUG` or `INFO` (takes precedence over config.toml when both are present) |
| `SENTIMENT_MODEL_NAME` | `[sentiment] model_name` | `ProsusAI/finbert` |
| `EMAIL_RECIPIENTS` | `[email] recipients` | `you@example.com,other@example.com` |
| `SMTP_HOST` | `[email] smtp_host` | `smtp.gmail.com` |
| `SMTP_PORT` | `[email] smtp_port` | `587` |
| `SMTP_FROM` | `[email] smtp_from` | `sender@example.com` |

## Common commands (uv-focused)
- Install all groups (required for tests): `uv sync --all-groups`
- Add a dependency: `uv add "package_name"` or dev: `uv add --dev "pkg>=x.y"` or diagnostic: edit `pyproject.toml` and run `uv sync --all-groups`
- Run a quick Python check inside project env: `uv run python -c 'import numpy as np; print(np.__version__)'`
- Run baseline CI checks locally: `uv run ruff check .` and `uv run pytest`

Use `CONTRIBUTING.md` for the standard validation flow.

If not using `uv` you can use the venv's pip directly:
```bash
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt  # if you maintain one
```

## Tests strategy
- **Unit tests for numeric logic:** `test_calculate_z_score.py`.
- **Integration tests for news-volume detection:** `test_get_news_volume.py` (patches `calculate_z_score` and `fetch_news` as appropriate).
- **Boundary enforcement tests:** `test_boundary.py` verifies the deterministic/LLM boundary is maintained (no model inference in the signal layer).
- **Configuration system tests:** `test_config.py` covers config loading, defaults, validation, and error handling.
- **Logging system tests:** `test_logging.py` verifies the timestamp-based rolling file handler and compression behavior.
- **Diagnostic agent tests:** `test_diagnostic.py` tests error log parsing, diagnostic reporting, and the LLM agentic loop (using mocked Anthropic client responses — no real API calls in CI).
- **Briefing agent tests:** `test_briefing.py` tests stat collection, prompt formatting (including today's per-article "time" and the elevated/unusual-only intraday cache), and the two-tool LLM agentic loop (`get_news_headlines` + `get_price_intraday`, mocked — no real API calls in CI).
- **Enrichment pipeline tests:** `test_enrichment.py` tests `enrich_stats`, `enrich_ticker`, and `select_articles` — covering sentiment scoring, confidence-threshold filtering, neutral-article discarding for elevated/unusual tickers, and fallback behaviour for zero-news tickers.
- **Sentiment tests:** `test_sentiment.py` tests finBERT scoring (pipeline mocking, label normalisation, score rounding).
- **E2E deterministic tests:** `test_e2e_deterministic.py` runs against the live Finnhub API; skipped in CI unless a real `FINNHUB_API_KEY` is provided (`uv run pytest -m e2e`).
- **Price-move signal tests:** `test_price.py` (z-score/winsorize unit tests, `compute_price_stats` with mocked yfinance) and `test_e2e_price.py` (live yfinance, requires `RUN_PRICE_E2E=1`, `uv run pytest -m price`).
- **Monitor agent tests:** `test_monitor.py` covers run() success/failure/partial for both news and price signals independently, gauge value correctness, OTel provider wiring, and GAUGE_SPECS/PRICE_GAUGE_SPECS contracts.
- **Test fixtures:** `conftest.py` sets `FINNHUB_API_KEY` to a harmless default for test imports and ensures test runners can find the package.

## Packaging rationale
- Converting to a package makes imports stable and enables editable installs for local development and CI.
- `pyproject.toml` declares the build system so modern tools can build/install the package consistently.

## Gotchas & lessons
- Do not commit secrets (API keys) to the repo. Use env vars or secret managers.
- `conftest.py` is a helpful bridge during the transition; once all collaborators use editable installs/CI it can be simplified or removed.
- Prefer extracting pure logic into small helper functions (e.g., `calculate_z_score`) to make unit testing straightforward.


## Next steps
- Remove `conftest.py` after CI/IDEs use editable install consistently, if desired.
