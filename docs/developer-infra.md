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
  - `server.py` — MCP server exposing tools (`get_news_volume`, `health_check`).
  - `config.py` — configuration loader for logging and analysis settings (baseline window, classification thresholds).
  - `briefing.py` — daily briefing agent; runs the enrichment pipeline over watchlist stats, then calls Claude to produce a plain-language briefing.
  - `enrichment.py` — centralised enrichment pipeline: finBERT scoring, confidence-threshold article selection, and neutral-article filtering for elevated/unusual tickers; used by both the MCP server and the briefing agent.
  - `sentiment.py` — finBERT sentiment scoring; deterministic preprocessing for the briefing agent. Requires the `sentiment` dep group (`uv sync --group sentiment`).
  - `diagnostic.py` — LLM-powered diagnostic agent; reads error logs then calls Claude to identify root cause and propose a fix.
  - `monitor.py` — daily monitoring agent; pushes z-score/count/EWM mean as OTel gauges to Grafana Cloud.
  - `__init__.py` — package initialization.
- `tests/` — pytest test files and fixtures:
  - `test_calculate_z_score.py` — unit tests for z-score calculation.
  - `test_get_news_volume.py` — integration tests for news-volume detection.
  - `test_boundary.py` — tests enforcing the deterministic/LLM boundary.
  - `test_config.py` — tests for the configuration system.
  - `test_logging.py` — tests for logging and rollover behavior.
  - `test_diagnostic.py` — tests for the diagnostic agent.
  - `test_monitor.py` — tests for the monitoring agent (run(), gauge values, OTel wiring).
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
- `pyproject.toml` — project metadata, dependencies, and `build-system` config for setuptools. The `diagnostic` dependency group (`anthropic`) is separate from the main runtime deps to keep the MCP server's install surface free of LLM packages.
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
- **Briefing agent tests:** `test_briefing.py` tests stat collection, prompt formatting, and the LLM agentic loop (mocked — no real API calls in CI).
- **Enrichment pipeline tests:** `test_enrichment.py` tests `enrich_stats`, `enrich_ticker`, and `select_articles` — covering sentiment scoring, confidence-threshold filtering, neutral-article discarding for elevated/unusual tickers, and fallback behaviour for zero-news tickers.
- **Sentiment tests:** `test_sentiment.py` tests finBERT scoring (pipeline mocking, label normalisation, score rounding).
- **E2E deterministic tests:** `test_e2e_deterministic.py` runs against the live Finnhub API; skipped in CI unless a real `FINNHUB_API_KEY` is provided (`uv run pytest -m e2e`).
- **Monitor agent tests:** `test_monitor.py` covers run() success/failure/partial, gauge value correctness, OTel provider wiring, and GAUGE_SPECS contract.
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
