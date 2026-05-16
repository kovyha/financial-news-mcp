# financial-news-mcp

An MCP server that detects statistically unusual news volume on any stock ticker. The deterministic signal layer (z-score computation via pandas EWM) feeds Claude as structured context; Claude then reasons over the signal and headlines.

## What this project is

- Exposes two MCP tools: `get_news_volume` and `health_check`
- `get_news_volume` fetches Finnhub news, computes a 30-day EWM baseline, calculates a z-score, and classifies the result as `normal` / `elevated` / `unusual` against configurable thresholds
- All LLM reasoning begins *after* the tool returns structured output — no model inference occurs inside the tool

## Architecture: the boundary that matters

The deterministic/LLM boundary is the most important architectural property of this system. It is enforced by tests (`tests/test_boundary.py`), documented in `docs/SKILL.md`, and reviewed on every PR touching `financial_news/server.py`.

Never move business logic (z-score math, classification thresholds, Finnhub fetch behaviour) into prompts, and never move interpretation into the deterministic layer.

## Key source files

| File | What it does |
|---|---|
| [financial_news/server.py](financial_news/server.py) | MCP server, tool definitions, orchestration |
| [financial_news/analysis.py](financial_news/analysis.py) | Z-score computation and EWM baseline logic |
| [financial_news/config.py](financial_news/config.py) | Config loader — thresholds, baseline window, logging settings |
| [financial_news/diagnostic.py](financial_news/diagnostic.py) | Diagnostic agent for error log analysis |
| [financial_news/monitor.py](financial_news/monitor.py) | Daily monitoring agent — fetches watchlist z-scores and exports OTel gauges to Grafana Cloud |
| [config.example.toml](config.example.toml) | Copy to `config.toml` to customize thresholds and logging |

## Governance docs — read these before making changes

| Doc | Purpose |
|---|---|
| [docs/AGENTS.md](docs/AGENTS.md) | Agent catalog — roles, triggers, scope, approval gates |
| [docs/SKILL.md](docs/SKILL.md) | Behavioral rules for agents — boundary constraints, validation workflow, commit guidance |
| [docs/engineering-standards.md](docs/engineering-standards.md) | Code quality expectations (ALWAYS / PREFER / AVOID / REQUIRE) |
| [docs/developer-infra.md](docs/developer-infra.md) | Packaging, install, test strategy, common commands |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Canonical validation workflow — run before every PR |

## Validation — always run before proposing changes

```bash
uv run ruff check .
uv run ruff format .
uv run pytest --cov=financial_news --cov-report=term-missing -q
```

Coverage threshold is enforced by `pyproject.toml`. Do not drop it.

## Pre-push checklist — complete before every `git push`

- [ ] Update any affected docs (`CLAUDE.md`, `docs/*.md`, `CONTRIBUTING.md`) to reflect the change

## Protected files — require explicit human instruction to modify

- `.github/workflows/ci.yaml`
- `uv.lock`
- `docs/AGENTS.md`
- `docs/SKILL.md`

## Human review gate

All changes — including agent-authored ones — require explicit human approval before landing on `main`. No agent may self-approve or merge its own changes.

## Running the server

```bash
FINNHUB_API_KEY=<key> uv run python -m financial_news.server
```

## Running the monitor

```bash
FINNHUB_API_KEY=<key> \
GRAFANA_CLOUD_OTLP_ENDPOINT=<url> \
GRAFANA_CLOUD_BASIC_AUTH_HEADER="Basic <token>" \
uv run python -m financial_news.monitor
```

The monitor runs automatically via `.github/workflows/monitor.yaml` daily at 21:00 UTC (after US market close). The workflow is disabled by default until secrets are configured.

Requires Python 3.12+ and `uv`.
