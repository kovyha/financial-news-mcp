# financial-news-mcp

An MCP server that detects statistically unusual news activity on any stock ticker, with a multi-agent pipeline that scores sentiment, reasons over signals, and delivers a daily market briefing.

## Why This Exists

Financial news moves markets, but not all news activity is equal. This project detects when news volume on a stock is statistically unusual — giving traders and analysts an early signal that something significant may be happening before it is widely reflected in price.

Built on Model Context Protocol (MCP), it connects Claude directly to live financial news data and turns a natural-language question into a quantitative signal with contextual reasoning.

## What It Looks Like

Ask Claude: "Any unusual news activity on HSBC today?"

Claude responds with something like: "Yes, HSBC is showing significant unusual news volume today. Z-score: 3.1. Article count is 3.4x the recent average. Likely driver: [reasoning based on the headlines and sentiment signals]."

Every morning before the US market opens, the daily briefing agent scans the watchlist, scores every headline and article summary with finBERT sentiment analysis, and uses Claude to write a concise briefing covering notable tickers, likely drivers, and cross-watchlist themes.

## Architecture

### Deterministic and LLM layers

The architecture maintains an explicit boundary between two layers:

**Deterministic signal layer** (`get_news_volume` MCP tool): fetches data from Finnhub, computes article counts, a 30-day exponentially weighted mean and standard deviation using pandas, and classifies the result against configurable thresholds (default: z < 2 = normal, z < 3 = elevated, z ≥ 3 = unusual). No model inference occurs here. Output is a structured, reproducible dict.

**LLM reasoning layer** (Claude, via MCP): receives that structured output as tool context and interprets what a statistically significant spike may mean — drawing on the headlines, sentiment scores, and its broader knowledge.

This boundary is intentional. In any regulated or auditable environment you can demonstrate precisely what data was fetched, what was computed deterministically, and exactly where model judgement begins.

### Agents

| Agent | File | Trigger | What it does |
|---|---|---|---|
| MCP server | `server.py` | On-demand (MCP client) | Exposes `get_news_volume` and `health_check` tools |
| Daily briefing | `briefing.py` | Daily pre-market (CI) | Scores all watchlist headlines and summaries with finBERT, filters to high-confidence signals (≥ 0.85), calls Claude to produce a plain-language briefing; emails result |
| Daily monitor | `monitor.py` | Daily pre-market (CI) | Fetches watchlist z-scores, exports OTel gauges to Grafana Cloud |
| Diagnostic | `diagnostic.py` | On-demand | Reads error logs, calls Claude to identify root cause and propose a fix |

### Sentiment pipeline

All watchlist headlines and article summaries are scored locally by [finBERT](https://huggingface.co/ProsusAI/finbert) (no API cost). When a Finnhub summary is present it is appended to the headline before scoring so finBERT has more signal; the original headline is preserved in the output. Articles are then filtered by confidence threshold before being passed to Claude:

- Headlines with finBERT confidence ≥ `confidence_threshold` (default: 0.85) are included
- If fewer than `prompt_headlines_min` (default: 5) clear the bar, the top-scoring headlines fill the gap
- No more than `prompt_headlines_max` (default: 50) headlines are passed per ticker
- For `elevated` and `unusual` tickers, high-confidence neutral headlines (score ≥ `confidence_threshold`) are discarded before selection — they are either low-signal or feed noise; weakly-neutral headlines (score < threshold) are kept as borderline signals
- All thresholds are configurable in `config.toml`

### Stage × mode matrix

| | **Interactive (MCP)** | **GitHub Actions (briefing)** |
|---|---|---|
| **Input** | Ticker symbol | Watchlist tickers (snapshot or live fetch) |
| **Z-score** | Fetches today + 30-day baseline → EWM z-score → `normal` / `elevated` / `unusual` | Same: `compute_volume_stats` per ticker, same 30-day EWM baseline |
| **finBERT — today** | Scores today's articles → confidence-filtered → `articles` (label + score) | Same: `selected_articles` shown in prompt stats block with labels |
| **finBERT — 7-day window** | Scores `headline_articles` → neutral-filtered → `headline_context` (label + score + date + source) | Same: `selected_headline_articles` → formatted with labels → pre-loaded into `headlines_cache` |
| **Reasoning input** | `articles` (today, scored) + `headline_context` (7-day, scored, always provided) | Stats block with `selected_articles` + `headlines_cache` served on Claude's tool call |
| **Reasoning trigger** | Claude decides immediately on receiving the tool result | Claude calls `get_news_headlines` for elevated/unusual tickers |
| **Reasoning output** | Claude's response in the conversation | Plain-text briefing, optionally emailed |

### CI pipeline

The GitHub Actions `monitor.yaml` workflow runs daily at 12:00 UTC (8am ET, pre-market):

1. **Monitor step** — fetches z-scores, exports to Grafana Cloud, writes a snapshot
2. **Briefing step** — reads the snapshot, runs finBERT sentiment, calls Claude, sends email

## Setup

### Prerequisites

- Python 3.12+
- `uv`
- `FINNHUB_API_KEY` environment variable

### Install

```bash
uv sync
```

For finBERT sentiment scoring (required by the briefing agent):

```bash
uv sync --group sentiment
```

### Run the MCP server

```bash
FINNHUB_API_KEY=<key> uv run python -m financial_news.server
```

### Run the daily briefing

```bash
FINNHUB_API_KEY=<key> \
ANTHROPIC_API_KEY=<key> \
uv run python -m financial_news.briefing
```

### Run the monitor

```bash
FINNHUB_API_KEY=<key> \
GRAFANA_CLOUD_OTLP_ENDPOINT=<url> \
GRAFANA_CLOUD_BASIC_AUTH_HEADER="Basic <token>" \
uv run python -m financial_news.monitor
```

### Configuration

Copy `config.example.toml` to `config.toml` to customise thresholds, watchlist tickers, sentiment model, and email settings. `config.toml` is gitignored — do not commit it.

### Example MCP queries

Once connected to Claude Desktop or another MCP client:

- "Any unusual news activity on HSBC today?"
- "Check whether Tesla has elevated news flow right now."
- "Why is there unusual news volume on Apple today?"

## Validation

```bash
uv run ruff check .
uv run ruff format .
uv run pytest --cov=financial_news --cov-report=term-missing -q
```

Coverage threshold is enforced at 95%. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full pre-push checklist.

## Future enhancements

- Link the news signal to a kdb-based trading and order-flow view so traders can see news activity alongside positioning, execution, and immediate market reaction.
- Add real-time signal and algo attribution to measure whether news-volume and sentiment signals are actually improving trading outcomes.

## Docs

- Agent catalog and behavioral rules: [docs/AGENTS.md](docs/AGENTS.md), [docs/SKILL.md](docs/SKILL.md)
- Engineering standards: [docs/engineering-standards.md](docs/engineering-standards.md)
- Developer infra (packaging, install, test strategy): [docs/developer-infra.md](docs/developer-infra.md)
- Z-score iteration notes: [docs/z_score_iteration.md](docs/z_score_iteration.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
