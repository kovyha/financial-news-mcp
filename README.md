# financial-news-mcp

An MCP server that detects statistically unusual news activity and price moves on any stock ticker, with a multi-agent pipeline that scores sentiment, cross-references price against news, and delivers a daily market briefing.

## Why This Exists

Financial news moves markets, but not all news activity is equal. This project detects when news volume on a stock is statistically unusual, and separately whether today's price move is statistically unusual for that stock — giving traders and analysts a way to judge whether a price move looks news-driven, or whether an unusual move is happening without a matching news signal.

Built on Model Context Protocol (MCP), it connects Claude directly to live financial news data and turns a natural-language question into a quantitative signal with contextual reasoning.

## What It Looks Like

Ask Claude: "Any unusual news activity on HSBC today?"

Claude responds with something like: "Yes, HSBC is showing significant unusual news volume today. Z-score: 3.1. Article count is 3.4x the recent average. Likely driver: [reasoning based on the headlines, article summaries, and sentiment signals]."

Every morning before the US market opens, the daily briefing agent scans the watchlist, scores every headline and article summary with finBERT sentiment analysis, and uses Claude to write a concise briefing covering notable tickers, likely drivers, and cross-watchlist themes.

## Architecture

### Deterministic and LLM layers

The architecture maintains an explicit boundary between two layers:

**Deterministic signal layer** (`get_news_volume` MCP tool): fetches data from Finnhub, computes article counts, a 30-day exponentially weighted mean and standard deviation using pandas, and classifies the result against configurable thresholds (default: z < 2 = normal, z < 3 = elevated, z ≥ 3 = unusual). No LLM inference occurs here. finBERT runs locally as a preprocessing step to score and select articles; it doesn't affect the z-score or classification, and articles are labelled `unavailable` if the `sentiment` group isn't installed. Output is a structured dict.

A second tool, **`get_price_move`**, mirrors this signal for price: it fetches daily price history via yfinance (no API key required), computes a volatility-normalized return z-score against an EWM baseline of prior daily returns, and classifies it the same way — expressed in the same z-score units so it can be directly compared against the news-volume signal.

A third tool, **`get_price_intraday`**, is a same-day attribution drill-down rather than a second detection signal: for a ticker already flagged elevated/unusual, it fetches today's intraday price bars (configurable interval, default 5-minute) so a specific timestamped headline (today's `get_news_volume` articles each carry a "time" field) can be lined up against a specific intraday move — e.g. a pre-market gap on one headline vs. an unrelated afternoon move on another.

**LLM reasoning layer** (Claude, via MCP): receives that structured output as tool context and interprets what a statistically significant spike may mean — drawing on the headlines, article summaries, sentiment scores, and its broader knowledge.

This boundary is intentional. In any regulated or auditable environment you can demonstrate precisely what data was fetched, what was computed deterministically, and exactly where model judgement begins.

### How the boundary is enforced

A written rule that the deterministic layer doesn't call an LLM is only advice. Here it's enforced in two layers.

**At runtime, by a network sandbox.** A hosted LLM call has to send the prompt over the network, whatever library or code makes it. So the MCP server and the monitor run inside [Anthropic's sandbox runtime](https://www.npmjs.com/package/@anthropic-ai/sandbox-runtime), which blocks every destination at the operating-system level except Finnhub, Yahoo Finance and the Grafana endpoint ([financial_news/sandbox.py](financial_news/sandbox.py)). Both processes refuse to start outside it, LLM API keys are removed from their environment, and finBERT loads offline. [tests/test_sandbox_integration.py](tests/test_sandbox_integration.py) tries real escape routes on every PR and checks each one is blocked:

- OpenAI, Anthropic, other providers and Hugging Face, through Python's own networking
- the same calls through `curl_cffi`, the compiled HTTP library yfinance uses, including with the proxy turned off
- a `curl` subprocess, including one told to ignore the proxy
- a raw socket to a hardcoded IP address
- a model server listening on `localhost`

A control run without the sandbox confirms those routes are really open, so the test isn't passing vacuously.

**Statically, by tests that run before the code does.** [tests/test_boundary.py](tests/test_boundary.py) follows every import reachable from the server and monitor, including imports inside functions, and fails on:

- LLM client libraries and local model runtimes, including any in the lockfile
- MCP sampling (asking the client to run an LLM, which needs no network call)
- generative pipelines, dynamic imports, and LLM provider URLs
- model weights committed to the repo
- sentiment models that aren't approved and pinned to a commit
- a runnable module that isn't declared deterministic, LLM or tooling, or a deterministic one that doesn't require the sandbox

**What this doesn't cover:**

- **AI features on an allowed host:** the sandbox checks where traffic goes, not what it does, so a future AI endpoint on Finnhub or Yahoo would pass.
- **A model loaded in-process by an unfamiliar library:** it makes no network call. Weights can only arrive through a dependency change (`uv.lock` is protected) or a committed file (rejected by test), but a new runtime that isn't on the list would get past the static check.
- **Determined code:** the sandbox is enforced by the operating system, but the static tests are pattern checks, not proof.
- **Changes to the guards themselves:** the sandbox, its allowlist, the approved models and the boundary tests are protected files. A PR that changes them fails CI until the maintainer approves it.

### Agents

| Agent | File | Trigger | What it does |
|---|---|---|---|
| MCP server | `server.py` | On-demand (MCP client) | Exposes `get_news_volume`, `get_price_move`, `get_price_intraday`, and `health_check` tools |
| Daily briefing | `briefing.py` | Daily pre-market (CI) | Scores all watchlist headlines and summaries with finBERT, filters to high-confidence signals (≥ 0.85), calls Claude to produce a plain-language briefing that cross-references news and price-move signals; emails result |
| Daily monitor | `monitor.py` | Daily pre-market (CI) | Fetches watchlist news and price-move z-scores, exports OTel gauges to Grafana Cloud (each signal fails independently) |
| Diagnostic | `diagnostic.py` | On-demand, or automatically on CI failure (monitor.yaml) | Reads error logs, calls Claude to identify root cause and propose a fix |

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
3. **Diagnostic step** (on failure of either step above) — reads error logs, calls Claude to identify root cause and propose a fix

## Setup

### Prerequisites

- Python 3.12+
- `uv`
- `FINNHUB_API_KEY` environment variable (news data; `get_price_move`'s yfinance data source needs no API key)
- Node.js, for the network sandbox (`npx` fetches the pinned sandbox runtime on first use). On Linux, also `bubblewrap`, `socat` and `ripgrep`.

### Install

```bash
uv sync
```

For finBERT sentiment scoring (used by the MCP server and the briefing agent), install the group and download the approved model at its pinned revision. The server loads it offline, so this step is needed once:

```bash
uv sync --group sentiment
uv run python -m financial_news.sandbox prefetch-model
```

### Run the MCP server

```bash
FINNHUB_API_KEY=<key> uv run python -m financial_news.sandbox server
```

To register it with Claude Code:

```bash
claude mcp add financial-news -- uv run --project /path/to/financial-news-mcp python -m financial_news.sandbox server
```

Starting `python -m financial_news.server` directly exits with "Refusing to start outside the network sandbox".

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
uv run python -m financial_news.sandbox monitor
```

### Configuration

Copy `config.example.toml` to `config.toml` to customise thresholds, watchlist tickers, sentiment model, Claude model and effort for the briefing and diagnostic agents, and email settings. `config.toml` is gitignored — do not commit it.

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
- Add real-time signal and algo attribution to measure whether news-volume and sentiment signals are actually improving trading outcomes. (A retrospective baseline comparison against historical data now exists — see [docs/evaluation.md](docs/evaluation.md) — but it doesn't yet measure live trading outcomes.)
- Grow the human-labeled evaluation set beyond the current small pilot to tighten the confidence interval on baseline comparisons.

## Docs

- Agent catalog and behavioral rules: [docs/AGENTS.md](docs/AGENTS.md), [docs/SKILL.md](docs/SKILL.md)
- Engineering standards: [docs/engineering-standards.md](docs/engineering-standards.md)
- Developer infra (packaging, install, test strategy): [docs/developer-infra.md](docs/developer-infra.md)
- Z-score iteration notes: [docs/z_score_iteration.md](docs/z_score_iteration.md)
- Baseline evaluation methodology and results: [docs/evaluation.md](docs/evaluation.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
