# financial-news-mcp

An MCP server that gives Claude the ability to detect statistically unusual news activity on any stock, with quantitative signal detection and contextual reasoning.

## Why This Exists

Financial news moves markets, but not all news activity is equal. This project detects when news volume on a stock is statistically unusual, giving traders and analysts an early signal that something significant may be happening before it is widely reflected in price.

Built on Model Context Protocol (MCP), it connects Claude directly to live financial news data and turns a natural-language question into a quantitative signal with contextual reasoning.

## What It Looks Like

Ask Claude: "Any unusual news activity on HSBC today?"

Claude can respond with something like: "Yes, HSBC is showing significant unusual news volume today. Z-score: 3.1. Article count is 3.4x the recent average. Likely driver: [reasoning based on the headlines and coverage]."

In practice, the output includes volume statistics, z-score significance, and a short explanation of what may be driving the spike.

## Architecture

This project is built as a Python MCP server with a small, focused signal-detection pipeline.

- `financial_news/server.py` exposes the MCP tool and orchestrates the workflow.
- Finnhub provides live company news data.
- NumPy is used to calculate the mean, standard deviation, and z-score.
- The z-score methodology is used to compare recent news activity against a recent baseline and classify it as normal, elevated, or unusual.
- Claude sits on top of that signal layer and can explain what the spike may mean in context.

### Deterministic and LLM layers

The architecture maintains an explicit boundary between two layers:

**Deterministic signal layer** (`get_news_volume`): fetches data from Finnhub, computes article counts, mean, standard deviation, and z-score using NumPy, and classifies the result against fixed thresholds (z < 2 = normal, z < 3 = elevated, z ≥ 3 = unusual). No model inference occurs here. The output is a structured, reproducible string.

**LLM reasoning layer** (Claude, via MCP): receives that structured output as tool context and interprets what a statistically significant spike may mean — drawing on the headlines, market context, and its broader knowledge.

#### Auditability

This boundary is intentional. In any regulated or auditable environment, it means you can demonstrate precisely:
- What data was fetched from Finnhub (dates, symbol, API parameters)
- What was computed deterministically (article counts, mean, std dev, z-score)
- Exactly where model judgement begins (the LLM reasoning layer receives structured input)

The deterministic layer is **fully reproducible and testable** (see `tests/test_calculate_z_score.py` and `tests/test_get_news_volume.py` for coverage at 98-99%).

**Important caveat:** Auditability assumes Finnhub API is trustworthy and accessible. The system includes a `health_check` tool to verify upstream API health. Comprehensive monitoring of upstream behavior is a future enhancement.

## Setup and usage

### Prerequisites

- Python 3.12+
- `uv`
- A `FINNHUB_API_KEY` environment variable

### Install

```bash
uv pip install -e .
```

### Run the MCP server

```bash
uv run python -m financial_news.server
```

### Example MCP usage

Once connected to Claude/Desktop or another MCP client, ask questions such as:

- "Any unusual news activity on HSBC today?"
- "Check whether Tesla has elevated news flow right now."
- "Why is there unusual news volume on Apple today?"

CI runs baseline lint and test checks on pushes and pull requests. For local validation steps, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Future enhancements

- Add sentiment analysis to help traders judge whether a news spike is likely to turn into a real price move, and if the bias is more likely up or down.
- Link the news signal to a kdb-based trading and order-flow view so traders can see news activity alongside positioning, execution, and immediate market reaction.
- Add real-time signal and algo attribution to measure whether news-volume and sentiment signals are actually improving trading outcomes, with LLM reasoning to explain changes in performance.

## Changelog / iterations

- Z-score iteration notes: [docs/z_score_iteration.md](docs/z_score_iteration.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Contributing

- Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md)

