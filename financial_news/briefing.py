"""Daily market briefing agent.

Computes news-volume and price-move z-scores for the configured watchlist, then
calls Claude (Opus 4.7) with get_news_headlines and get_price_intraday tools to
produce a plain-language briefing for any tickers showing elevated or unusual
activity. get_price_intraday is only offered for tickers already flagged
elevated/unusual on either signal — it's a same-day attribution drill-down
(lining up a specific timestamped headline against a specific intraday price
move), not a second detection signal.

Run via GitHub Actions (monitor.yaml, briefing step) or manually:
    FINNHUB_API_KEY=<key> \\
    ANTHROPIC_API_KEY=<key> \\
    uv run python -m financial_news.briefing
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import (
    log_setup,  # noqa: F401
    snapshot,
)
from financial_news.analysis import compute_volume_stats
from financial_news.config import load_config
from financial_news.email_report import send_run_summary
from financial_news.enrichment import EnrichmentConfig, enrich_stats
from financial_news.price import compute_price_stats, fetch_intraday_prices

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-7"
_MAX_TOKENS = 4096
_THINKING = {"type": "adaptive"}
_SYSTEM_PROMPT = (
    "You are producing a daily news-volume briefing for a financial watchlist.\n\n"
    "For any ticker classified as 'elevated' or 'unusual', use get_news_headlines"
    " to retrieve broader headline context and identify the likely driver."
    " Each ticker's stats also include a price-move z-score (today's return vs its"
    " own volatility baseline). Cross-reference the two signals: a news-volume"
    " anomaly alongside an elevated/unusual price move suggests the move looks"
    " news-driven; a price move without a matching news anomaly (or vice versa)"
    " is worth calling out as such rather than assumed to be related."
    " If a ticker's price move is elevated/unusual and today's headlines (each"
    " timestamped) suggest more than one plausible news-driven move that day, use"
    " get_price_intraday to see today's intraday price bars and attribute which"
    " headline lines up with which specific move."
    " Then write a concise briefing covering:\n"
    "1. Notable tickers with elevated or unusual activity and the likely driver\n"
    "2. Any cross-watchlist themes or patterns\n"
    "3. A brief note on quiet tickers\n\n"
    "If the entire watchlist is normal, say so clearly."
    " Keep the briefing factual and suitable for a daily internal review."
)


def _collect_stats(tickers: list[str]) -> list[dict]:
    """Compute volume and price stats for each ticker, returning result dicts."""
    results = []
    for ticker in tickers:
        try:
            stats = compute_volume_stats(ticker)
            stats["ticker"] = ticker
            logger.info(
                "ticker=%s z=%.2f classification=%s",
                ticker,
                stats["z_score"],
                stats["classification"],
            )
        except Exception:
            logger.exception("ticker=%s failed", ticker)
            results.append({"ticker": ticker, "error": "fetch failed"})
            continue

        try:
            stats["price"] = compute_price_stats(ticker)
        except Exception:
            logger.exception("ticker=%s price fetch failed", ticker)

        results.append(stats)
    return results


def _format_stats_for_prompt(stats: list[dict]) -> str:
    """Format enriched stats as a readable block for the LLM prompt."""
    sections = []
    for s in stats:
        if "error" in s:
            sections.append(f"**{s['ticker']}**: ERROR — fetch failed")
            continue
        selected = s.get("selected_articles") or []
        if selected and any(item["label"] != "unavailable" for item in selected):
            parts = []
            for item in selected:
                time_prefix = f"[{item['time']}] " if item.get("time") else ""
                line = (
                    f"- {time_prefix}[{item['label']} {item['score']:.2f}]"
                    f" {item['headline']}"
                )
                if item.get("summary"):
                    line += f"\n    {item['summary']}"
                parts.append(line)
            headline_text = "\n  ".join(parts)
        else:
            headlines = s.get("headlines", [])
            headline_text = (
                "\n  ".join(f"- {h}" for h in headlines)
                if headlines
                else "(no headlines today)"
            )
        price = s.get("price")
        price_text = (
            f", price_move={price['return_pct']:+.2%}"
            f" (z={price['z_score']:.2f}, {price['classification']})"
            if price
            else ""
        )
        sections.append(
            f"**{s['ticker']}**: z={s['z_score']:.2f}, count={s['recent_count']},"
            f" classification={s['classification']}{price_text}\n  {headline_text}"
        )
    return "\n\n".join(sections)


def _format_headline_articles(articles: list[dict], max_headlines: int) -> str:
    """Format pre-fetched headline articles as a readable string for tool results."""
    if not articles:
        return "No headline context available."
    lines = []
    for item in articles[:max_headlines]:
        label = item.get("label", "")
        score = item.get("score")
        sentiment = (
            f" [{label} {score:.2f}]"
            if label and label != "unavailable" and score is not None
            else ""
        )
        headline = item.get("headline", "no headline")
        source = item.get("source", "unknown")
        line = f"- [{item['date']}]{sentiment} {headline} ({source})"
        s = item.get("summary", "")
        if s:
            line += f"\n  {s}"
        lines.append(line)
    return "\n".join(lines)


def _build_headlines_cache(stats: list[dict], max_headlines: int) -> dict[str, str]:
    """Build a {ticker: formatted_headlines} dict from pre-fetched enriched stats."""
    return {
        s["ticker"]: _format_headline_articles(
            s.get("selected_headline_articles", []), max_headlines
        )
        for s in stats
        if "error" not in s
    }


def _is_flagged(s: dict) -> bool:
    """True if either the news or price signal is elevated/unusual for this ticker."""
    if s.get("classification") in ("elevated", "unusual"):
        return True
    price = s.get("price")
    return bool(price and price.get("classification") in ("elevated", "unusual"))


def _format_intraday_bars(bars: list[dict]) -> str:
    """Format intraday price bars as a readable string for tool results."""
    if not bars:
        return "No intraday price data available."
    lines = [
        f"- {bar['time']}: ${bar['price']:.2f}"
        f" ({bar['return_from_prev_close']:+.2%} vs previous close)"
        for bar in bars
    ]
    return "\n".join(lines)


def _build_intraday_cache(stats: list[dict]) -> dict[str, str]:
    """Build a {ticker: formatted_intraday_bars} dict, flagged tickers only.

    Only fetched for tickers already classified elevated/unusual on either
    signal — fetching intraday bars for the whole watchlist would multiply
    yfinance call volume for tickers that don't need same-day attribution.
    """
    cache: dict[str, str] = {}
    for s in stats:
        if "error" in s or not _is_flagged(s):
            continue
        ticker = s["ticker"]
        try:
            result = fetch_intraday_prices(ticker)
            cache[ticker] = _format_intraday_bars(result["bars"])
        except Exception:
            logger.exception("ticker=%s intraday fetch failed", ticker)
            cache[ticker] = "No intraday price data available."
    return cache


def _run_briefing(
    stats: list[dict],
    baseline_days: int,
    headline_days: int,
    headlines_cache: dict[str, str],
    intraday_cache: dict[str, str],
) -> str:
    """Call Claude to produce a daily market briefing over the collected stats."""
    client = anthropic.Anthropic()
    tools = [
        {
            "name": "get_news_headlines",
            "description": (
                f"Get the last {headline_days} days of news headlines for a stock"
                " ticker to understand what is driving unusual or elevated activity."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'NVDA'",
                    }
                },
                "required": ["ticker"],
            },
            "cache_control": {"type": "ephemeral"},
        },
        {
            "name": "get_price_intraday",
            "description": (
                "Get today's intraday price bars for a stock ticker to attribute a"
                " specific price move to a specific timestamped headline. Only"
                " useful for tickers whose price move or news volume is already"
                " classified elevated or unusual."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'NVDA'",
                    }
                },
                "required": ["ticker"],
            },
            "cache_control": {"type": "ephemeral"},
        },
    ]
    stats_block = _format_stats_for_prompt(stats)
    prompt = (
        f"Today is {datetime.now(timezone.utc).date().isoformat()} (UTC).\n\n"
        f"Watchlist statistics (z-score vs {baseline_days}-day EWM baseline,"
        f" today's headlines included):\n\n{stats_block}\n\n"
        f"Retrieve {headline_days}-day headline context for any elevated or unusual"
        " tickers before writing the briefing."
    )
    messages = [{"role": "user", "content": prompt}]
    while True:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            thinking=_THINKING,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "tool_use":
            return f"Briefing ended unexpectedly (stop_reason={response.stop_reason})."
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            ticker = block.input.get("ticker", "")
            if block.name == "get_price_intraday":
                content = intraday_cache.get(
                    ticker, "No intraday price data available."
                )
            else:
                content = headlines_cache.get(ticker, "No headline context available.")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        messages.append({"role": "user", "content": tool_results})


def main() -> int:
    cfg = load_config()
    stats = snapshot.read(Path(cfg.monitor.snapshot_path))
    if stats is None:
        stats = _collect_stats(cfg.monitor.tickers)

    enrich_cfg = EnrichmentConfig(
        model_name=cfg.sentiment.model_name,
        valid_labels=frozenset(cfg.sentiment.labels),
        confidence_threshold=cfg.briefing.confidence_threshold,
        min_articles=cfg.briefing.prompt_headlines_min,
        max_articles=cfg.briefing.prompt_headlines_max,
    )
    enriched = [enrich_stats(s, enrich_cfg) for s in stats]

    headlines_cache = _build_headlines_cache(enriched, cfg.briefing.max_headlines)
    intraday_cache = _build_intraday_cache(enriched)
    briefing_text = _run_briefing(
        enriched,
        cfg.analysis.baseline_days,
        cfg.analysis.headline_days,
        headlines_cache,
        intraday_cache,
    )
    print()
    print("=" * 60)
    print(f"DAILY BRIEFING — {datetime.now(timezone.utc).date().isoformat()}")
    print("=" * 60)
    print(briefing_text)
    print("=" * 60)
    print()

    if cfg.email is not None:
        good_stats = [s for s in enriched if "error" not in s]
        failed_tickers = [s["ticker"] for s in enriched if "error" in s]
        send_run_summary(cfg.email, good_stats, failed_tickers, briefing_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
