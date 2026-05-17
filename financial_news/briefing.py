"""Daily market briefing agent.

Computes news-volume z-scores for the configured watchlist, then calls
Claude (Opus 4.7) with a get_news_headlines tool to produce a plain-language
briefing for any tickers showing elevated or unusual activity.

Run via GitHub Actions (monitor.yaml, briefing step) or manually:
    FINNHUB_API_KEY=<key> \\
    ANTHROPIC_API_KEY=<key> \\
    uv run python -m financial_news.briefing
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import (
    log_setup,  # noqa: F401
    snapshot,
)
from financial_news.analysis import compute_volume_stats, fetch_news
from financial_news.config import load_config
from financial_news.email_report import send_run_summary

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-7"
_MAX_TOKENS = 4096
_THINKING = {"type": "adaptive"}


def _collect_stats(tickers: list[str], z_score_cap: float) -> list[dict]:
    """Compute volume stats for each ticker, returning a list of result dicts."""
    results = []
    for ticker in tickers:
        try:
            stats = compute_volume_stats(ticker)
            stats["z_score"] = min(stats["z_score"], z_score_cap)
            stats["ticker"] = ticker
            results.append(stats)
            logger.info(
                "ticker=%s z=%.2f classification=%s",
                ticker,
                stats["z_score"],
                stats["classification"],
            )
        except Exception:
            logger.exception("ticker=%s failed", ticker)
            results.append({"ticker": ticker, "error": "fetch failed"})
    return results


def _format_stats_for_prompt(stats: list[dict]) -> str:
    """Format collected stats as a readable block for the LLM prompt."""
    sections = []
    for s in stats:
        if "error" in s:
            sections.append(f"**{s['ticker']}**: ERROR — fetch failed")
            continue
        headlines = s.get("headlines", [])
        headline_text = (
            "\n  ".join(f"- {h}" for h in headlines)
            if headlines
            else "(no headlines today)"
        )
        sections.append(
            f"**{s['ticker']}**: z={s['z_score']:.2f}, count={s['recent_count']},"
            f" classification={s['classification']}\n  {headline_text}"
        )
    return "\n\n".join(sections)


def _fetch_headlines_for_tool(
    ticker: str, headline_days: int, max_headlines: int
) -> str:
    """Fetch recent headline context for use as a tool result."""
    today = datetime.now(timezone.utc).date()
    from_date = today - timedelta(days=headline_days)
    try:
        news = fetch_news(ticker, from_date=from_date, to_date=today)
    except Exception as exc:
        return f"Error fetching news for {ticker}: {exc}"
    if not news:
        return f"No news found for {ticker} in the last {headline_days} days."
    lines = []
    for item in news[:max_headlines]:
        dt = (
            datetime.fromtimestamp(item["datetime"], tz=timezone.utc).date().isoformat()
        )
        h = item.get("headline", "no headline")
        src = item.get("source", "unknown")
        lines.append(f"- [{dt}] {h} ({src})")
    return "\n".join(lines)


def _run_briefing(
    stats: list[dict], baseline_days: int, headline_days: int, max_headlines: int
) -> str:
    """Call Claude to produce a daily market briefing over the collected stats."""
    client = anthropic.Anthropic()
    tools = [
        {
            "name": "get_news_headlines",
            "description": (
                f"Fetch the last {headline_days} days of news headlines for a stock"
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
        }
    ]
    stats_block = _format_stats_for_prompt(stats)
    prompt = (
        f"Today is {datetime.now(timezone.utc).date().isoformat()} (UTC)."
        " You are producing a daily news-volume "
        "briefing for a financial watchlist.\n\n"
        f"Watchlist statistics (z-score vs {baseline_days}-day EWM baseline,"
        f" today's headlines included):\n\n{stats_block}\n\n"
        "For any ticker classified as 'elevated' or 'unusual', use"
        f" get_news_headlines to fetch broader {headline_days}-day headline context"
        " and identify the likely driver. Then write a concise briefing covering:\n"
        "1. Notable tickers with elevated or unusual activity and the likely driver\n"
        "2. Any cross-watchlist themes or patterns\n"
        "3. A brief note on quiet tickers\n\n"
        "If the entire watchlist is normal, say so clearly. "
        "Keep the briefing factual and suitable for a daily internal review."
    )
    messages = [{"role": "user", "content": prompt}]
    while True:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            thinking=_THINKING,
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
            content = _fetch_headlines_for_tool(ticker, headline_days, max_headlines)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        messages.append({"role": "user", "content": tool_results})


def main() -> int:
    cfg = load_config()
    stats = snapshot.read(Path(cfg.monitor.snapshot_path))
    if stats is None:
        stats = _collect_stats(cfg.monitor.tickers, cfg.analysis.z_score_cap)
    briefing_text = _run_briefing(
        stats,
        cfg.analysis.baseline_days,
        cfg.briefing.headline_days,
        cfg.briefing.max_headlines,
    )
    print()
    print("=" * 60)
    print(f"DAILY BRIEFING — {datetime.now(timezone.utc).date().isoformat()}")
    print("=" * 60)
    print(briefing_text)
    print("=" * 60)
    print()

    if cfg.email is not None:
        good_stats = [s for s in stats if "error" not in s]
        failed_tickers = [s["ticker"] for s in stats if "error" in s]
        send_run_summary(cfg.email, good_stats, failed_tickers, briefing_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
