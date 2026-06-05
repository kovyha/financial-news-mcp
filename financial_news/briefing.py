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
    sentiment,
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


def _enrich_stats_with_sentiment(
    stats: list[dict],
    model_name: str,
    valid_labels: frozenset[str],
) -> list[dict]:
    """Add headline_sentiment field to each stat entry using finBERT scoring.

    Scores all headlines per ticker, drawn from recent_headlines (all today's
    articles) if present, falling back to the top-5 headlines field otherwise.
    Tickers with no headlines (zero-news baseline) receive an empty
    headline_sentiment list.
    """
    enriched = []
    for s in stats:
        if "error" in s:
            enriched.append(s)
            continue
        recent = s.get("recent_headlines") or []
        headlines = s.get("headlines") or []
        source = recent or headlines
        logger.info(
            "ticker=%s recent_headlines=%d headlines=%d source=%d",
            s.get("ticker", "?"),
            len(recent),
            len(headlines),
            len(source),
        )
        if not source:
            logger.info("ticker=%s has no headlines to score", s.get("ticker", "?"))
            enriched.append({**s, "headline_sentiment": []})
            continue
        scored = sentiment.score_headlines(source, model_name, valid_labels)
        label_counts = {}
        for item in scored:
            label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
        logger.info(
            "ticker=%s scored %d headlines label_counts=%s",
            s.get("ticker", "?"),
            len(scored),
            label_counts,
        )
        enriched.append({**s, "headline_sentiment": scored})
    return enriched


def _select_prompt_headlines(
    scored: list[dict],
    confidence_threshold: float,
    min_headlines: int,
    max_headlines: int,
) -> list[dict]:
    """Select headlines for the LLM prompt using confidence-threshold filtering.

    Returns headlines with score >= confidence_threshold, clamped to
    [min_headlines, max_headlines]. When too few meet the threshold the
    top-scoring headlines fill up to min_headlines regardless of threshold.
    """
    by_score = sorted(scored, key=lambda x: x["score"], reverse=True)
    confident = [item for item in by_score if item["score"] >= confidence_threshold]
    if len(confident) < min_headlines:
        return by_score[:min_headlines]
    return confident[:max_headlines]


def _format_stats_for_prompt(
    stats: list[dict],
    confidence_threshold: float,
    min_headlines: int,
    max_headlines: int,
) -> str:
    """Format collected stats as a readable block for the LLM prompt."""
    sections = []
    for s in stats:
        if "error" in s:
            sections.append(f"**{s['ticker']}**: ERROR — fetch failed")
            continue
        scored = s.get("headline_sentiment")
        if scored and any(item["label"] != "unavailable" for item in scored):
            candidates = scored
            if s.get("classification") in ("elevated", "unusual"):
                non_neutral = [
                    item
                    for item in scored
                    if item["label"] != "neutral"
                    or item["score"] < confidence_threshold
                ]
                if non_neutral:
                    candidates = non_neutral
            top = _select_prompt_headlines(
                candidates, confidence_threshold, min_headlines, max_headlines
            )
            for item in top:
                logger.info(
                    "prompt_headline ticker=%s sentiment_label=%s"
                    " sentiment_score=%.4f headline=%r",
                    s["ticker"],
                    item["label"],
                    item["score"],
                    item["headline"],
                )
            headline_text = "\n  ".join(
                f"- [{item['label']} {item['score']:.2f}] {item['headline']}"
                for item in top
            )
        else:
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
    stats: list[dict],
    baseline_days: int,
    headline_days: int,
    max_headlines: int,
    confidence_threshold: float,
    prompt_headlines_min: int,
    prompt_headlines_max: int,
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
    stats_block = _format_stats_for_prompt(
        stats, confidence_threshold, prompt_headlines_min, prompt_headlines_max
    )
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
    stats = _enrich_stats_with_sentiment(
        stats,
        cfg.sentiment.model_name,
        frozenset(cfg.sentiment.labels),
    )
    briefing_text = _run_briefing(
        stats,
        cfg.analysis.baseline_days,
        cfg.briefing.headline_days,
        cfg.briefing.max_headlines,
        cfg.briefing.confidence_threshold,
        cfg.briefing.prompt_headlines_min,
        cfg.briefing.prompt_headlines_max,
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
