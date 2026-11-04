"""Daily market briefing agent.

Computes news-volume and price-move z-scores for the configured watchlist, then
calls Claude (model from [llm] config) with get_news_headlines and
get_price_intraday tools to produce a plain-language briefing for any tickers
showing elevated or unusual activity. get_price_intraday is only offered for
tickers already flagged elevated/unusual on either signal — it's a same-day
attribution drill-down (lining up a specific timestamped headline against a
specific intraday price move), not a second detection signal.

Run via GitHub Actions (monitor.yaml, briefing step) or manually:
    FINNHUB_API_KEY=<key> \\
    ANTHROPIC_API_KEY=<key> \\
    uv run python -m financial_news.briefing
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import (
    log_setup,  # noqa: F401
    snapshot,
)
from financial_news.analysis import compute_volume_stats
from financial_news.config import LLMConfig, load_config
from financial_news.email_report import send_run_summary
from financial_news.enrichment import EnrichmentConfig, enrich_stats
from financial_news.llm import ToolLoopResult, run_tool_loop
from financial_news.price import compute_price_stats, fetch_intraday_prices
from financial_news.telemetry import build_meter_provider, grafana_configured

logger = logging.getLogger(__name__)

# Headlines listed per quiet (not elevated/unusual) ticker in the briefing prompt.
_QUIET_HEADLINES = 3

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


def _shows_scored_articles(s: dict) -> bool:
    """True if the prompt lists this ticker's finBERT-selected articles.

    False means finBERT was unavailable, so the prompt falls back to plain headlines.
    """
    selected = s.get("selected_articles") or []
    return any(item["label"] != "unavailable" for item in selected)


def _articles_for_prompt(s: dict) -> list[dict]:
    """Today's articles the prompt lists for this ticker.

    A flagged ticker gets every finBERT-selected article. A quiet ticker only
    needs a brief note, so it gets its top _QUIET_HEADLINES: positive/negative
    articles first, since finBERT's score is label confidence, not importance,
    and a quiet ticker's most confident articles are usually routine neutral news.
    Without finBERT, falls back to plain headlines.
    """
    if _shows_scored_articles(s):
        items = s["selected_articles"]
    else:
        items = [{"headline": h} for h in s.get("headlines", [])]
    if _is_flagged(s):
        return items
    # Stable sort keeps the score order within each group. "unavailable" (finBERT
    # gave an unexpected label) carries no direction, so it ranks with neutral.
    ranked = sorted(
        items, key=lambda item: item.get("label") not in ("positive", "negative")
    )
    return ranked[:_QUIET_HEADLINES]


def _headlines_in_prompt(s: dict) -> set[str]:
    """Headlines _format_stats_for_prompt already shows for this ticker."""
    return {item["headline"] for item in _articles_for_prompt(s)}


def _format_stats_for_prompt(stats: list[dict]) -> str:
    """Format enriched stats as a readable block for the LLM prompt.

    Quiet tickers get headlines only; summaries are kept for flagged tickers.
    """
    sections = []
    for s in stats:
        if "error" in s:
            sections.append(f"**{s['ticker']}**: ERROR — fetch failed")
            continue
        items = _articles_for_prompt(s)
        with_summaries = _is_flagged(s)
        if not items:
            headline_text = "(no headlines today)"
        elif _shows_scored_articles(s):
            parts = []
            for item in items:
                time_prefix = f"[{item['time']}] " if item.get("time") else ""
                line = (
                    f"- {time_prefix}[{item['label']} {item['score']:.2f}]"
                    f" {item['headline']}"
                )
                if with_summaries and item.get("summary"):
                    line += f"\n    {item['summary']}"
                parts.append(line)
            headline_text = "\n  ".join(parts)
        else:
            headline_text = "\n  ".join(f"- {item['headline']}" for item in items)
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
    """Build a {ticker: formatted_headlines} dict from pre-fetched enriched stats.

    The multi-day window includes today, so articles the prompt already shows are
    dropped rather than sent to Claude a second time.
    """
    cache: dict[str, str] = {}
    for s in stats:
        if "error" in s:
            continue
        window = s.get("selected_headline_articles", [])
        shown = _headlines_in_prompt(s)
        unseen = [item for item in window if item.get("headline") not in shown]
        if window and not unseen:
            cache[s["ticker"]] = (
                "No headlines beyond those already listed in the watchlist statistics."
            )
        else:
            cache[s["ticker"]] = _format_headline_articles(unseen, max_headlines)
    return cache


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


def _quiet_briefing(stats: list[dict]) -> str:
    """Briefing for a day with no elevated/unusual ticker, written without Claude.

    With nothing flagged, the briefing would only restate that every ticker is
    normal, so the Claude call is skipped.
    """
    good = [s for s in stats if "error" not in s]
    normal = [s["ticker"] for s in good]
    no_price = [s["ticker"] for s in good if not s.get("price")]
    failed = [s["ticker"] for s in stats if "error" in s]
    lines = []
    if normal:
        lines.append(
            "No elevated or unusual news-volume or price activity today."
            f" All {len(normal)} tickers are normal: {', '.join(normal)}."
        )
    if no_price:
        lines.append(
            f"Price signal unavailable (news volume only): {', '.join(no_price)}."
        )
    if failed:
        lines.append(f"Fetch failed: {', '.join(failed)}.")
    return "\n".join(lines) or "No watchlist data available."


def _run_briefing(
    stats: list[dict],
    baseline_days: int,
    headline_days: int,
    headlines_cache: dict[str, str],
    intraday_cache: dict[str, str],
    llm: LLMConfig | None = None,
) -> ToolLoopResult:
    """Call Claude to produce a daily market briefing over the collected stats."""
    tools = [
        {
            "name": "get_news_headlines",
            "description": (
                f"Get the last {headline_days} days of news headlines for a stock"
                " ticker to understand what is driving unusual or elevated activity."
                " Articles already listed in the watchlist statistics are omitted."
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
        },
    ]
    stats_block = _format_stats_for_prompt(stats)
    prompt = (
        f"Today is {datetime.now(timezone.utc).date().isoformat()} (UTC).\n\n"
        f"Watchlist statistics (z-score vs {baseline_days}-day EWM baseline,"
        " today's headlines included; normal tickers list only their top"
        f" {_QUIET_HEADLINES} headlines):\n\n{stats_block}\n\n"
        f"Retrieve {headline_days}-day headline context for any elevated or unusual"
        " tickers before writing the briefing."
    )

    def answer_tool(block) -> str:
        ticker = block.input.get("ticker", "")
        if block.name == "get_price_intraday":
            return intraday_cache.get(ticker, "No intraday price data available.")
        return headlines_cache.get(ticker, "No headline context available.")

    return run_tool_loop(
        _SYSTEM_PROMPT, prompt, tools, answer_tool, llm or LLMConfig(), "Briefing"
    )


def _export_usage(result: ToolLoopResult | None, flagged: int) -> None:
    """Send today's briefing cost metrics to Grafana Cloud, where alerts watch them.

    result is None on a quiet day (Claude not called): only the flagged-ticker
    count is sent. Ratios are computed here rather than in Grafana because the
    briefing reports only a few samples a day, too sparse for PromQL to divide
    one series by another reliably. Skipped when Grafana Cloud isn't configured
    (local runs), and a failed export never fails the briefing.
    """
    if not grafana_configured():
        logger.info("briefing usage export skipped: Grafana Cloud not configured")
        return
    try:
        provider = build_meter_provider()
        meter = provider.get_meter("financial_news.briefing")

        def send(name: str, description: str, value: float | None) -> None:
            if value is not None:
                gauge = meter.create_gauge(
                    f"financial_news.{name}", description=description
                )
                gauge.set(value)

        send(
            "briefing_flagged_tickers",
            "Watchlist tickers elevated/unusual on news or price today",
            flagged,
        )
        if result is not None:
            tokens = meter.create_gauge(
                "financial_news.briefing_tokens",
                description="Claude tokens used by today's briefing, by billing type",
            )
            for token_type, count in result.usage.items():
                tokens.set(count, {"type": token_type})
            send(
                "briefing_turns",
                "Claude calls made by today's briefing tool loop",
                result.turns,
            )
            send(
                "briefing_cache_hit_ratio",
                "Share of the briefing's input tokens read from cache",
                result.cache_hit_ratio,
            )
            cost = result.cost_usd
            send(
                "briefing_cost_usd",
                "Estimated Claude cost of today's briefing (USD)",
                cost,
            )
            send(
                "briefing_cost_per_flagged_ticker_usd",
                "Estimated briefing cost per flagged ticker (USD)",
                None if cost is None else cost / max(flagged, 1),
            )
        provider.force_flush()
        provider.shutdown()
        logger.info("briefing usage exported flagged=%d", flagged)
    except Exception:
        logger.exception("briefing usage export failed")


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

    flagged = sum(_is_flagged(s) for s in enriched)
    if flagged:
        headlines_cache = _build_headlines_cache(enriched, cfg.briefing.max_headlines)
        intraday_cache = _build_intraday_cache(enriched)
        result = _run_briefing(
            enriched,
            cfg.analysis.baseline_days,
            cfg.analysis.headline_days,
            headlines_cache,
            intraday_cache,
            cfg.llm,
        )
        briefing_text = result.text
    else:
        logger.info("briefing llm_skipped=true reason=no_flagged_tickers")
        result = None
        briefing_text = _quiet_briefing(enriched)
    _export_usage(result, flagged)
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
