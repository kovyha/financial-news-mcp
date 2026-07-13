import logging
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import log_setup  # noqa: F401
from financial_news.analysis import client
from financial_news.config import load_config
from financial_news.enrichment import EnrichmentConfig, enrich_ticker
from financial_news.price import compute_price_stats, fetch_intraday_prices

logger = logging.getLogger(__name__)

mcp = FastMCP("financial-news")

_cfg = load_config()
_enrich_cfg = EnrichmentConfig(
    model_name=_cfg.sentiment.model_name,
    valid_labels=frozenset(_cfg.sentiment.labels),
    confidence_threshold=_cfg.briefing.confidence_threshold,
    min_articles=_cfg.briefing.prompt_headlines_min,
    max_articles=_cfg.briefing.prompt_headlines_max,
)


@mcp.tool()
def health_check() -> str:
    """Verify that upstream Finnhub API is accessible and healthy.

    This is a deterministic health check tool that attempts a simple API call
    and returns the status. Use this to diagnose connectivity issues before
    invoking get_news_volume.

    Returns:
        A status string indicating API health and timestamp of check.
    """
    logger.info("health_check called")
    start = time.time()
    try:
        response = client.company_news("AAPL", _from="2026-04-19", to="2026-04-20")
        elapsed = time.time() - start
        status = (
            f"✅ Finnhub API healthy\n"
            f"Last check: {datetime.now().isoformat()}\n"
            f"Response time: {elapsed:.2f}s\n"
            f"Test call returned {len(response) if response else 0} articles"
        )
        logger.info("health_check passed elapsed=%.2fs", elapsed)
        return status
    except Exception as exc:
        elapsed = time.time() - start
        status = (
            f"❌ Finnhub API unreachable\n"
            f"Last check: {datetime.now().isoformat()}\n"
            f"Error: {exc}"
        )
        logger.error("health_check failed elapsed=%.2fs error=%s", elapsed, exc)
        return status


@mcp.tool()
def get_news_volume(symbol: str) -> dict:
    """Detect unusual news volume for a stock symbol.

    Deterministic layer (this function):
      - Fetches raw article data from Finnhub over defined date windows.
      - Computes article counts, mean, standard deviation, and z-score.
      - Classifies the signal as normal / elevated / unusual against fixed thresholds.
      - Enriches articles with finBERT sentiment and applies confidence-threshold
        selection (with neutral-article filtering for elevated/unusual tickers).
      - Returns structured data: symbol, counts, statistics, classification, articles,
        and headline_context (last N days of articles for narrative context).
        Each item in `articles` carries a "time" (exchange-local HH:MM:SS) so a
        specific today's headline can be lined up against a specific intraday
        price move from get_price_intraday, when this ticker's price move is
        also elevated/unusual.

    LLM reasoning layer (the caller — Claude via MCP):
      - Receives the structured output above as tool result context.
      - Interprets what a statistically significant spike may mean given the articles,
        sentiment signals, and its broader knowledge.
      - All model judgement begins after this function returns; none occurs inside it.

    This boundary means the signal is fully auditable without involving a
    non-deterministic model: the inputs, computation, and classification can be
    reproduced and verified independently of any generative LLM inference.
    finBERT is a fixed-weight classification model — given the same input it
    produces the same output.
    """
    logger.info("get_news_volume called symbol=%s", symbol)

    enriched = enrich_ticker(symbol, _enrich_cfg)

    z = enriched["z_score"]
    return {
        "symbol": symbol,
        "recent_count": enriched["recent_count"],
        "ewm_mean": round(enriched["mean"], 2),
        "ewm_std": round(enriched["std"], 2),
        "z_score": z,
        "classification": enriched["classification"],
        "articles": [
            {
                "headline": item["headline"],
                "summary": item["summary"],
                "label": item["label"],
                "score": item["score"],
                "time": item.get("time", ""),
            }
            for item in enriched.get("selected_articles", [])
        ],
        "headline_context": enriched.get("selected_headline_articles", []),
    }


@mcp.tool()
def get_price_move(symbol: str) -> dict:
    """Detect an unusual price move for a stock symbol.

    Deterministic layer (this function):
      - Fetches daily price history and a live quote from yfinance.
      - Computes today's return vs previous close, and a volatility baseline
        (EWM mean/std) from prior daily returns.
      - Calculates a return z-score and classifies it as normal / elevated / unusual
        against fixed thresholds, applied to the z-score's magnitude (a large move
        in either direction is anomalous).
      - Returns structured data: symbol, current price, previous close, return,
        and the z-score/classification.

    LLM reasoning layer (the caller — Claude via MCP):
      - Receives the structured output above as tool result context.
      - Cross-references this against get_news_volume's output for the same symbol
        to judge whether an unusual price move looks news-driven.
      - All model judgement begins after this function returns; none occurs inside it.

    This boundary means the signal is fully auditable without involving a
    non-deterministic model: the inputs, computation, and classification can be
    reproduced and verified independently of any generative LLM inference.
    """
    logger.info("get_price_move called symbol=%s", symbol)
    return compute_price_stats(symbol)


@mcp.tool()
def get_price_intraday(symbol: str) -> dict:
    """Fetch today's intraday price bars for a stock symbol.

    Deterministic layer (this function):
      - Fetches today's intraday price bars from yfinance at a configurable bar
        size ([price] intraday_interval in config.toml, default 5-minute).
      - Returns each bar's exchange-local time, price, and cumulative return
        since the previous close.

    This is a same-day attribution drill-down, not a second anomaly-detection
    signal — call it only after get_price_move (or get_news_volume) flags today
    as elevated/unusual, to line up a specific intraday move against a specific
    today-timestamped headline from get_news_volume's `articles` (each of which
    carries a "time" field for exactly this purpose). Same-day multiple price
    moves (e.g. a pre-market gap vs. an unrelated afternoon move) are exactly the
    case this tool exists to disambiguate — the daily z-score from get_price_move
    only tells you *that* today looks unusual, not *when* during the day or
    *which* headline plausibly caused it.

    LLM reasoning layer (the caller — Claude via MCP):
      - Cross-references bar timestamps against get_news_volume's timestamped
        articles to attribute intraday moves to specific news.
      - All model judgement begins after this function returns; none occurs
        inside it.

    This boundary means the signal is fully auditable without involving a
    non-deterministic model: the inputs and bar computation can be reproduced
    and verified independently of any generative LLM inference.
    """
    logger.info("get_price_intraday called symbol=%s", symbol)
    result = fetch_intraday_prices(symbol)
    return {"symbol": symbol, **result}


if __name__ == "__main__":
    mcp.run()
