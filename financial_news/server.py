import logging
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import log_setup  # noqa: F401
from financial_news.analysis import BASELINE_DAYS, client, compute_volume_stats

logger = logging.getLogger(__name__)

mcp = FastMCP("financial-news")


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
def get_news_volume(symbol: str) -> str:
    """Detect unusual news volume for a stock symbol.

    Deterministic layer (this function):
      - Fetches raw article data from Finnhub over defined date windows.
      - Computes article counts, mean, standard deviation, and z-score using NumPy.
      - Classifies the signal as normal / elevated / unusual against fixed thresholds.
      - Returns a structured string: symbol, counts, statistics, classification,
        headlines.

    LLM reasoning layer (the caller — Claude via MCP):
      - Receives the structured output above as tool result context.
      - Interprets what a statistically significant spike may mean given the headlines,
        market context, and any other information available to it.
      - All model judgement begins after this function returns; none occurs inside it.

    This boundary means the signal is fully auditable without involving the model:
    the inputs, computation, and classification can be reproduced and verified
    independently of any LLM inference.
    """
    logger.info("get_news_volume called symbol=%s", symbol)

    stats = compute_volume_stats(symbol)
    recent_count = stats["recent_count"]
    mean = stats["mean"]
    std = stats["std"]
    z_score = stats["z_score"]
    classification = stats["classification"]
    headlines = stats["headlines"]

    summary_lines = [
        f"Symbol: {symbol}",
        f"News articles (last 24hrs): {recent_count}",
        f"Mean ({BASELINE_DAYS}-day EWM): {mean:.1f}",
        f"Std Dev ({BASELINE_DAYS}-day EWM): {std:.1f}",
        f"Z-score: {z_score:.1f}",
    ]

    if classification == "normal":
        summary_lines.append("✅ Normal news volume")
    elif classification == "elevated":
        summary_lines.append("⚠️ Elevated news volume")
    else:
        summary_lines.append("🚨 Unusual news volume detected")

    if recent_count == 0 and mean == 0.0:
        summary_lines.append(
            "No news data found for this symbol. This may mean the ticker is "
            "invalid, unsupported, or simply has no recent coverage."
        )

    summary_lines.append("")
    summary_lines.append("Recent headlines:")
    summary_lines.extend(f"- {headline}" for headline in headlines)

    return "\n".join(summary_lines) + "\n"


if __name__ == "__main__":
    mcp.run()
