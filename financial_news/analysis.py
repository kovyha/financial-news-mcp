import logging
import os
import time
from collections import Counter
from datetime import date, datetime, timedelta

import finnhub
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# These thresholds classify news volume as normal, elevated, or unusual.
# Changes to these values require explicit human review. See SKILL.md.

THRESHOLD_ELEVATED = 2  # z-score threshold for elevated news volume
THRESHOLD_UNUSUAL = 3  # z-score threshold for unusual news volume


def _validate_thresholds() -> None:
    """Guard: ensure thresholds are in valid order."""
    assert THRESHOLD_ELEVATED < THRESHOLD_UNUSUAL, (
        f"Invalid thresholds: elevated ({THRESHOLD_ELEVATED}) "
        f"must be < unusual ({THRESHOLD_UNUSUAL})"
    )
    logger.info(
        "Classification thresholds loaded: elevated=%.1f, unusual=%.1f",
        THRESHOLD_ELEVATED,
        THRESHOLD_UNUSUAL,
    )


api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise RuntimeError(
        "FINNHUB_API_KEY is not set. Set it in the environment before running "
        "the server. Example: export FINNHUB_API_KEY='your_key'."
    )

client = finnhub.Client(api_key=api_key)

_validate_thresholds()


def fetch_news(symbol: str, days: int) -> list:
    today = date.today()
    to_date = today.isoformat()
    from_date = (today - timedelta(days=days)).isoformat()
    logger.debug(
        "fetch_news symbol=%s days=%d from=%s to=%s", symbol, days, from_date, to_date
    )
    start = time.time()
    try:
        response = client.company_news(symbol, _from=from_date, to=to_date)
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "fetch_news failed symbol=%s days=%d elapsed=%.2fs error=%s",
            symbol,
            days,
            elapsed,
            exc,
        )
        raise RuntimeError(
            "Failed to fetch news from Finnhub. Check that FINNHUB_API_KEY is valid "
            "and that the upstream API is available."
        ) from exc
    result = response if response else []
    elapsed = time.time() - start
    logger.info(
        "fetch_news symbol=%s days=%d articles_returned=%d elapsed=%.2fs",
        symbol,
        days,
        len(result),
        elapsed,
    )
    return result


def calculate_z_score(recent_count: int, mean: float, std: float) -> float:
    """Calculate z-score with guards for zero mean/std.

    Behavior:
    - If mean == 0 and recent_count == 0 -> return 0
    - If mean == 0 and recent_count != 0 -> return +inf
    - If std == 0 -> return recent_count / mean
    - Otherwise -> (recent_count - mean) / std
    """
    if mean == 0 and recent_count == 0:
        return 0.0
    if mean == 0:
        return float("inf")
    if std == 0:
        return recent_count / mean
    return (recent_count - mean) / std


def compute_volume_stats(symbol: str) -> dict:
    """Fetch news and compute volume statistics for a symbol.

    Returns a dict with keys: recent_count, mean, std, z_score, classification,
    headlines, baseline_counts.
    """
    recent = fetch_news(symbol, days=1)
    baseline = fetch_news(symbol, days=7)

    baseline_counts = list(
        Counter(
            datetime.fromtimestamp(article["datetime"]).date() for article in baseline
        ).values()
    )
    if baseline_counts:
        mean = float(np.mean(baseline_counts))
        std = float(np.std(baseline_counts, ddof=1))
    else:
        mean = 0.0
        std = 0.0

    recent_count = len(recent)
    z_score = calculate_z_score(recent_count, mean, std)

    if z_score < THRESHOLD_ELEVATED:
        classification = "normal"
    elif z_score < THRESHOLD_UNUSUAL:
        classification = "elevated"
    else:
        classification = "unusual"

    logger.info(
        "compute_volume_stats symbol=%s recent_count=%d mean=%.2f std=%.2f "
        "z_score=%.2f classification=%s",
        symbol,
        recent_count,
        mean,
        std,
        z_score,
        classification,
    )

    headlines = [article["headline"] for article in recent[:5]]
    logger.debug(
        "compute_volume_stats symbol=%s headlines_passed_to_model=%s",
        symbol,
        headlines,
    )

    return {
        "recent_count": recent_count,
        "mean": mean,
        "std": std,
        "z_score": z_score,
        "classification": classification,
        "headlines": headlines,
        "baseline_counts": baseline_counts,
    }
