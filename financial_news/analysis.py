import logging
import os
import time
from collections import Counter
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import finnhub
import pandas as pd
import pandas_market_calendars as mcal

from financial_news.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration-sourced constants
# ---------------------------------------------------------------------------
# Values default to config.toml [analysis] section; fall back to dataclass
# defaults when the file is absent. Keep these as module-level names so tests
# and boundary checks can reference them directly.

_cfg = load_config()
THRESHOLD_ELEVATED: float = _cfg.analysis.threshold_elevated
THRESHOLD_UNUSUAL: float = _cfg.analysis.threshold_unusual
BASELINE_DAYS: int = _cfg.analysis.baseline_days

# Registry of numeric outputs from compute_volume_stats that should be exported
# as OTel gauges. Each entry is (stats_key, otel_metric_name, description).
# Add a row here whenever a new numeric field is added to the return dict —
# monitor.py iterates this to register and set gauges automatically.
GAUGE_SPECS: list[tuple[str, str, str]] = [
    (
        "z_score",
        "financial_news.z_score",
        "EWM z-score for 24h news volume vs 30-day baseline",
    ),
    (
        "recent_count",
        "financial_news.recent_count",
        "News articles published in the last 24h",
    ),
    (
        "mean",
        "financial_news.ewm_mean",
        "30-day EWM baseline mean article count",
    ),
]


def _validate_thresholds() -> None:
    """Guard: ensure thresholds are in valid order."""
    assert THRESHOLD_ELEVATED < THRESHOLD_UNUSUAL, (
        f"Invalid thresholds: elevated ({THRESHOLD_ELEVATED}) "
        f"must be < unusual ({THRESHOLD_UNUSUAL})"
    )
    logger.info(
        "Classification thresholds loaded: elevated=%.1f unusual=%.1f baseline_days=%d",
        THRESHOLD_ELEVATED,
        THRESHOLD_UNUSUAL,
        BASELINE_DAYS,
    )


api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise RuntimeError(
        "FINNHUB_API_KEY is not set. Set it in the environment before running "
        "the server. Example: export FINNHUB_API_KEY='your_key'."
    )

client = finnhub.Client(api_key=api_key)

_validate_thresholds()

# ---------------------------------------------------------------------------
# Exchange timezone lookup (via pandas_market_calendars)
# ---------------------------------------------------------------------------
# pmcal owns all timezone data.  This code only owns:
#   1. A dynamically-built alias map from pmcal's own .aliases attributes.
#   2. A small country-code fallback (unavoidable: pmcal has no country_code).


def _build_pmcal_alias_map() -> dict[str, str]:
    """Return {uppercase_alias: pmcal_calendar_name} built from pmcal aliases.

    Filters to purely-alphabetic aliases of length >= 3 and skips generic
    routing/index names that would produce misleading matches.
    """
    _skip = frozenset({"BATS", "DJIA", "DOW", "FX", "FOREX", "stock"})
    result: dict[str, str] = {}
    for name in mcal.get_calendar_names():
        try:
            cal = mcal.get_calendar(name)
            for alias in getattr(cal, "aliases", []):
                key = alias.upper()
                if key in _skip or not alias.isalpha() or len(alias) < 3:
                    continue
                if key not in result:
                    result[key] = name
        except Exception:
            pass
    return result


# Built once at import; pmcal is the authoritative source for timezone data.
_PMCAL_ALIASES: dict[str, str] = _build_pmcal_alias_map()


@lru_cache(maxsize=128)
def _exchange_tz(symbol: str) -> ZoneInfo:
    """Return the exchange timezone for *symbol* via Finnhub profile + pmcal."""
    try:
        profile = client.company_profile2(symbol=symbol)
        exchange = (profile.get("exchange") or "").upper()
        for alias, cal_name in _PMCAL_ALIASES.items():
            if alias in exchange:
                return mcal.get_calendar(cal_name).tz
    except Exception:
        logger.warning(
            "exchange timezone lookup failed for %s, defaulting to NYSE", symbol
        )
        return mcal.get_calendar("NYSE").tz
    logger.warning("no exchange timezone match for %s, defaulting to NYSE", symbol)
    return mcal.get_calendar("NYSE").tz


def _ewma_mean_std(values: list[float], span: int) -> tuple[float, float]:
    """Return the exponentially weighted mean and std over values (adjust=True)."""
    if not values:
        return 0.0, 0.0
    ewm = pd.Series(values, dtype=float).ewm(span=span, adjust=True)
    mean = float(ewm.mean().iloc[-1])
    std = ewm.std().iloc[-1]
    return mean, 0.0 if pd.isna(std) else float(std)


def fetch_news(symbol: str, from_date: date, to_date: date) -> list:
    logger.debug(
        "fetch_news symbol=%s from=%s to=%s",
        symbol,
        from_date.isoformat(),
        to_date.isoformat(),
    )
    start = time.time()
    try:
        response = client.company_news(
            symbol, _from=from_date.isoformat(), to=to_date.isoformat()
        )
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "fetch_news failed symbol=%s from=%s to=%s elapsed=%.2fs error=%s",
            symbol,
            from_date.isoformat(),
            to_date.isoformat(),
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
        "fetch_news symbol=%s from=%s to=%s articles_returned=%d elapsed=%.2fs",
        symbol,
        from_date.isoformat(),
        to_date.isoformat(),
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
    tz = _exchange_tz(symbol)
    today = datetime.now(tz).date()
    yesterday = today - timedelta(days=1)
    baseline_start = today - timedelta(days=BASELINE_DAYS)

    # Fetch with to_date+1 so Finnhub includes all of today regardless of how
    # it interprets the upper bound of a same-day range.
    recent_raw = fetch_news(symbol, from_date=today, to_date=today + timedelta(days=1))
    baseline_articles = fetch_news(symbol, from_date=baseline_start, to_date=yesterday)

    # Keep only articles that fall on today in the exchange's local timezone.
    recent = [
        a
        for a in recent_raw
        if datetime.fromtimestamp(a["datetime"], tz=tz).date() == today
    ]

    article_date_counts = Counter(
        datetime.fromtimestamp(article["datetime"], tz=tz).date()
        for article in baseline_articles
    )

    # One count per calendar day in the window; zero-fill days with no coverage.
    baseline_counts: list[float] = []
    current = baseline_start
    while current <= yesterday:
        baseline_counts.append(float(article_date_counts.get(current, 0)))
        current += timedelta(days=1)

    mean, std = _ewma_mean_std(baseline_counts, span=BASELINE_DAYS)

    recent_count = len(recent)
    z_score = calculate_z_score(recent_count, mean, std)

    if z_score < THRESHOLD_ELEVATED:
        classification = "normal"
    elif z_score < THRESHOLD_UNUSUAL:
        classification = "elevated"
    else:
        classification = "unusual"

    logger.info(
        "compute_volume_stats symbol=%s tz=%s recent_count=%d ewm_mean=%.2f "
        "ewm_std=%.2f z_score=%.2f classification=%s",
        symbol,
        tz.key,
        recent_count,
        mean,
        std,
        z_score,
        classification,
    )

    headlines = [article["headline"] for article in recent[:5]]
    recent_headlines = [article["headline"] for article in recent]
    articles = [
        {"headline": a["headline"], "summary": a.get("summary") or ""}
        for a in recent[:5]
    ]
    recent_articles = [
        {"headline": a["headline"], "summary": a.get("summary") or ""} for a in recent
    ]
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
        "recent_headlines": recent_headlines,
        "articles": articles,
        "recent_articles": recent_articles,
        "baseline_counts": baseline_counts,
    }
