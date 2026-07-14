"""Price-move signal: a volatility-normalized return z-score per ticker.

Deterministic counterpart to analysis.py's news-volume z-score, expressed in the
same "z-score vs baseline" units so the two signals can be directly compared by
the LLM reasoning layer. Intentionally decoupled from analysis.py: it does not
depend on Finnhub or the exchange-timezone lookup in analysis.py, since yfinance's
daily bars are already exchange-calendar-aware and fast_info carries its own
timezone.
"""

import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from financial_news.config import load_config
from financial_news.timeseries import ewma_mean_std

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration-sourced constants
# ---------------------------------------------------------------------------
# Values default to config.toml [price] section; fall back to dataclass defaults
# when the file is absent. Keep these as module-level names so tests and boundary
# checks can reference them directly.

_cfg = load_config()
PRICE_BASELINE_DAYS: int = _cfg.price.baseline_days
PRICE_THRESHOLD_ELEVATED: float = _cfg.price.threshold_elevated
PRICE_THRESHOLD_UNUSUAL: float = _cfg.price.threshold_unusual
PRICE_Z_SCORE_CAP: float = _cfg.price.z_score_cap
PRICE_WINSORIZE_K: float = _cfg.price.winsorize_k
PRICE_INTRADAY_INTERVAL: str = _cfg.price.intraday_interval

# Registry of numeric outputs from compute_price_stats that should be exported
# as OTel gauges. Each entry is (stats_key, otel_metric_name, description).
# monitor.py iterates this to register and set gauges automatically.
PRICE_GAUGE_SPECS: list[tuple[str, str, str]] = [
    (
        "z_score",
        "financial_news.price_z_score",
        "Return z-score for today's price move vs EWM baseline volatility",
    ),
    (
        "return_pct",
        "financial_news.price_return_pct",
        "Latest daily return vs previous close",
    ),
    (
        "mean",
        "financial_news.price_ewm_mean",
        "N-day EWM baseline mean daily return (two-sided winsorized)",
    ),
]


def _validate_price_thresholds() -> None:
    """Guard: ensure price thresholds are in valid order."""
    assert PRICE_THRESHOLD_ELEVATED < PRICE_THRESHOLD_UNUSUAL, (
        f"Invalid price thresholds: elevated ({PRICE_THRESHOLD_ELEVATED}) "
        f"must be < unusual ({PRICE_THRESHOLD_UNUSUAL})"
    )
    logger.info(
        "Price classification thresholds loaded: elevated=%.1f unusual=%.1f "
        "baseline_days=%d",
        PRICE_THRESHOLD_ELEVATED,
        PRICE_THRESHOLD_UNUSUAL,
        PRICE_BASELINE_DAYS,
    )


_validate_price_thresholds()


def _winsorize_two_sided(values: list[float], k: float) -> list[float]:
    """Cap both-tail outliers in *values* at median +/- k * MAD.

    Two-sided: unlike article counts, daily returns move in either direction, so
    a single extreme historical day (crash or spike) in either tail could
    otherwise inflate the EWM std and mask a genuinely unusual day. If MAD is 0
    (no dispersion around the median), values are returned unchanged.
    """
    if not values:
        return values
    series = pd.Series(values, dtype=float)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return values
    lower = median - k * mad
    upper = median + k * mad
    return series.clip(lower=lower, upper=upper).tolist()


def calculate_price_z_score(return_pct: float, mean: float, std: float) -> float:
    """Calculate a return z-score, guarding only against zero variance.

    Unlike analysis.calculate_z_score, a zero baseline *mean* is not treated as a
    special case here: an average daily return near 0 is an ordinary outcome for
    a random-walk price series, not a signal of missing baseline data. Only a
    zero *std* (no variance to compare against) is guarded, returning 0.0.
    """
    if std == 0:
        return 0.0
    return (return_pct - mean) / std


def _fetch_price_data(
    symbol: str, as_of_date: date | None = None
) -> tuple[pd.DataFrame, object | None]:
    """Fetch daily price history via yfinance for *symbol*.

    as_of_date=None (default, live mode): period-based history plus fast_info for
    the live quote. as_of_date given (historical backtesting mode, used by
    financial_news.evaluation): history fetched over an explicit date range ending
    at as_of_date instead; there is no live quote for a past date, so fast_info is
    None and compute_price_stats resolves current/previous close from the bars
    themselves.
    """
    logger.debug("fetch_price_data symbol=%s as_of_date=%s", symbol, as_of_date)
    start = time.time()
    try:
        ticker = yf.Ticker(symbol)
        if as_of_date is None:
            history = ticker.history(
                period=f"{PRICE_BASELINE_DAYS + 15}d", interval="1d", auto_adjust=True
            )
            fast_info = ticker.fast_info
        else:
            fetch_start = as_of_date - timedelta(days=PRICE_BASELINE_DAYS + 15)
            fetch_end = as_of_date + timedelta(days=1)
            history = ticker.history(
                start=fetch_start.isoformat(),
                end=fetch_end.isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            fast_info = None
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "fetch_price_data failed symbol=%s elapsed=%.2fs error=%s",
            symbol,
            elapsed,
            exc,
        )
        raise RuntimeError(
            "Failed to fetch price data from yfinance. Check that the symbol is "
            "valid and that the upstream API is available."
        ) from exc
    elapsed = time.time() - start
    logger.info(
        "fetch_price_data symbol=%s rows=%d elapsed=%.2fs",
        symbol,
        len(history),
        elapsed,
    )
    return history, fast_info


def compute_price_stats(symbol: str, as_of_date: date | None = None) -> dict:
    """Fetch price history and compute return statistics for a symbol.

    as_of_date=None (default): live mode — "today" and the current/previous close
    come from the live quote (fast_info), matching the get_price_move MCP tool's
    existing behavior exactly. as_of_date given: historical backtesting mode — the
    same z-score math computed as of that past date instead, resolving current/
    previous close from daily bars rather than a live quote. This parameter exists
    so financial_news.evaluation can build a labeled historical dataset without
    duplicating this function's math in a parallel module.

    Returns a dict with keys: symbol, current_price, previous_close, return_pct,
    mean, std, z_score, classification, baseline_days.
    """
    history, fast_info = _fetch_price_data(symbol, as_of_date)
    closes = history["Close"]

    if as_of_date is None:
        tz = ZoneInfo(fast_info.timezone)
        today = datetime.now(tz).date()
        current_price = float(fast_info.last_price)
        previous_close = float(fast_info.previous_close)
    else:
        today = as_of_date
        on_or_before = closes[closes.index.date <= today]
        if len(on_or_before) < 2:
            raise RuntimeError(
                f"Not enough price history for {symbol} as of {today.isoformat()} "
                "to compute a previous close."
            )
        current_price = float(on_or_before.iloc[-1])
        previous_close = float(on_or_before.iloc[-2])

    return_pct = (
        (current_price - previous_close) / previous_close if previous_close else 0.0
    )

    completed = closes[closes.index.date < today]

    baseline_returns = completed.pct_change().dropna().tolist()[-PRICE_BASELINE_DAYS:]

    winsorized_returns = _winsorize_two_sided(baseline_returns, k=PRICE_WINSORIZE_K)
    mean, std = ewma_mean_std(winsorized_returns, span=PRICE_BASELINE_DAYS)

    z_score = calculate_price_z_score(return_pct, mean, std)
    z_score = max(-PRICE_Z_SCORE_CAP, min(z_score, PRICE_Z_SCORE_CAP))

    if abs(z_score) < PRICE_THRESHOLD_ELEVATED:
        classification = "normal"
    elif abs(z_score) < PRICE_THRESHOLD_UNUSUAL:
        classification = "elevated"
    else:
        classification = "unusual"

    logger.info(
        "compute_price_stats symbol=%s current=%.2f prev_close=%.2f "
        "return_pct=%.4f ewm_mean=%.5f ewm_std=%.5f z_score=%.2f classification=%s",
        symbol,
        current_price,
        previous_close,
        return_pct,
        mean,
        std,
        z_score,
        classification,
    )

    return {
        "symbol": symbol,
        "current_price": current_price,
        "previous_close": previous_close,
        "return_pct": return_pct,
        "mean": mean,
        "std": std,
        "z_score": z_score,
        "classification": classification,
        "baseline_days": PRICE_BASELINE_DAYS,
    }


def fetch_intraday_prices(symbol: str) -> dict:
    """Fetch today's intraday price bars for *symbol* via yfinance.

    Returns {date, bars}: date is the exchange-local calendar day the bars belong
    to (derived here, the same way compute_price_stats derives "today", rather than
    left for the caller to guess independently); bars is a list of
    {time, price, return_from_prev_close} dicts (exchange-local HH:MM:SS,
    chronological order). This is a same-day attribution aid, not part of the
    anomaly-detection signal in compute_price_stats — intended to be called only
    when a ticker's price move is already flagged elevated/unusual, to line up a
    specific intraday move against a specific timestamped news article (see
    analysis.compute_volume_stats's "time" field on today's articles).

    intraday_interval (config: [price] intraday_interval) is passed straight through
    to yfinance rather than validated against a whitelist here, since yfinance's set
    of supported values isn't owned by this project. An invalid value, a non-trading
    day, or an upstream failure all surface as a RuntimeError.
    """
    logger.debug(
        "fetch_intraday_prices symbol=%s interval=%s", symbol, PRICE_INTRADAY_INTERVAL
    )
    start = time.time()
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1d", interval=PRICE_INTRADAY_INTERVAL)
        previous_close = float(ticker.fast_info.previous_close)
        tz = ZoneInfo(ticker.fast_info.timezone)
    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "fetch_intraday_prices failed symbol=%s interval=%s elapsed=%.2fs error=%s",
            symbol,
            PRICE_INTRADAY_INTERVAL,
            elapsed,
            exc,
        )
        raise RuntimeError(
            "Failed to fetch intraday price data from yfinance. Check that the "
            "symbol is valid and that the upstream API is available."
        ) from exc

    if history.empty:
        elapsed = time.time() - start
        logger.warning(
            "fetch_intraday_prices symbol=%s interval=%s returned no bars "
            "elapsed=%.2fs",
            symbol,
            PRICE_INTRADAY_INTERVAL,
            elapsed,
        )
        raise RuntimeError(
            f"No intraday bars returned for {symbol} at "
            f"intraday_interval={PRICE_INTRADAY_INTERVAL!r}. Check that this is a "
            "valid yfinance interval and that the market has traded today."
        )

    local_index = history.index.tz_convert(tz)
    bars = [
        {
            "time": idx.strftime("%H:%M:%S"),
            "price": float(close),
            "return_from_prev_close": (
                (float(close) - previous_close) / previous_close
                if previous_close
                else 0.0
            ),
        }
        for idx, close in zip(local_index, history["Close"])
    ]
    date = local_index[-1].date().isoformat()

    elapsed = time.time() - start
    logger.info(
        "fetch_intraday_prices symbol=%s interval=%s date=%s bars=%d elapsed=%.2fs",
        symbol,
        PRICE_INTRADAY_INTERVAL,
        date,
        len(bars),
        elapsed,
    )
    return {"date": date, "bars": bars}
