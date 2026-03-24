import gzip
import logging
import logging.handlers
import os
import shutil
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import finnhub
import numpy as np
from mcp.server.fastmcp import FastMCP

from financial_news.config import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_config = load_config()
LOG_DIR = Path(_config.logging.log_dir)
_LOG_MAX_BYTES = _config.logging.max_bytes
_ACTIVE_LOG_FILENAME = _config.logging.filename
_LOG_LEVEL = logging.DEBUG if _config.logging.level == "DEBUG" else logging.INFO


class _TimestampRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """Rolling file handler that names rolled files with their creation
    timestamp and gzip-compresses them immediately on rollover.

    Each rolled filename takes the form:
        <stem>_<YYYYMMDD_HHMMSS><ext>.gz
    where the timestamp reflects when the *current* file was opened,
    not when it was rolled off.
    """

    def __init__(
        self,
        filename: Path,
        max_bytes: int = _LOG_MAX_BYTES,
        created_at: datetime | None = None,
    ) -> None:
        super().__init__(str(filename), mode="a", encoding="utf-8")
        self.maxBytes = max_bytes
        self._created_at: datetime = (
            created_at if created_at is not None else datetime.now()
        )

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.stream is None:
            self.stream = self._open()
        msg = self.format(record) + "\n"
        self.stream.seek(0, 2)
        current_size = self.stream.tell()
        message_size = len(msg.encode("utf-8"))
        return current_size + message_size >= self.maxBytes

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record, ensuring writes are flushed to disk immediately."""
        try:
            super().emit(record)
            if self.stream:
                self.stream.flush()
        except Exception:
            self.handleError(record)

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        path = Path(self.baseFilename)
        dest = path.with_suffix(path.suffix + ".gz")

        with open(path, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        path.unlink()  # Delete the old file after gzipping

        # Start new active file with fresh timestamp
        new_ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        stem_parts = path.stem.rsplit("_", 2)
        base_stem = "_".join(stem_parts[:-2]) if len(stem_parts) >= 3 else path.stem
        new_path = path.parent / f"{base_stem}_{new_ts}{path.suffix}"

        self.baseFilename = str(new_path)
        self._created_at = datetime.now()
        self.stream = self._open()


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("financial_news")
    if log.handlers:
        # Already configured — guard against duplicate handlers on module re-import.
        return log

    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    # Generate startup timestamp to embed in log filenames
    startup_ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    startup_dt = datetime.now()

    app_handler = _TimestampRotatingFileHandler(
        LOG_DIR / f"{_ACTIVE_LOG_FILENAME}_{startup_ts}.log",
        created_at=startup_dt,
    )
    app_handler.setLevel(_LOG_LEVEL)
    app_handler.setFormatter(fmt)

    error_handler = _TimestampRotatingFileHandler(
        LOG_DIR / f"{_ACTIVE_LOG_FILENAME}_{startup_ts}.error.log",
        created_at=startup_dt,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    log.setLevel(logging.DEBUG)  # logger always at DEBUG; handlers filter by level
    log.addHandler(app_handler)
    log.addHandler(error_handler)
    return log


logger = _setup_logger()


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


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("financial-news")  # Creates the MCP Server

api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise RuntimeError(
        "FINNHUB_API_KEY is not set. Set it in the environment before running "
        "the server. Example: export FINNHUB_API_KEY='your_key'."
    )

client = finnhub.Client(api_key=api_key)

# Validate thresholds on startup
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
        # Simple test call to Finnhub API
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

    recent = fetch_news(symbol, days=1)
    baseline = fetch_news(symbol, days=7)

    baseline_counts = list(
        Counter(
            datetime.fromtimestamp(article["datetime"]).date() for article in baseline
        ).values()
    )
    if baseline_counts:
        mean = np.mean(baseline_counts)
        std = np.std(baseline_counts, ddof=1)
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
        "get_news_volume symbol=%s recent_count=%d mean=%.2f std=%.2f "
        "z_score=%.2f classification=%s",
        symbol,
        recent_count,
        float(mean),
        float(std),
        z_score,
        classification,
    )

    headlines = [article["headline"] for article in recent[:5]]
    logger.debug(
        "get_news_volume symbol=%s headlines_passed_to_model=%s", symbol, headlines
    )

    summary_lines = [
        f"Symbol: {symbol}",
        f"News articles (last 24hrs): {recent_count}",
        f"Mean (7-day): {mean:.1f}",
        f"Standard Deviation (7-day, delta degree of freedom=1): {std:.1f}",
        f"Z-score: {z_score:.1f}",
    ]

    if classification == "normal":
        summary_lines.append("✅ Normal news volume")
    elif classification == "elevated":
        summary_lines.append("⚠️ Elevated news volume")
    else:
        summary_lines.append("🚨 Unusual news volume detected")

    if recent_count == 0 and not baseline_counts:
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
