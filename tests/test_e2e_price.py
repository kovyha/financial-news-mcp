"""End-to-end tests for the price-move signal against the live yfinance API.

These tests make real network calls and are skipped by default (yfinance needs
no API key, so there's no "key absent" signal to gate on the way the Finnhub
e2e tests do — an explicit opt-in env var is used instead).

Run explicitly:
    RUN_PRICE_E2E=1 uv run pytest -m price -v

They are excluded from the regular validation suite (uv run pytest) unless
RUN_PRICE_E2E=1 is set.
"""

import math
import os

import pytest

from financial_news.price import (
    PRICE_THRESHOLD_ELEVATED,
    PRICE_THRESHOLD_UNUSUAL,
    compute_price_stats,
)

_run_e2e = pytest.mark.skipif(
    os.environ.get("RUN_PRICE_E2E", "") != "1",
    reason="set RUN_PRICE_E2E=1 to run live yfinance e2e tests",
)

pytestmark = [pytest.mark.price, _run_e2e]

# Stable, high-volume ticker — reliably has baseline coverage.
_TICKER = "AAPL"


@pytest.fixture(scope="module")
def stats():
    """Run compute_price_stats once and share the result across all tests."""
    return compute_price_stats(_TICKER)


def test_stats_has_all_required_keys(stats):
    required = {
        "symbol",
        "current_price",
        "previous_close",
        "return_pct",
        "mean",
        "std",
        "z_score",
        "classification",
        "baseline_days",
    }
    missing = required - stats.keys()
    assert not missing, f"missing keys in compute_price_stats output: {missing}"


def test_stats_value_types(stats):
    assert isinstance(stats["current_price"], float)
    assert isinstance(stats["previous_close"], float)
    assert isinstance(stats["return_pct"], float)
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
    assert isinstance(stats["z_score"], float)
    assert isinstance(stats["classification"], str)


def test_current_price_is_positive(stats):
    assert stats["current_price"] > 0


def test_previous_close_is_positive(stats):
    assert stats["previous_close"] > 0


def test_std_is_non_negative(stats):
    assert stats["std"] >= 0.0


def test_z_score_is_not_nan(stats):
    assert not math.isnan(stats["z_score"]), "z_score must never be NaN"


def test_classification_is_valid(stats):
    assert stats["classification"] in {"normal", "elevated", "unusual"}


def test_classification_is_consistent_with_z_score(stats):
    """Classification must follow the same thresholds used by compute_price_stats."""
    z = abs(stats["z_score"])
    cls = stats["classification"]
    if z < PRICE_THRESHOLD_ELEVATED:
        assert cls == "normal", (
            f"|z|={z:.2f} < PRICE_THRESHOLD_ELEVATED={PRICE_THRESHOLD_ELEVATED} "
            f"but got {cls!r}"
        )
    elif z < PRICE_THRESHOLD_UNUSUAL:
        assert cls == "elevated", (
            f"PRICE_THRESHOLD_ELEVATED={PRICE_THRESHOLD_ELEVATED} <= |z|={z:.2f} < "
            f"PRICE_THRESHOLD_UNUSUAL={PRICE_THRESHOLD_UNUSUAL} but got {cls!r}"
        )
    else:
        assert cls == "unusual", (
            f"|z|={z:.2f} >= PRICE_THRESHOLD_UNUSUAL={PRICE_THRESHOLD_UNUSUAL} "
            f"but got {cls!r}"
        )
