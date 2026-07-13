"""Tests for financial_news.price."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from financial_news import price

_ET = ZoneInfo("America/New_York")


# ── calculate_price_z_score ──────────────────────────────────────────────────


def test_std_zero_returns_zero():
    """Unlike analysis.calculate_z_score, zero variance yields 0.0, not a ratio."""
    assert price.calculate_price_z_score(0.05, 0.01, 0.0) == 0.0


def test_zero_mean_is_not_a_special_case():
    """A zero baseline mean return is ordinary, not a 'no data' signal."""
    z = price.calculate_price_z_score(0.02, 0.0, 0.01)
    assert z == pytest.approx(2.0)


def test_zero_mean_and_zero_return():
    z = price.calculate_price_z_score(0.0, 0.0, 0.01)
    assert z == 0.0


@pytest.mark.parametrize(
    "return_pct,mean,std,expected",
    [
        (0.03, 0.01, 0.01, 2.0),
        (-0.03, 0.01, 0.01, -4.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_calculate_price_z_score_parametrized(return_pct, mean, std, expected):
    assert price.calculate_price_z_score(return_pct, mean, std) == pytest.approx(
        expected
    )


# ── _winsorize_two_sided ─────────────────────────────────────────────────────


def test_winsorize_two_sided_empty_list_returns_empty():
    assert price._winsorize_two_sided([], k=3.5) == []


def test_winsorize_two_sided_zero_mad_returns_unchanged():
    values = [0.0, 0.0, 0.0, 0.05]
    assert price._winsorize_two_sided(values, k=3.5) == values


def test_winsorize_two_sided_caps_high_outlier():
    values = [0.01] * 10 + [0.02] * 10 + [0.90]
    result = price._winsorize_two_sided(values, k=3.5)
    assert result[:-1] == values[:-1]
    assert result[-1] < 0.90


def test_winsorize_two_sided_caps_low_outlier():
    """Unlike analysis's one-sided winsorize, a large negative move is also capped."""
    values = [0.01] * 10 + [0.02] * 10 + [-0.90]
    result = price._winsorize_two_sided(values, k=3.5)
    assert result[:-1] == values[:-1]
    assert result[-1] > -0.90


def test_winsorize_two_sided_no_op_when_all_within_cap():
    values = [0.01, 0.02, 0.01, 0.03, 0.02]
    assert price._winsorize_two_sided(values, k=10.0) == values


# ── compute_price_stats ──────────────────────────────────────────────────────


def _make_history(closes: list[float], tz=_ET) -> pd.DataFrame:
    """Build a daily-bar DataFrame ending "yesterday" relative to *tz*'s today."""
    today = datetime.now(tz).date()
    n = len(closes)
    dates = [pd.Timestamp(today - timedelta(days=n - i), tz=tz) for i in range(n)]
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates, name="Date"))


def _mock_ticker(
    monkeypatch, closes, current_price, previous_close, tz_name="America/New_York"
):
    history = _make_history(closes, tz=ZoneInfo(tz_name))
    fast_info = MagicMock()
    fast_info.last_price = current_price
    fast_info.previous_close = previous_close
    fast_info.timezone = tz_name

    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history
    ticker_obj.fast_info = fast_info

    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)
    return ticker_obj


def test_compute_price_stats_returns_expected_keys(monkeypatch):
    _mock_ticker(
        monkeypatch, closes=[100.0] * 35, current_price=101.0, previous_close=100.0
    )
    stats = price.compute_price_stats("FAKE")
    assert set(stats.keys()) == {
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
    assert stats["symbol"] == "FAKE"


def test_compute_price_stats_flat_baseline_normal_move(monkeypatch):
    """A tiny move against a flat (zero-variance) baseline should be 'normal'."""
    _mock_ticker(
        monkeypatch, closes=[100.0] * 35, current_price=100.01, previous_close=100.0
    )
    stats = price.compute_price_stats("FAKE")
    assert stats["z_score"] == 0.0
    assert stats["classification"] == "normal"


def test_compute_price_stats_z_score_cap_applied(monkeypatch):
    closes = [100.0, 100.1] * 17 + [100.0]
    _mock_ticker(monkeypatch, closes=closes, current_price=1000.0, previous_close=100.0)
    stats = price.compute_price_stats("FAKE")
    assert stats["z_score"] == price.PRICE_Z_SCORE_CAP


def test_compute_price_stats_large_drop_caps_negative(monkeypatch):
    closes = [100.0, 100.1] * 17 + [100.0]
    _mock_ticker(monkeypatch, closes=closes, current_price=10.0, previous_close=100.0)
    stats = price.compute_price_stats("FAKE")
    assert stats["z_score"] == -price.PRICE_Z_SCORE_CAP


def test_compute_price_stats_excludes_today_from_baseline(monkeypatch):
    """Only completed trading days (before 'today') feed the baseline."""
    today_ts = pd.Timestamp(datetime.now(_ET).date(), tz=_ET)
    history = pd.DataFrame(
        {"Close": [100.0] * 30 + [500.0]},
        index=pd.DatetimeIndex(
            [today_ts - timedelta(days=30 - i) for i in range(30)] + [today_ts],
            name="Date",
        ),
    )
    fast_info = MagicMock()
    fast_info.last_price = 101.0
    fast_info.previous_close = 100.0
    fast_info.timezone = "America/New_York"
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history
    ticker_obj.fast_info = fast_info
    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)

    stats = price.compute_price_stats("FAKE")
    # If today's 500.0 bar leaked into the baseline it would blow up std/mean.
    assert stats["std"] < 1.0


def test_fetch_price_data_wraps_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(price.yf, "Ticker", raise_upstream_error)

    with pytest.raises(RuntimeError, match="Failed to fetch price data from yfinance"):
        price._fetch_price_data("FAKE")


# ── PRICE_GAUGE_SPECS contract ───────────────────────────────────────────────


def test_price_gauge_specs_keys_exist_in_compute_price_stats_output(monkeypatch):
    _mock_ticker(
        monkeypatch, closes=[100.0] * 35, current_price=101.0, previous_close=100.0
    )
    stats = price.compute_price_stats("FAKE")
    missing = {key for key, _, _ in price.PRICE_GAUGE_SPECS} - stats.keys()
    assert not missing


def test_price_gauge_specs_has_no_duplicate_keys():
    keys = [key for key, _, _ in price.PRICE_GAUGE_SPECS]
    assert len(keys) == len(set(keys))


def test_price_gauge_specs_has_no_duplicate_otel_names():
    names = [name for _, name, _ in price.PRICE_GAUGE_SPECS]
    assert len(names) == len(set(names))


# ── fetch_intraday_prices ────────────────────────────────────────────────────


def _make_intraday_history(
    times: list[str], closes: list[float], tz=_ET
) -> pd.DataFrame:
    """Build an intraday bar DataFrame for "today" in the given timezone."""
    today = datetime.now(tz).date()
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{today.isoformat()} {t}", tz=tz) for t in times],
        name="Datetime",
    )
    return pd.DataFrame({"Close": closes}, index=index)


def _mock_intraday_ticker(
    monkeypatch, times, closes, previous_close, tz_name="America/New_York"
):
    history = _make_intraday_history(times, closes, tz=ZoneInfo(tz_name))
    fast_info = MagicMock()
    fast_info.previous_close = previous_close
    fast_info.timezone = tz_name

    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history
    ticker_obj.fast_info = fast_info

    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)
    return ticker_obj


def test_fetch_intraday_prices_returns_expected_shape(monkeypatch):
    _mock_intraday_ticker(
        monkeypatch,
        times=["09:30:00", "09:35:00"],
        closes=[100.0, 102.0],
        previous_close=100.0,
    )
    result = price.fetch_intraday_prices("FAKE")
    assert set(result.keys()) == {"date", "bars"}
    assert len(result["bars"]) == 2
    assert set(result["bars"][0].keys()) == {"time", "price", "return_from_prev_close"}


def test_fetch_intraday_prices_bars_are_chronological(monkeypatch):
    _mock_intraday_ticker(
        monkeypatch,
        times=["09:30:00", "09:35:00", "09:40:00"],
        closes=[100.0, 101.0, 99.0],
        previous_close=100.0,
    )
    result = price.fetch_intraday_prices("FAKE")
    assert [bar["time"] for bar in result["bars"]] == [
        "09:30:00",
        "09:35:00",
        "09:40:00",
    ]


def test_fetch_intraday_prices_return_from_prev_close(monkeypatch):
    _mock_intraday_ticker(
        monkeypatch, times=["09:30:00"], closes=[110.0], previous_close=100.0
    )
    result = price.fetch_intraday_prices("FAKE")
    assert result["bars"][0]["return_from_prev_close"] == pytest.approx(0.10)


def test_fetch_intraday_prices_date_matches_exchange_local_today(monkeypatch):
    _mock_intraday_ticker(
        monkeypatch, times=["09:30:00"], closes=[100.0], previous_close=100.0
    )
    result = price.fetch_intraday_prices("FAKE")
    assert result["date"] == datetime.now(_ET).date().isoformat()


def test_fetch_intraday_prices_calls_history_with_configured_interval(monkeypatch):
    ticker_obj = _mock_intraday_ticker(
        monkeypatch, times=["09:30:00"], closes=[100.0], previous_close=100.0
    )
    price.fetch_intraday_prices("FAKE")
    ticker_obj.history.assert_called_once_with(
        period="1d", interval=price.PRICE_INTRADAY_INTERVAL
    )


def test_fetch_intraday_prices_wraps_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(price.yf, "Ticker", raise_upstream_error)

    with pytest.raises(
        RuntimeError, match="Failed to fetch intraday price data from yfinance"
    ):
        price.fetch_intraday_prices("FAKE")


def test_fetch_intraday_prices_empty_history_raises(monkeypatch):
    empty = pd.DataFrame({"Close": []})
    fast_info = MagicMock()
    fast_info.previous_close = 100.0
    fast_info.timezone = "America/New_York"
    ticker_obj = MagicMock()
    ticker_obj.history.return_value = empty
    ticker_obj.fast_info = fast_info
    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)

    with pytest.raises(RuntimeError, match="No intraday bars returned"):
        price.fetch_intraday_prices("FAKE")
