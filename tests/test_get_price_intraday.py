"""Tests for the get_price_intraday MCP tool."""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

os.environ.setdefault("FINNHUB_API_KEY", "test")

from financial_news import price, server


@pytest.fixture()
def mock_intraday_ticker(monkeypatch):
    import datetime as dt

    et = dt.timezone(dt.timedelta(hours=-4))
    today = dt.datetime.now(et).date()
    index = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{today.isoformat()} 09:30:00", tz="America/New_York"),
            pd.Timestamp(f"{today.isoformat()} 09:35:00", tz="America/New_York"),
        ]
    )
    history = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    fast_info = MagicMock()
    fast_info.previous_close = 100.0
    fast_info.timezone = "America/New_York"

    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history
    ticker_obj.fast_info = fast_info

    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)
    return ticker_obj


def test_get_price_intraday_returns_symbol(mock_intraday_ticker):
    out = server.get_price_intraday("AAPL")
    assert out["symbol"] == "AAPL"


def test_get_price_intraday_returns_expected_keys(mock_intraday_ticker):
    out = server.get_price_intraday("AAPL")
    assert set(out.keys()) == {"symbol", "date", "bars"}
    assert len(out["bars"]) == 2
    assert set(out["bars"][0].keys()) == {"time", "price", "return_from_prev_close"}


def test_get_price_intraday_propagates_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(price.yf, "Ticker", raise_upstream_error)

    with pytest.raises(
        RuntimeError, match="Failed to fetch intraday price data from yfinance"
    ):
        server.get_price_intraday("AAPL")
