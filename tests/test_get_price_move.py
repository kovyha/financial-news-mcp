"""Tests for the get_price_move MCP tool."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("FINNHUB_API_KEY", "test")

from financial_news import price, server


@pytest.fixture()
def mock_ticker(monkeypatch):
    import pandas as pd

    history = pd.DataFrame(
        {"Close": [100.0] * 35},
        index=pd.date_range("2026-01-01", periods=35, freq="D", tz="America/New_York"),
    )
    fast_info = MagicMock()
    fast_info.last_price = 101.0
    fast_info.previous_close = 100.0
    fast_info.timezone = "America/New_York"

    ticker_obj = MagicMock()
    ticker_obj.history.return_value = history
    ticker_obj.fast_info = fast_info

    monkeypatch.setattr(price.yf, "Ticker", lambda symbol: ticker_obj)
    return ticker_obj


def test_get_price_move_returns_symbol(mock_ticker):
    out = server.get_price_move("AAPL")
    assert out["symbol"] == "AAPL"


def test_get_price_move_returns_expected_keys(mock_ticker):
    out = server.get_price_move("AAPL")
    assert set(out.keys()) == {
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


def test_get_price_move_propagates_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(price.yf, "Ticker", raise_upstream_error)

    with pytest.raises(RuntimeError, match="Failed to fetch price data from yfinance"):
        server.get_price_move("AAPL")
