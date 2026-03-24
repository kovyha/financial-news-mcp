import importlib
import os
import sys
from datetime import datetime, timedelta

import pytest

# Ensure FINNHUB_API_KEY is set so importing server doesn't error
os.environ.setdefault("FINNHUB_API_KEY", "test")

from financial_news import server


def make_articles(count, day_offset=0):
    ts = int((datetime.now() - timedelta(days=day_offset)).timestamp())
    return [
        {"datetime": ts, "headline": f"headline {day_offset}-{i}"} for i in range(count)
    ]


def make_baseline_from_counts(counts):
    items = []
    for day_offset, c in enumerate(counts):
        items.extend(make_articles(c, day_offset=day_offset))
    return items


def test_fetch_news_returns_empty_list_for_empty_response(monkeypatch):
    monkeypatch.setattr(server.client, "company_news", lambda symbol, _from, to: [])
    assert server.fetch_news("FAKE", 7) == []


def test_fetch_news_returns_api_response(monkeypatch):
    expected = make_articles(2)
    monkeypatch.setattr(
        server.client, "company_news", lambda symbol, _from, to: expected
    )
    assert server.fetch_news("FAKE", 7) == expected


def test_fetch_news_wraps_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol, _from, to):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(server.client, "company_news", raise_upstream_error)

    with pytest.raises(RuntimeError, match="Failed to fetch news from Finnhub"):
        server.fetch_news("FAKE", 7)


@pytest.mark.parametrize(
    "z_value,expected",
    [
        (0.0, "Normal news volume"),
        (2.5, "Elevated news volume"),
        (3.5, "Unusual news volume detected"),
    ],
)
def test_get_news_volume_classification(monkeypatch, z_value, expected):
    # Provide simple baseline and recent articles; patch calculate_z_score to
    # return the desired z_value and assert classification string appears.
    baseline = make_baseline_from_counts([1, 1, 1, 1, 1, 1, 1])
    recent = make_articles(2, day_offset=0)

    def fake_fetch(symbol, days):
        return recent if days == 1 else baseline

    monkeypatch.setattr(server, "fetch_news", fake_fetch)
    monkeypatch.setattr(server, "calculate_z_score", lambda rc, m, s: z_value)

    out = server.get_news_volume("FAKE")
    assert expected in out


def test_get_news_volume_with_no_data(monkeypatch):
    monkeypatch.setattr(server, "fetch_news", lambda symbol, days: [])

    out = server.get_news_volume("FAKE")

    assert "News articles (last 24hrs): 0" in out
    assert "Mean (7-day): 0.0" in out
    assert "Standard Deviation (7-day, delta degree of freedom=1): 0.0" in out
    assert "Z-score: 0.0" in out
    assert "Normal news volume" in out
    assert (
        "No news data found for this symbol. This may mean the ticker is invalid, "
        "unsupported, or simply has no recent coverage." in out
    )


def test_get_news_volume_with_no_baseline_but_recent(monkeypatch):
    recent = make_articles(3)

    def fake_fetch(symbol, days):
        return recent if days == 1 else []

    monkeypatch.setattr(server, "fetch_news", fake_fetch)

    out = server.get_news_volume("FAKE")

    assert "News articles (last 24hrs): 3" in out
    assert "Mean (7-day): 0.0" in out
    assert "Z-score: inf" in out
    assert "Unusual news volume detected" in out


def test_server_import_fails_with_empty_api_key(monkeypatch):
    sys.modules.pop("financial_news.server", None)
    monkeypatch.setenv("FINNHUB_API_KEY", "")

    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY is not set"):
        importlib.import_module("financial_news.server")

    monkeypatch.setenv("FINNHUB_API_KEY", "test")
    sys.modules.pop("financial_news.server", None)
    importlib.import_module("financial_news.server")


@pytest.mark.parametrize(
    "api_response,expected_status",
    [
        ([], "✅ Finnhub API healthy"),
        ([{"headline": "test"}], "✅ Finnhub API healthy"),
    ],
)
def test_health_check_success(monkeypatch, api_response, expected_status):
    monkeypatch.setattr(
        server.client, "company_news", lambda symbol, **kw: api_response
    )
    out = server.health_check()
    assert expected_status in out
    assert "Response time:" in out


def test_health_check_failure(monkeypatch):
    def raise_error(symbol, **kw):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(server.client, "company_news", raise_error)
    out = server.health_check()
    assert "❌ Finnhub API unreachable" in out
    assert "API unavailable" in out
