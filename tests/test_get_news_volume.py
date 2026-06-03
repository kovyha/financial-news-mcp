import importlib
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

# Ensure FINNHUB_API_KEY is set so importing analysis doesn't error
os.environ.setdefault("FINNHUB_API_KEY", "test")

from financial_news import analysis, server


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
    monkeypatch.setattr(analysis.client, "company_news", lambda symbol, _from, to: [])
    assert analysis.fetch_news("FAKE", date.today(), date.today()) == []


def test_fetch_news_returns_api_response(monkeypatch):
    expected = make_articles(2)
    monkeypatch.setattr(
        analysis.client, "company_news", lambda symbol, _from, to: expected
    )
    assert analysis.fetch_news("FAKE", date.today(), date.today()) == expected


def test_fetch_news_wraps_upstream_errors(monkeypatch):
    def raise_upstream_error(symbol, _from, to):
        raise Exception("arbitrary upstream failure")

    monkeypatch.setattr(analysis.client, "company_news", raise_upstream_error)

    with pytest.raises(RuntimeError, match="Failed to fetch news from Finnhub"):
        analysis.fetch_news("FAKE", date.today(), date.today())


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
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()

    baseline = make_baseline_from_counts([1, 1, 1, 1, 1, 1, 1])
    recent = make_articles(2, day_offset=0)

    def fake_fetch(symbol, from_date, to_date):
        return recent if from_date == today else baseline

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    monkeypatch.setattr(analysis, "calculate_z_score", lambda *_: z_value)

    out = server.get_news_volume("FAKE")
    assert expected in out


def test_get_news_volume_with_no_data(monkeypatch):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    monkeypatch.setattr(analysis, "fetch_news", lambda *_, **__: [])

    out = server.get_news_volume("FAKE")

    assert "News articles (last 24hrs): 0" in out
    assert "Mean (30-day EWM): 0.0" in out
    assert "Std Dev (30-day EWM): 0.0" in out
    assert "Z-score: 0.0" in out
    assert "Normal news volume" in out
    assert (
        "No news data found for this symbol. This may mean the ticker is invalid, "
        "unsupported, or simply has no recent coverage." in out
    )


def test_get_news_volume_with_no_baseline_but_recent(monkeypatch):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()

    recent = make_articles(3)

    def fake_fetch(symbol, from_date, to_date):
        return recent if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)

    out = server.get_news_volume("FAKE")

    assert "News articles (last 24hrs): 3" in out
    assert "Mean (30-day EWM): 0.0" in out
    assert "Z-score: inf" in out
    assert "Unusual news volume detected" in out


def test_tomorrow_articles_excluded_from_recent(monkeypatch):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()
    tomorrow = today + timedelta(days=1)

    def noon(d):
        return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=et).timestamp())

    today_ts = noon(today)
    tomorrow_ts = noon(tomorrow)
    raw = [
        {"datetime": today_ts, "headline": "today article"},
        {"datetime": tomorrow_ts, "headline": "tomorrow article"},
    ]

    def fake_fetch(symbol, from_date, to_date):
        return raw if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)

    out = server.get_news_volume("FAKE")
    assert "News articles (last 24hrs): 1" in out


def test_server_import_fails_with_empty_api_key(monkeypatch):
    for key in ("financial_news.analysis", "financial_news.server"):
        if key in sys.modules:
            monkeypatch.setitem(sys.modules, key, sys.modules[key])

    sys.modules.pop("financial_news.analysis", None)
    sys.modules.pop("financial_news.server", None)
    monkeypatch.setenv("FINNHUB_API_KEY", "")

    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY is not set"):
        importlib.import_module("financial_news.analysis")

    monkeypatch.setenv("FINNHUB_API_KEY", "test")
    sys.modules.pop("financial_news.analysis", None)
    sys.modules.pop("financial_news.server", None)
    importlib.import_module("financial_news.analysis")


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
