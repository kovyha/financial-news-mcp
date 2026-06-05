import importlib
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

# Ensure FINNHUB_API_KEY is set so importing analysis doesn't error
os.environ.setdefault("FINNHUB_API_KEY", "test")

from financial_news import analysis, sentiment, server

_ET = ZoneInfo("America/New_York")


def make_articles(count, day_offset=0, summary=""):
    ts = int((datetime.now() - timedelta(days=day_offset)).timestamp())
    return [
        {"datetime": ts, "headline": f"headline {day_offset}-{i}", "summary": summary}
        for i in range(count)
    ]


def make_baseline_from_counts(counts):
    items = []
    for day_offset, c in enumerate(counts):
        items.extend(make_articles(c, day_offset=day_offset))
    return items


def _mock_sentiment_unavailable(monkeypatch):
    """Patch score_headlines to return unavailable labels without loading finBERT."""
    monkeypatch.setattr(
        sentiment,
        "score_headlines",
        lambda articles, *_: [
            {
                "headline": a["headline"],
                "summary": a.get("summary", ""),
                "label": "unavailable",
                "score": 0.0,
            }  # noqa: E501
            for a in articles
        ],
    )


# ── fetch_news ────────────────────────────────────────────────────────────────


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


# ── compute_volume_stats ─────────────────────────────────────────────────────


def test_compute_volume_stats_caps_z_score(monkeypatch):
    monkeypatch.setattr(analysis, "_exchange_tz", lambda s: _ET)
    monkeypatch.setattr(analysis, "fetch_news", lambda *_, **__: [])
    monkeypatch.setattr(analysis, "calculate_z_score", lambda *_: float("inf"))

    stats = analysis.compute_volume_stats("FAKE")
    assert stats["z_score"] == analysis.Z_SCORE_CAP


# ── get_news_volume ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "z_value,expected_classification",
    [
        (0.0, "normal"),
        (2.5, "elevated"),
        (3.5, "unusual"),
    ],
)
def test_get_news_volume_classification(monkeypatch, z_value, expected_classification):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()

    baseline = make_baseline_from_counts([1, 1, 1, 1, 1, 1, 1])
    recent = make_articles(2, day_offset=0)

    def fake_fetch(symbol, from_date, to_date):
        return recent if from_date == today else baseline

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    monkeypatch.setattr(analysis, "calculate_z_score", lambda *_: z_value)
    _mock_sentiment_unavailable(monkeypatch)

    out = server.get_news_volume("FAKE")
    assert out["classification"] == expected_classification


def test_get_news_volume_returns_symbol(monkeypatch):
    monkeypatch.setattr(analysis, "_exchange_tz", lambda s: _ET)
    monkeypatch.setattr(analysis, "fetch_news", lambda *_, **__: [])

    out = server.get_news_volume("NVDA")
    assert out["symbol"] == "NVDA"


def test_get_news_volume_with_no_data(monkeypatch):
    monkeypatch.setattr(analysis, "_exchange_tz", lambda s: _ET)
    monkeypatch.setattr(analysis, "fetch_news", lambda *_, **__: [])

    out = server.get_news_volume("FAKE")

    assert out["recent_count"] == 0
    assert out["ewm_mean"] == 0.0
    assert out["ewm_std"] == 0.0
    assert out["z_score"] == 0.0
    assert out["classification"] == "normal"
    assert out["articles"] == []


def test_get_news_volume_with_no_baseline_but_recent(monkeypatch):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()

    def fake_fetch(symbol, from_date, to_date):
        return make_articles(3) if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    _mock_sentiment_unavailable(monkeypatch)

    out = server.get_news_volume("FAKE")

    assert out["recent_count"] == 3
    assert out["ewm_mean"] == 0.0
    assert out["z_score"] == analysis.Z_SCORE_CAP
    assert out["classification"] == "unusual"


def test_tomorrow_articles_excluded_from_recent(monkeypatch):
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()
    tomorrow = today + timedelta(days=1)

    def noon(d):
        return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=et).timestamp())

    raw = [
        {"datetime": noon(today), "headline": "today article"},
        {"datetime": noon(tomorrow), "headline": "tomorrow article"},
    ]

    def fake_fetch(symbol, from_date, to_date):
        return raw if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    _mock_sentiment_unavailable(monkeypatch)

    out = server.get_news_volume("FAKE")
    assert out["recent_count"] == 1


def test_sentiment_enrichment_in_get_news_volume(monkeypatch):
    """finBERT scores and summaries reach the structured response articles."""
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()
    recent = make_articles(1, summary="Strong quarterly results.")

    def fake_fetch(symbol, from_date, to_date):
        return recent if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    monkeypatch.setattr(
        sentiment,
        "score_headlines",
        lambda articles, *_: [
            {
                "headline": a["headline"],
                "summary": a.get("summary", ""),
                "label": "positive",
                "score": 0.92,
            }  # noqa: E501
            for a in articles
        ],
    )

    out = server.get_news_volume("FAKE")
    assert len(out["articles"]) == 1
    assert out["articles"][0]["label"] == "positive"
    assert out["articles"][0]["score"] == 0.92
    assert out["articles"][0]["summary"] == "Strong quarterly results."


def test_get_news_volume_articles_structure(monkeypatch):
    """Each article in the response has the expected keys."""
    et = ZoneInfo("America/New_York")
    monkeypatch.setattr(analysis, "_exchange_tz", lambda symbol: et)
    today = datetime.now(et).date()

    def fake_fetch(symbol, from_date, to_date):
        return make_articles(2, summary="A summary.") if from_date == today else []

    monkeypatch.setattr(analysis, "fetch_news", fake_fetch)
    _mock_sentiment_unavailable(monkeypatch)

    out = server.get_news_volume("FAKE")
    for article in out["articles"]:
        assert "headline" in article
        assert "summary" in article
        assert "label" in article
        assert "score" in article


# ── server import guard ───────────────────────────────────────────────────────


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


# ── health_check ──────────────────────────────────────────────────────────────


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
