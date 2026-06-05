"""Tests for financial_news.enrichment."""

from financial_news import enrichment, sentiment
from financial_news.enrichment import EnrichmentConfig

_CFG = EnrichmentConfig(
    model_name="test-model",
    valid_labels=frozenset({"positive", "negative", "neutral"}),
    confidence_threshold=0.85,
    min_articles=5,
    max_articles=50,
)


def _base_stats(classification: str = "unusual") -> dict:
    return {
        "ticker": "NVDA",
        "recent_count": 3,
        "mean": 2.0,
        "std": 0.5,
        "z_score": 2.5,
        "classification": classification,
        "headlines": ["Headline A", "Headline B"],
        "recent_articles": [
            {"headline": "Headline A", "summary": "Summary A."},
            {"headline": "Headline B", "summary": ""},
            {"headline": "Headline C", "summary": "Summary C."},
        ],
        "recent_headlines": ["Headline A", "Headline B", "Headline C"],
        "baseline_counts": [2.0] * 30,
    }


def _mock_score(monkeypatch, label: str = "positive", score: float = 0.90):
    monkeypatch.setattr(
        sentiment,
        "score_headlines",
        lambda articles, *_: [
            {
                "headline": a["headline"],
                "summary": a.get("summary", ""),
                "label": label,
                "score": score,
            }
            for a in articles
        ],
    )


# ── enrich_stats ─────────────────────────────────────────────────────────────


def test_enrich_stats_adds_headline_sentiment(monkeypatch):
    _mock_score(monkeypatch)
    result = enrichment.enrich_stats(_base_stats(), _CFG)
    assert "headline_sentiment" in result
    assert len(result["headline_sentiment"]) == 3  # all recent_articles scored


def test_enrich_stats_adds_selected_articles(monkeypatch):
    _mock_score(monkeypatch, label="positive", score=0.92)
    result = enrichment.enrich_stats(_base_stats(), _CFG)
    assert "selected_articles" in result
    assert len(result["selected_articles"]) > 0


def test_enrich_stats_prefers_recent_articles(monkeypatch):
    scored_headlines = []

    def capture(articles, *_):
        scored_headlines.extend(a["headline"] for a in articles)
        return [
            {
                "headline": a["headline"],
                "summary": "",
                "label": "positive",
                "score": 0.9,
            }
            for a in articles
        ]

    monkeypatch.setattr(sentiment, "score_headlines", capture)
    enrichment.enrich_stats(_base_stats(), _CFG)
    assert "Headline C" in scored_headlines


def test_enrich_stats_falls_back_to_headline_strings(monkeypatch):
    """Falls back to recent_headlines strings when recent_articles is absent."""
    scored_headlines = []

    def capture(articles, *_):
        scored_headlines.extend(a["headline"] for a in articles)
        return [
            {"headline": a["headline"], "summary": "", "label": "neutral", "score": 0.5}
            for a in articles
        ]

    monkeypatch.setattr(sentiment, "score_headlines", capture)
    s = _base_stats()
    del s["recent_articles"]
    enrichment.enrich_stats(s, _CFG)
    assert "Headline A" in scored_headlines


def test_enrich_stats_zero_news_skips_scoring(monkeypatch):
    called = []
    monkeypatch.setattr(
        sentiment, "score_headlines", lambda *_: called.append(True) or []
    )
    s = {
        **_base_stats(),
        "recent_articles": [],
        "recent_headlines": [],
        "headlines": [],
    }
    result = enrichment.enrich_stats(s, _CFG)
    assert result["headline_sentiment"] == []
    assert result["selected_articles"] == []
    assert not called


def test_enrich_stats_passes_through_error_entry(monkeypatch):
    _mock_score(monkeypatch)
    error_entry = {"ticker": "FAIL", "error": "fetch failed"}
    result = enrichment.enrich_stats(error_entry, _CFG)
    assert result == error_entry
    assert "headline_sentiment" not in result
    assert "selected_articles" not in result


def test_enrich_stats_passes_classification_to_select(monkeypatch):
    """Neutral articles are filtered out for elevated/unusual classification."""
    monkeypatch.setattr(
        sentiment,
        "score_headlines",
        lambda articles, *_: [
            {
                "headline": "Neutral noise",
                "summary": "",
                "label": "neutral",
                "score": 0.95,
            },
            {
                "headline": "Negative signal",
                "summary": "",
                "label": "negative",
                "score": 0.92,
            },
        ],
    )
    result = enrichment.enrich_stats(_base_stats("unusual"), _CFG)
    selected_headlines = [a["headline"] for a in result["selected_articles"]]
    assert "Negative signal" in selected_headlines
    assert "Neutral noise" not in selected_headlines


def test_enrich_stats_preserves_original_fields(monkeypatch):
    _mock_score(monkeypatch)
    s = _base_stats()
    result = enrichment.enrich_stats(s, _CFG)
    assert result["z_score"] == s["z_score"]
    assert result["classification"] == s["classification"]
    assert result["recent_count"] == s["recent_count"]


# ── enrich_ticker ─────────────────────────────────────────────────────────────


def test_enrich_ticker_calls_compute_volume_stats(monkeypatch):
    called = []

    def fake_compute(symbol):
        called.append(symbol)
        return _base_stats()

    monkeypatch.setattr(enrichment, "compute_volume_stats", fake_compute)
    _mock_score(monkeypatch)
    enrichment.enrich_ticker("NVDA", _CFG)
    assert called == ["NVDA"]


def test_enrich_ticker_sets_ticker_field(monkeypatch):
    def fake_compute(symbol):
        s = _base_stats()
        del s["ticker"]
        return s

    monkeypatch.setattr(enrichment, "compute_volume_stats", fake_compute)
    _mock_score(monkeypatch)
    result = enrichment.enrich_ticker("TSLA", _CFG)
    assert result["ticker"] == "TSLA"


def test_enrich_ticker_returns_enriched_stats(monkeypatch):
    monkeypatch.setattr(enrichment, "compute_volume_stats", lambda s: _base_stats())
    _mock_score(monkeypatch)
    result = enrichment.enrich_ticker("NVDA", _CFG)
    assert "headline_sentiment" in result
    assert "selected_articles" in result


# ── select_articles ───────────────────────────────────────────────────────────


def _scored_items(headlines_and_scores: list[tuple[str, float]]) -> list[dict]:
    return [
        {"headline": h, "summary": "", "label": "positive", "score": s}
        for h, s in headlines_and_scores
    ]


def test_select_articles_confidence_filter():
    items = _scored_items([("H-high", 0.90), ("H-mid", 0.80), ("H-low", 0.60)])
    result = enrichment.select_articles(items, "normal", 0.85, 2, 10)
    headlines = [r["headline"] for r in result]
    assert "H-high" in headlines
    assert "H-low" not in headlines


def test_select_articles_min_cap():
    """Falls back to top-min when fewer than min meet the threshold."""
    items = _scored_items([("H1", 0.60), ("H2", 0.55), ("H3", 0.50), ("H4", 0.45)])
    result = enrichment.select_articles(items, "normal", 0.85, 3, 10)
    assert len(result) == 3
    assert result[0]["headline"] == "H1"


def test_select_articles_max_cap():
    items = _scored_items([(f"H{i}", 0.90) for i in range(20)])
    result = enrichment.select_articles(items, "normal", 0.85, 5, 10)
    assert len(result) == 10


def test_select_articles_sorted_by_score_desc():
    items = _scored_items([("Lo", 0.88), ("Hi", 0.95), ("Mid", 0.91)])
    result = enrichment.select_articles(items, "normal", 0.85, 1, 10)
    assert [r["headline"] for r in result] == ["Hi", "Mid", "Lo"]


def test_select_articles_discards_high_confidence_neutral_for_elevated():
    """High-confidence neutral articles are dropped for elevated/unusual tickers."""
    items = [
        {"headline": "Meh news", "summary": "", "label": "neutral", "score": 0.95},
        {"headline": "Bad news", "summary": "", "label": "negative", "score": 0.88},
    ]
    result = enrichment.select_articles(items, "elevated", 0.85, 1, 10)
    headlines = [r["headline"] for r in result]
    assert "Bad news" in headlines
    assert "Meh news" not in headlines


def test_select_articles_keeps_low_confidence_neutral_as_borderline():
    """Weakly-neutral articles (score < threshold) are kept even for elevated."""
    items = [
        {
            "headline": "Borderline news",
            "summary": "",
            "label": "neutral",
            "score": 0.70,
        },
    ]
    result = enrichment.select_articles(items, "elevated", 0.85, 1, 10)
    assert result[0]["headline"] == "Borderline news"


def test_select_articles_normal_ticker_keeps_neutral():
    """Neutral filtering does not apply for normal classification."""
    items = [
        {"headline": "Quiet day", "summary": "", "label": "neutral", "score": 0.95},
    ]
    result = enrichment.select_articles(items, "normal", 0.85, 1, 10)
    assert result[0]["headline"] == "Quiet day"
