"""Tests for financial_news.sentiment."""

import pytest

from financial_news import sentiment

try:
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers not installed; run `uv sync --group sentiment`",
)

_REAL_MODEL = "ProsusAI/finbert"
_MOCK_MODEL = "test-model"
_LABELS = frozenset({"positive", "negative", "neutral"})

# ── helpers ───────────────────────────────────────────────────────────────────


def _is_rounded_3dp(value: float) -> bool:
    return value == round(value, 3)


# ── empty input ───────────────────────────────────────────────────────────────


def test_score_headlines_empty():
    assert sentiment.score_headlines([], _MOCK_MODEL, _LABELS) == []


def test_score_headlines_none_summary_treated_as_empty(monkeypatch):
    """Articles with summary=None are scored on headline only, not 'headline. None'."""

    captured = []

    def mock_pipe(texts, **_):
        captured.extend(texts)
        return [{"label": "positive", "score": 0.9}] * len(texts)

    monkeypatch.setattr("financial_news.sentiment._get_pipeline", lambda _: mock_pipe)
    sentiment.score_headlines(
        [{"headline": "Big earnings beat", "summary": None}], _MOCK_MODEL, _LABELS
    )
    assert captured == ["Big earnings beat"]


def test_score_headlines_missing_summary_key_treated_as_empty(monkeypatch):
    """Articles with no summary key at all are scored on headline only."""

    captured = []

    def mock_pipe(texts, **_):
        captured.extend(texts)
        return [{"label": "positive", "score": 0.9}] * len(texts)

    monkeypatch.setattr("financial_news.sentiment._get_pipeline", lambda _: mock_pipe)
    sentiment.score_headlines([{"headline": "Big earnings beat"}], _MOCK_MODEL, _LABELS)
    assert captured == ["Big earnings beat"]


def test_compose_text_with_summary():
    result = sentiment._compose_text("Headline", "Summary text")
    assert result == "Headline. Summary text"


def test_compose_text_empty_summary():
    assert sentiment._compose_text("Headline", "") == "Headline"


def test_compose_text_very_long_summary():
    long_summary = "word " * 1000
    result = sentiment._compose_text("Headline", long_summary)
    assert result.startswith("Headline. ")
    assert long_summary in result


# ── real model (happy path) ───────────────────────────────────────────────────


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_returns_valid_label_and_score():
    results = sentiment.score_headlines(
        [{"headline": "Earnings beat expectations", "summary": ""}],
        _REAL_MODEL,
        _LABELS,
    )
    assert len(results) == 1
    assert results[0]["label"] in _LABELS
    assert 0.0 <= results[0]["score"] <= 1.0
    assert results[0]["headline"] == "Earnings beat expectations"
    assert results[0]["summary"] == ""


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_with_summary_scores_combined_text():
    results = sentiment.score_headlines(
        [{"headline": "Earnings beat expectations", "summary": "Revenue up 20% YoY."}],
        _REAL_MODEL,
        _LABELS,
    )
    assert len(results) == 1
    assert results[0]["headline"] == "Earnings beat expectations"
    assert results[0]["summary"] == "Revenue up 20% YoY."
    assert results[0]["label"] in _LABELS


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_label_is_lowercase():
    results = sentiment.score_headlines(
        [{"headline": "Stock rises on strong demand", "summary": ""}],
        _REAL_MODEL,
        _LABELS,
    )
    assert results[0]["label"] == results[0]["label"].lower()


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_score_rounded_to_3dp():
    results = sentiment.score_headlines(
        [{"headline": "Company holds guidance", "summary": ""}], _REAL_MODEL, _LABELS
    )
    assert _is_rounded_3dp(results[0]["score"])


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_multiple_headlines():
    articles = [
        {"headline": "Good news", "summary": ""},
        {"headline": "Bad news", "summary": "Details follow."},
        {"headline": "Meh news", "summary": ""},
    ]
    results = sentiment.score_headlines(articles, _REAL_MODEL, _LABELS)
    assert len(results) == len(articles)
    assert [r["headline"] for r in results] == [a["headline"] for a in articles]
    for r in results:
        assert r["label"] in _LABELS
        assert 0.0 <= r["score"] <= 1.0
        assert _is_rounded_3dp(r["score"])


# ── mock-based (error paths and label-handling behaviour) ──────────────────────


def test_score_headlines_label_lowercased(monkeypatch):
    """Uppercase pipeline output is normalised to lowercase."""

    def mock_pipe(texts, **_):
        return [{"label": "NEGATIVE", "score": 0.88}] * len(texts)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        [{"headline": "Losses mount for retailer", "summary": ""}], _MOCK_MODEL, _LABELS
    )
    assert results[0]["label"] == "negative"


def test_score_headlines_custom_valid_labels(monkeypatch):
    """Custom valid_labels accepts model-specific class names."""
    custom_labels = frozenset({"bullish", "bearish", "neutral"})

    def mock_pipe(texts, **_):
        return [{"label": "Bullish", "score": 0.80}] * len(texts)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        [{"headline": "NVDA surges on earnings", "summary": ""}],
        _MOCK_MODEL,
        custom_labels,
    )
    assert results[0]["label"] == "bullish"


def test_score_headlines_unavailable_when_pipeline_is_none(monkeypatch):
    """Returns unavailable entries when pipeline could not be loaded."""
    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: None
    )
    results = sentiment.score_headlines(
        [{"headline": "NVDA up 5%", "summary": ""}], _MOCK_MODEL, _LABELS
    )
    assert len(results) == 1
    assert results[0]["label"] == "unavailable"
    assert results[0]["score"] == 0.0
    assert results[0]["headline"] == "NVDA up 5%"


def test_score_headlines_unavailable_on_inference_error(monkeypatch):
    """Returns unavailable entries if the pipeline raises during inference."""

    def failing_pipe(*_, **__):
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: failing_pipe
    )
    results = sentiment.score_headlines(
        [{"headline": "TSLA recall announced", "summary": ""}], _MOCK_MODEL, _LABELS
    )
    assert results[0]["label"] == "unavailable"


def test_score_headlines_unexpected_label_marked_unavailable(monkeypatch):
    """Labels outside valid_labels are replaced with 'unavailable'."""

    def mock_pipe(texts, **_):
        return [{"label": "Bullish", "score": 0.88}] * len(texts)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        [{"headline": "NVDA surges on earnings", "summary": ""}], _MOCK_MODEL, _LABELS
    )
    assert results[0]["label"] == "unavailable"


def test_score_headlines_summary_preserved_in_output(monkeypatch):
    """Summary from the input article is echoed back in the output dict."""

    def mock_pipe(texts, **_):
        return [{"label": "positive", "score": 0.91}] * len(texts)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        [
            {
                "headline": "NVDA beats estimates",
                "summary": "EPS of $5.16 vs $4.59 expected.",
            }
        ],  # noqa: E501
        _MOCK_MODEL,
        _LABELS,
    )
    assert results[0]["headline"] == "NVDA beats estimates"
    assert results[0]["summary"] == "EPS of $5.16 vs $4.59 expected."


def test_score_headlines_very_long_summary_returns_valid_result(monkeypatch):
    """A very long headline+summary is passed to the pipe with truncation=True and
    returns a valid scored entry — the original headline is preserved unchanged."""
    long_summary = "This is a long sentence about financial results. " * 50

    received_kwargs = {}

    def mock_pipe(texts, **kwargs):
        received_kwargs.update(kwargs)
        return [{"label": "positive", "score": 0.88}] * len(texts)

    monkeypatch.setattr("financial_news.sentiment._get_pipeline", lambda _: mock_pipe)
    results = sentiment.score_headlines(
        [{"headline": "NVDA earnings", "summary": long_summary}],
        _MOCK_MODEL,
        _LABELS,
    )
    assert len(results) == 1
    assert results[0]["headline"] == "NVDA earnings"
    assert results[0]["label"] == "positive"
    assert received_kwargs.get("truncation") is True
    assert received_kwargs.get("max_length") == 512


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_very_long_summary_with_real_model():
    """A headline+summary exceeding 512 tokens is silently truncated by finBERT
    and still returns a valid label and score without raising."""
    long_summary = "Strong quarterly results driven by data center demand. " * 40
    results = sentiment.score_headlines(
        [{"headline": "NVDA beats estimates", "summary": long_summary}],
        _REAL_MODEL,
        _LABELS,
    )
    assert len(results) == 1
    assert results[0]["label"] in _LABELS
    assert 0.0 <= results[0]["score"] <= 1.0
    assert results[0]["headline"] == "NVDA beats estimates"
