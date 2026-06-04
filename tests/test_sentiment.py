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


# ── real model (happy path) ───────────────────────────────────────────────────


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_returns_valid_label_and_score():
    results = sentiment.score_headlines(
        ["Earnings beat expectations"], _REAL_MODEL, _LABELS
    )
    assert len(results) == 1
    assert results[0]["label"] in _LABELS
    assert 0.0 <= results[0]["score"] <= 1.0
    assert results[0]["headline"] == "Earnings beat expectations"


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_label_is_lowercase():
    results = sentiment.score_headlines(
        ["Stock rises on strong demand"], _REAL_MODEL, _LABELS
    )
    assert results[0]["label"] == results[0]["label"].lower()


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_score_rounded_to_3dp():
    results = sentiment.score_headlines(
        ["Company holds guidance"], _REAL_MODEL, _LABELS
    )
    assert _is_rounded_3dp(results[0]["score"])


@requires_transformers
@pytest.mark.sentiment
def test_score_headlines_multiple_headlines():
    headlines = ["Good news", "Bad news", "Meh news"]
    results = sentiment.score_headlines(headlines, _REAL_MODEL, _LABELS)
    assert len(results) == len(headlines)
    assert [r["headline"] for r in results] == headlines
    for r in results:
        assert r["label"] in _LABELS
        assert 0.0 <= r["score"] <= 1.0
        assert _is_rounded_3dp(r["score"])


# ── mock-based (error paths and label-handling behaviour) ──────────────────────


def test_score_headlines_label_lowercased(monkeypatch):
    """Uppercase pipeline output is normalised to lowercase."""

    def mock_pipe(headlines, truncation, max_length):
        return [{"label": "NEGATIVE", "score": 0.88}] * len(headlines)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        ["Losses mount for retailer"], _MOCK_MODEL, _LABELS
    )
    assert results[0]["label"] == "negative"


def test_score_headlines_custom_valid_labels(monkeypatch):
    """Custom valid_labels accepts model-specific class names."""
    custom_labels = frozenset({"bullish", "bearish", "neutral"})

    def mock_pipe(headlines, truncation, max_length):
        return [{"label": "Bullish", "score": 0.80}] * len(headlines)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        ["NVDA surges on earnings"], _MOCK_MODEL, custom_labels
    )
    assert results[0]["label"] == "bullish"


def test_score_headlines_unavailable_when_pipeline_is_none(monkeypatch):
    """Returns unavailable entries when pipeline could not be loaded."""
    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: None
    )
    results = sentiment.score_headlines(["NVDA up 5%"], _MOCK_MODEL, _LABELS)
    assert len(results) == 1
    assert results[0]["label"] == "unavailable"
    assert results[0]["score"] == 0.0
    assert results[0]["headline"] == "NVDA up 5%"


def test_score_headlines_unavailable_on_inference_error(monkeypatch):
    """Returns unavailable entries if the pipeline raises during inference."""

    def failing_pipe(headlines, truncation, max_length):
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: failing_pipe
    )
    results = sentiment.score_headlines(["TSLA recall announced"], _MOCK_MODEL, _LABELS)
    assert results[0]["label"] == "unavailable"


def test_score_headlines_unexpected_label_marked_unavailable(monkeypatch):
    """Labels outside valid_labels are replaced with 'unavailable'."""

    def mock_pipe(headlines, truncation, max_length):
        return [{"label": "Bullish", "score": 0.88}] * len(headlines)

    monkeypatch.setattr(
        "financial_news.sentiment._get_pipeline", lambda model_name: mock_pipe
    )
    results = sentiment.score_headlines(
        ["NVDA surges on earnings"], _MOCK_MODEL, _LABELS
    )
    assert results[0]["label"] == "unavailable"
