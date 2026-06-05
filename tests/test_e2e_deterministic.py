"""End-to-end tests for the deterministic layer against the live Finnhub API.

These tests require a real FINNHUB_API_KEY — they are skipped automatically
when the key is absent or set to the test placeholder ("test").

The finBERT section additionally requires the `sentiment` dependency group:
    uv sync --group sentiment

Run explicitly:
    FINNHUB_API_KEY=<key> uv run pytest -m e2e -v               # all e2e
    FINNHUB_API_KEY=<key> uv run pytest -m e2e -k sentiment -v  # sentiment only

They are excluded from the regular validation suite (uv run pytest) because
conftest.py sets FINNHUB_API_KEY="test" as the default.
"""

import math
import os

import pytest

from financial_news.analysis import (
    BASELINE_DAYS,
    THRESHOLD_ELEVATED,
    THRESHOLD_UNUSUAL,
    compute_volume_stats,
    fetch_news,
)

# ---------------------------------------------------------------------------
# Skip guard — active whenever the key is absent or the test placeholder
# ---------------------------------------------------------------------------

_real_key = pytest.mark.skipif(
    os.environ.get("FINNHUB_API_KEY", "") in ("", "test"),
    reason="real FINNHUB_API_KEY required for e2e tests",
)

pytestmark = [pytest.mark.e2e, _real_key]

# Stable, high-volume ticker — reliably has baseline coverage.
_TICKER = "AAPL"

# ---------------------------------------------------------------------------
# fetch_news — raw API response structure
# ---------------------------------------------------------------------------


def test_fetch_news_returns_list():
    from datetime import date, timedelta

    today = date.today()
    articles = fetch_news(_TICKER, from_date=today - timedelta(days=7), to_date=today)
    assert isinstance(articles, list)


def test_fetch_news_articles_have_required_fields():
    """Every article consumed downstream must have datetime (int) and headline (str)."""
    from datetime import date, timedelta

    today = date.today()
    articles = fetch_news(_TICKER, from_date=today - timedelta(days=7), to_date=today)
    for article in articles:
        assert "datetime" in article, f"missing 'datetime' key: {article}"
        assert isinstance(article["datetime"], int), (
            f"'datetime' must be int, got {type(article['datetime'])}"
        )
        assert "headline" in article, f"missing 'headline' key: {article}"
        assert isinstance(article["headline"], str), (
            f"'headline' must be str, got {type(article['headline'])}"
        )


# ---------------------------------------------------------------------------
# compute_volume_stats — full pipeline invariants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stats():
    """Run compute_volume_stats once and share the result across all tests."""
    return compute_volume_stats(_TICKER)


def test_stats_has_all_required_keys(stats):
    required = {
        "recent_count",
        "mean",
        "std",
        "z_score",
        "classification",
        "headlines",
        "recent_headlines",
        "baseline_counts",
    }
    missing = required - stats.keys()
    assert not missing, f"missing keys in compute_volume_stats output: {missing}"


def test_stats_value_types(stats):
    assert isinstance(stats["recent_count"], int)
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
    assert isinstance(stats["z_score"], float)
    assert isinstance(stats["classification"], str)
    assert isinstance(stats["headlines"], list)
    assert isinstance(stats["recent_headlines"], list)
    assert isinstance(stats["baseline_counts"], list)


def test_recent_count_is_non_negative(stats):
    assert stats["recent_count"] >= 0


def test_mean_and_std_are_non_negative(stats):
    assert stats["mean"] >= 0.0
    assert stats["std"] >= 0.0


def test_z_score_is_not_nan(stats):
    assert not math.isnan(stats["z_score"]), "z_score must never be NaN"


def test_recent_count_matches_recent_headlines_length(stats):
    assert stats["recent_count"] == len(stats["recent_headlines"]), (
        "recent_count must equal len(recent_headlines)"
    )


def test_headlines_is_top_5_of_recent(stats):
    """headlines is capped at 5; recent_headlines is the uncapped full set."""
    assert len(stats["headlines"]) <= 5
    assert len(stats["headlines"]) == min(stats["recent_count"], 5)


def test_headlines_are_prefix_of_recent_headlines(stats):
    """headlines must be the first N elements of recent_headlines."""
    n = len(stats["headlines"])
    assert stats["headlines"] == stats["recent_headlines"][:n]


def test_baseline_counts_length_equals_baseline_days(stats):
    n = len(stats["baseline_counts"])
    assert n == BASELINE_DAYS, f"expected {BASELINE_DAYS} baseline buckets, got {n}"


def test_baseline_counts_are_non_negative(stats):
    for i, count in enumerate(stats["baseline_counts"]):
        assert count >= 0, f"baseline_counts[{i}] is negative: {count}"


def test_classification_is_valid(stats):
    assert stats["classification"] in {"normal", "elevated", "unusual"}


def test_classification_is_consistent_with_z_score(stats):
    """Classification must follow the same thresholds used by compute_volume_stats."""
    z = stats["z_score"]
    cls = stats["classification"]
    if z < THRESHOLD_ELEVATED:
        assert cls == "normal", (
            f"z={z:.2f} < THRESHOLD_ELEVATED={THRESHOLD_ELEVATED} but got {cls!r}"
        )
    elif z < THRESHOLD_UNUSUAL:
        assert cls == "elevated", (
            f"THRESHOLD_ELEVATED={THRESHOLD_ELEVATED} <= z={z:.2f} < "
            f"THRESHOLD_UNUSUAL={THRESHOLD_UNUSUAL} but got {cls!r}"
        )
    else:
        assert cls == "unusual", (
            f"z={z:.2f} >= THRESHOLD_UNUSUAL={THRESHOLD_UNUSUAL} but got {cls!r}"
        )


# ---------------------------------------------------------------------------
# finBERT sentiment — requires `uv sync --group sentiment`
# ---------------------------------------------------------------------------


def _transformers_installed() -> bool:
    try:
        import transformers  # noqa: F401  # type: ignore[import]

        return True
    except ImportError:
        return False


_sentiment_deps = pytest.mark.skipif(
    not _transformers_installed(),
    reason="transformers not installed — install with `uv sync --group sentiment`",
)


@_sentiment_deps
def test_score_headlines_result_length_matches_input(stats):
    """score_headlines returns exactly one entry per article."""
    from financial_news.config import load_config
    from financial_news.sentiment import score_headlines

    cfg = load_config()
    articles = stats["recent_articles"][:5] or [
        {"headline": "No recent news for AAPL.", "summary": ""}
    ]  # noqa: E501
    valid = frozenset(cfg.sentiment.labels)
    results = score_headlines(articles, cfg.sentiment.model_name, valid)
    assert len(results) == len(articles)


@_sentiment_deps
def test_score_headlines_result_has_required_keys(stats):
    """Each scored entry has headline, summary, label, and score keys."""
    from financial_news.config import load_config
    from financial_news.sentiment import score_headlines

    cfg = load_config()
    articles = stats["recent_articles"][:3] or [
        {"headline": "Fallback headline for AAPL.", "summary": ""}
    ]  # noqa: E501
    valid = frozenset(cfg.sentiment.labels)
    results = score_headlines(articles, cfg.sentiment.model_name, valid)
    for entry in results:
        assert "headline" in entry
        assert "summary" in entry
        assert "label" in entry
        assert "score" in entry


@_sentiment_deps
def test_score_headlines_labels_are_valid(stats):
    """All labels are in the configured valid set — no 'unavailable' with real model."""
    from financial_news.config import load_config
    from financial_news.sentiment import score_headlines

    cfg = load_config()
    valid = frozenset(cfg.sentiment.labels)
    articles = stats["recent_articles"][:5] or [
        {"headline": "Apple revenue rises on iPhone demand.", "summary": ""}
    ]
    results = score_headlines(articles, cfg.sentiment.model_name, valid)
    for entry in results:
        assert entry["label"] in valid, (
            f"unexpected label {entry['label']!r} for headline: {entry['headline']!r}"
        )


@_sentiment_deps
def test_score_headlines_scores_are_in_range(stats):
    """Confidence scores are valid softmax probabilities in [0, 1]."""
    from financial_news.config import load_config
    from financial_news.sentiment import score_headlines

    cfg = load_config()
    articles = stats["recent_articles"][:5] or [
        {"headline": "Apple reports quarterly results.", "summary": ""}
    ]  # noqa: E501
    results = score_headlines(
        articles, cfg.sentiment.model_name, frozenset(cfg.sentiment.labels)
    )
    for entry in results:
        assert 0.0 <= entry["score"] <= 1.0, (
            f"score {entry['score']} out of [0, 1] for headline: {entry['headline']!r}"
        )
        assert not math.isnan(entry["score"]), (
            f"NaN score for headline: {entry['headline']!r}"
        )


@_sentiment_deps
def test_score_headlines_headline_preserved(stats):
    """The input headline string is returned unchanged in each result entry."""
    from financial_news.config import load_config
    from financial_news.sentiment import score_headlines

    cfg = load_config()
    articles = stats["recent_articles"][:3] or [
        {"headline": "Apple introduces new MacBook.", "summary": ""}
    ]  # noqa: E501
    results = score_headlines(
        articles, cfg.sentiment.model_name, frozenset(cfg.sentiment.labels)
    )
    for article, entry in zip(articles, results):
        assert entry["headline"] == article["headline"]


@_sentiment_deps
def test_enrich_stats_full_pipeline(stats):
    """enrich_stats produces valid headline_sentiment and selected_articles."""
    from financial_news.config import load_config
    from financial_news.enrichment import EnrichmentConfig, enrich_stats

    cfg = load_config()
    enrich_cfg = EnrichmentConfig(
        model_name=cfg.sentiment.model_name,
        valid_labels=frozenset(cfg.sentiment.labels),
        confidence_threshold=cfg.briefing.confidence_threshold,
        min_articles=cfg.briefing.prompt_headlines_min,
        max_articles=cfg.briefing.prompt_headlines_max,
    )
    result = enrich_stats(stats, enrich_cfg)
    assert "headline_sentiment" in result
    assert "selected_articles" in result

    valid = frozenset(cfg.sentiment.labels)
    for entry in result["headline_sentiment"]:
        assert entry["label"] in valid
        assert 0.0 <= entry["score"] <= 1.0
        assert not math.isnan(entry["score"])


@_sentiment_deps
def test_enrich_stats_zero_news_ticker_skips_scoring():
    """Zero-news ticker (no recent or baseline articles) gets empty sentiment list."""
    from financial_news.config import load_config
    from financial_news.enrichment import EnrichmentConfig, enrich_stats

    cfg = load_config()
    enrich_cfg = EnrichmentConfig(
        model_name=cfg.sentiment.model_name,
        valid_labels=frozenset(cfg.sentiment.labels),
        confidence_threshold=cfg.briefing.confidence_threshold,
        min_articles=cfg.briefing.prompt_headlines_min,
        max_articles=cfg.briefing.prompt_headlines_max,
    )
    zero_news = {
        "ticker": "SYNTHETIC_ZERO",
        "recent_count": 0,
        "mean": 0.0,
        "std": 0.0,
        "z_score": 0.0,
        "classification": "normal",
        "headlines": [],
        "recent_headlines": [],
        "recent_articles": [],
        "baseline_counts": [0.0] * cfg.analysis.baseline_days,
    }
    result = enrich_stats(zero_news, enrich_cfg)
    assert result["headline_sentiment"] == []
    assert result["selected_articles"] == []
