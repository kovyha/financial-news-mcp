"""Centralised enrichment pipeline for news volume stats.

Combines volume statistics (analysis) with finBERT scoring and
confidence-threshold article selection (sentiment) into a single enriched
stats dict consumed by both the MCP server and the briefing agent.
"""

import logging
from dataclasses import dataclass

from financial_news import sentiment
from financial_news.analysis import compute_volume_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentConfig:
    model_name: str
    valid_labels: frozenset[str]
    confidence_threshold: float
    min_articles: int
    max_articles: int


def select_articles(
    scored: list[dict],
    classification: str,
    confidence_threshold: float,
    min_articles: int,
    max_articles: int,
) -> list[dict]:
    """Select articles for LLM context using confidence-threshold filtering.

    For elevated/unusual tickers, high-confidence neutral articles are discarded
    first — they add noise without directional signal. Then selects articles
    scoring >= confidence_threshold, clamped to [min_articles, max_articles].
    When too few meet the threshold, top-scoring articles fill to min_articles.
    """
    candidates = scored
    if classification in ("elevated", "unusual"):
        non_neutral = [
            item
            for item in scored
            if item["label"] != "neutral" or item["score"] < confidence_threshold
        ]
        if non_neutral:
            candidates = non_neutral

    by_score = sorted(candidates, key=lambda x: x["score"], reverse=True)
    confident = [item for item in by_score if item["score"] >= confidence_threshold]
    if len(confident) < min_articles:
        return by_score[:min_articles]
    return confident[:max_articles]


def enrich_stats(stats: dict, cfg: EnrichmentConfig) -> dict:
    """Enrich a pre-fetched stats dict with finBERT scores and selected articles.

    Error entries (dicts containing an 'error' key) are passed through unchanged.
    For all other entries, scores all of today's articles (falling back to headline
    strings from older snapshots), applies confidence-threshold filtering with
    neutral-article discarding for elevated/unusual tickers, and stores both the
    full scored list and the selected subset back into the returned stats dict.

    Also scores the multi-day headline_articles window (if present) using the same
    neutral-filtering logic, producing selected_headline_articles for LLM context.
    date and source fields are preserved via zip-join since score_headlines only
    outputs headline/summary/label/score.
    """
    if "error" in stats:
        return stats

    ticker = stats.get("ticker", "?")

    source = stats.get("recent_articles") or []
    if not source:
        strings = stats.get("recent_headlines") or stats.get("headlines") or []
        source = [{"headline": h, "summary": ""} for h in strings]

    if not source:
        logger.info("ticker=%s has no articles to score", ticker)
        return {
            **stats,
            "headline_sentiment": [],
            "selected_articles": [],
            "selected_headline_articles": [],
        }

    logger.info("ticker=%s articles_to_score=%d", ticker, len(source))
    all_scored = sentiment.score_headlines(source, cfg.model_name, cfg.valid_labels)

    label_counts: dict[str, int] = {}
    for item in all_scored:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
    logger.info(
        "ticker=%s scored=%d label_counts=%s", ticker, len(all_scored), label_counts
    )

    selected = select_articles(
        all_scored,
        stats["classification"],
        cfg.confidence_threshold,
        cfg.min_articles,
        cfg.max_articles,
    )
    for item in selected:
        logger.info(
            "selected_article ticker=%s label=%s score=%.4f headline=%r",
            ticker,
            item["label"],
            item["score"],
            item["headline"],
        )

    # Score the multi-day headline window with the same neutral-filtering logic.
    # score_headlines drops extra keys (date, source), so zip-join them back.
    headline_articles = stats.get("headline_articles") or []
    if headline_articles:
        logger.info(
            "ticker=%s headline_window_to_score=%d", ticker, len(headline_articles)
        )
        raw_scores = sentiment.score_headlines(
            headline_articles, cfg.model_name, cfg.valid_labels
        )
        headline_scored = [
            {**orig, "label": s["label"], "score": s["score"]}
            for orig, s in zip(headline_articles, raw_scores)
        ]
        selected_headline_articles = select_articles(
            headline_scored,
            stats["classification"],
            cfg.confidence_threshold,
            cfg.min_articles,
            cfg.max_articles,
        )
    else:
        selected_headline_articles = []

    return {
        **stats,
        "headline_sentiment": all_scored,
        "selected_articles": selected,
        "selected_headline_articles": selected_headline_articles,
    }


def enrich_ticker(symbol: str, cfg: EnrichmentConfig) -> dict:
    """Fetch volume stats for symbol and enrich with finBERT sentiment."""
    stats = compute_volume_stats(symbol)
    return enrich_stats({**stats, "ticker": symbol}, cfg)
