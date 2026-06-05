"""finBERT sentiment scoring for news articles.

Deterministic preprocessing step for the briefing agent. Scores each article
as positive/negative/neutral using a configurable HuggingFace model (default:
ProsusAI/finbert). When a summary is present it is appended to the headline
before scoring so finBERT has more signal; the original headline is preserved
in the output. Requires the optional `sentiment` dependency group
(transformers + torch).

Gracefully degrades to label='unavailable' when the group is not installed.
"""

import logging

logger = logging.getLogger(__name__)

# Module-level pipeline state — loaded once on first call to score_headlines.
_pipeline = None
_pipeline_attempted = False
_pipeline_model: str | None = None


def _get_pipeline(model_name: str):
    """Return the finBERT pipeline, loaded once. Returns None if unavailable.

    If called with a different model_name than the previously loaded pipeline,
    the existing pipeline is replaced.
    """
    global _pipeline, _pipeline_attempted, _pipeline_model
    if _pipeline_attempted and _pipeline_model == model_name:
        return _pipeline
    _pipeline_attempted = True
    _pipeline_model = model_name
    _pipeline = None
    try:
        from transformers import pipeline  # type: ignore[import]

        _pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU
        )
        logger.info("finBERT pipeline loaded (model=%s)", model_name)
    except ImportError:
        logger.warning(
            "transformers not installed; finBERT unavailable — install with "
            "`uv sync --group sentiment`"
        )
    except Exception as exc:
        logger.warning("finBERT pipeline load failed: %s", exc)
    return _pipeline


def _compose_text(headline: str, summary: str) -> str:
    """Build the text to score: appends summary to headline when available."""
    return f"{headline}. {summary}" if summary else headline


def score_headlines(
    articles: list[dict], model_name: str, valid_labels: frozenset[str]
) -> list[dict]:
    """Score each article with finBERT sentiment using headline + summary.

    Each article dict must have a 'headline' key and an optional 'summary' key.
    When summary is non-empty it is appended to the headline so finBERT has more
    signal; the original headline is preserved in the output.

    Returns a list of {headline, summary, label, score} dicts where label is one
    of the values in valid_labels, or 'unavailable' when the transformers
    dependency group is not installed or the model returns an unexpected label.
    """
    if not articles:
        return []
    texts = [_compose_text(a["headline"], a.get("summary") or "") for a in articles]
    pipe = _get_pipeline(model_name)
    if pipe is None:
        return [
            {
                "headline": a["headline"],
                "summary": a.get("summary") or "",
                "label": "unavailable",
                "score": 0.0,
            }
            for a in articles
        ]
    with_summary = sum(1 for a in articles if a.get("summary"))
    logger.debug(
        "score_headlines articles=%d with_summary=%d headline_only=%d",
        len(articles),
        with_summary,
        len(articles) - with_summary,
    )
    try:
        results = pipe(texts, truncation=True, max_length=512)
        scored = []
        for a, r in zip(articles, results):
            label = r["label"].lower()
            if label not in valid_labels:
                logger.warning(
                    "unexpected finBERT label %r for model %s; marking unavailable",
                    label,
                    model_name,
                )
                label = "unavailable"
            scored.append(
                {
                    "headline": a["headline"],
                    "summary": a.get("summary") or "",
                    "label": label,
                    "score": round(r["score"], 3),
                }
            )
        return scored
    except Exception as exc:
        logger.warning("finBERT inference failed: %s", exc)
        return [
            {
                "headline": a["headline"],
                "summary": a.get("summary") or "",
                "label": "unavailable",
                "score": 0.0,
            }
            for a in articles
        ]
