"""finBERT sentiment scoring for news headlines.

Deterministic preprocessing step for the briefing agent. Scores each headline
as positive/negative/neutral using a configurable HuggingFace model (default:
ProsusAI/finbert). Requires the optional `sentiment` dependency group
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


def score_headlines(
    headlines: list[str], model_name: str, valid_labels: frozenset[str]
) -> list[dict]:
    """Score each headline with finBERT sentiment.

    Returns a list of {headline, label, score} dicts where label is one of the
    values in valid_labels, or 'unavailable' when the transformers dependency
    group is not installed or the model returns an unexpected label.
    """
    if not headlines:
        return []
    pipe = _get_pipeline(model_name)
    if pipe is None:
        return [
            {"headline": h, "label": "unavailable", "score": 0.0} for h in headlines
        ]
    try:
        results = pipe(headlines, truncation=True, max_length=512)
        scored = []
        for h, r in zip(headlines, results):
            label = r["label"].lower()
            if label not in valid_labels:
                logger.warning(
                    "unexpected finBERT label %r for model %s; marking unavailable",
                    label,
                    model_name,
                )
                label = "unavailable"
            scored.append(
                {"headline": h, "label": label, "score": round(r["score"], 3)}
            )
        return scored
    except Exception as exc:
        logger.warning("finBERT inference failed: %s", exc)
        return [
            {"headline": h, "label": "unavailable", "score": 0.0} for h in headlines
        ]
