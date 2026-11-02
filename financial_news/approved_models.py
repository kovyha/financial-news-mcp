"""Sentiment models approved for use, each pinned to a Hugging Face commit.

Pinning means the weights can't change underneath us. Adding a model or moving a
pin needs review (this file is protected) and a fresh evaluation run
(docs/evaluation.md).
"""

APPROVED_SENTIMENT_MODELS: dict[str, str] = {
    "ProsusAI/finbert": "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
}
