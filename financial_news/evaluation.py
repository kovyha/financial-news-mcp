"""Baseline comparison harness for the news-driven/noise detection task.

Builds a labeled historical dataset of price-move days already flagged
elevated/unusual by compute_price_stats (via its as_of_date parameter — see
financial_news.price/financial_news.analysis; this module adds no parallel
historical-stats implementation), then compares three independent predictors of
whether each flagged move was actually news-driven ("real move") or not ("noise")
against human-confirmed ground truth:

1. news-volume z-score only — coincident article-count burst, independent of
   price and of what the coverage says (predict_news_zscore_only)
2. name-mention fraction only — fraction of that day's headlines that explicitly
   name the company, vs. merely appearing in its Finnhub feed (predict_name_mention)
3. full system — finBERT selection + both z-scores + LLM reasoning
   (predict_full_system), mirroring briefing.py's cross-referencing but scoped to
   a single historical candidate day instead of a live watchlist briefing

Price z-score magnitude (predict_zscore_only) is deliberately NOT a fourth
baseline here: the candidate pool itself is selected by
compute_price_stats(...).classification in (elevated, unusual), i.e. |price
z-score| >= cfg.price.threshold_elevated. Every candidate already clears that
bar by construction, so testing the same threshold against the same pool is
circular (100% recall, precision == the pool's raw base rate) — it measures the
pool definition, not the signal. What run_report() DOES report for
predict_zscore_only is a sensitivity check at the stricter threshold_unusual cut
only: does tightening the pool's own selection threshold trade recall for
precision. That's a real, useful finding (see docs/evaluation.md), but it answers
a different question than the three baselines above and is reported separately
rather than folded into the same comparison table.

A finBERT-sentiment-only baseline was tried and dropped: sentiment polarity
(positive/negative/confidence) measures whether news reads good or bad, not
whether it's genuinely about the ticker or explains the move — on its own, an axis
orthogonal to this task. It showed ~zero separation between news_driven and noise
candidates even conditioned on name-mention. Polarity signed against the move's
direction is an exploratory exception, not yet confirmed (see docs/evaluation.md).

See docs/evaluation.md for methodology and ground-truth caveats (small-N,
human-labeled by the author reading only price move + headlines).

Run via:
    FINNHUB_API_KEY=<key> uv run python -m financial_news.evaluation build-candidates
    FINNHUB_API_KEY=<key> ANTHROPIC_API_KEY=<key> \\
        uv run python -m financial_news.evaluation report
    FINNHUB_API_KEY=<key> ANTHROPIC_API_KEY=<key> \\
        uv run python -m financial_news.evaluation ablate
"""

import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import anthropic
import pandas_market_calendars as mcal

from financial_news import log_setup  # noqa: F401
from financial_news.analysis import compute_volume_stats
from financial_news.config import Config, load_config
from financial_news.enrichment import EnrichmentConfig, select_articles
from financial_news.price import compute_price_stats
from financial_news.sentiment import score_headlines

logger = logging.getLogger(__name__)

_CANDIDATES_PATH = Path("eval/candidates.json")
_LABELS_PATH = Path("eval/labels.json")
_FULL_SYSTEM_VERDICTS_PATH = Path("eval/baseline3_reasoning.json")
_FINBERT_ABLATION_PATH = Path("eval/finbert_ablation.json")
_ABLATION_PATH = Path("eval/ablation.json")

_LABEL_VALUES = frozenset({"news_driven", "noise", "ambiguous"})

_MODEL = "claude-opus-4-7"
_MAX_TOKENS = 300
_NEWS_ZSCORE_CLAUSE = " the news-volume z-score for the same day vs a 30-day baseline,"


def _judge_system_prompt(include_news_zscore: bool = True) -> str:
    """System prompt for predict_full_system. The default (include_news_zscore=
    True) is byte-identical to the prompt behind eval/baseline3_reasoning.json and
    eval/finbert_ablation.json, so stored verdicts stay comparable with new arms.
    """
    return (
        "You are auditing a single flagged price move for a financial monitoring"
        " system. You are given the price move and its z-score vs a volatility"
        " baseline,"
        + (_NEWS_ZSCORE_CLAUSE if include_news_zscore else "")
        + " and that day's headlines (with finBERT sentiment labels where available)."
        " Decide whether this specific price move looks news-driven or looks like"
        " statistical noise (mean reversion, a thin-liquidity print, a sector-wide"
        " move mislabeled as single-name, options-expiry mechanics, etc). Consider"
        " confounders explicitly. Respond with exactly one line in the form"
        " 'VERDICT: news_driven | REASON: <one sentence>' or"
        " 'VERDICT: noise | REASON: <one sentence>'."
    )


# ── candidate pool ────────────────────────────────────────────────────────────


def _trading_days(start: date, end: date) -> list[date]:
    """Trading days in [start, end] per the NYSE calendar.

    NASDAQ shares the same holiday calendar, so this is valid for both — avoids
    an extra yfinance call purely for date discovery.
    """
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [ts.date() for ts in schedule.index]


def build_candidate_pool(
    tickers: list[str],
    start_date: date,
    end_date: date,
    fetch_delay_seconds: float = 0.0,
) -> list[dict]:
    """Build the evaluation candidate pool over [start_date, end_date].

    One row per (ticker, trading day) whose price-move classification is
    elevated/unusual — normal-move days aren't a meaningful test of "was this
    news-driven," so they're excluded up front.
    """
    candidates: list[dict] = []
    for symbol in tickers:
        for day in _trading_days(start_date, end_date):
            try:
                price_stats = compute_price_stats(symbol, as_of_date=day)
            except Exception:
                logger.exception(
                    "build_candidate_pool price fetch failed symbol=%s date=%s",
                    symbol,
                    day,
                )
                continue
            if fetch_delay_seconds > 0:
                time.sleep(fetch_delay_seconds)
            if price_stats["classification"] not in ("elevated", "unusual"):
                continue
            try:
                news_stats = compute_volume_stats(symbol, as_of_date=day)
            except Exception:
                logger.exception(
                    "build_candidate_pool news fetch failed symbol=%s date=%s",
                    symbol,
                    day,
                )
                continue
            candidates.append(
                {
                    "ticker": symbol,
                    "date": day.isoformat(),
                    "price": price_stats,
                    "news": news_stats,
                }
            )
    logger.info(
        "build_candidate_pool tickers=%d start=%s end=%s candidates=%d",
        len(tickers),
        start_date.isoformat(),
        end_date.isoformat(),
        len(candidates),
    )
    return candidates


def save_candidates(candidates: list[dict], path: Path = _CANDIDATES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates, indent=2, sort_keys=True))


def load_candidates(path: Path = _CANDIDATES_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ── ground-truth labels ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Label:
    ticker: str
    date: str
    label: str  # "news_driven" | "noise" | "ambiguous"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.label not in _LABEL_VALUES:
            raise ValueError(
                f"Invalid label {self.label!r}; must be one of {sorted(_LABEL_VALUES)}"
            )


def _label_key(ticker: str, date_str: str) -> str:
    return f"{ticker}:{date_str}"


def load_labels(path: Path = _LABELS_PATH) -> dict[str, Label]:
    """Load confirmed labels keyed by 'TICKER:YYYY-MM-DD'. Missing file -> {}."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {_label_key(item["ticker"], item["date"]): Label(**item) for item in raw}


def save_labels(labels: dict[str, Label], path: Path = _LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(label) for label in labels.values()]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def draft_label_inputs(candidate: dict) -> dict:
    """The minimal inputs a human labeler should see: return_pct + that day's
    headlines only. Deliberately excludes z-scores/classification/finBERT/LLM
    fields so ground truth isn't drafted from the very signals under evaluation.
    """
    return {
        "ticker": candidate["ticker"],
        "date": candidate["date"],
        "return_pct": candidate["price"]["return_pct"],
        "headlines": [
            {"headline": a["headline"], "summary": a.get("summary", "")}
            for a in candidate["news"]["recent_articles"]
        ],
    }


def unlabeled_candidates(
    candidates: list[dict], labels: dict[str, Label]
) -> list[dict]:
    """Candidates not yet present in *labels*, for incremental labeling."""
    return [c for c in candidates if _label_key(c["ticker"], c["date"]) not in labels]


# ── baseline predictors ──────────────────────────────────────────────────────


def predict_zscore_only(candidate: dict, threshold: float) -> bool:
    """Predict 'real move' iff the price z-score magnitude alone clears
    *threshold*. NOT one of the three independent baselines (see module
    docstring) — the candidate pool is itself selected on this same statistic at
    threshold_elevated, so evaluating at threshold_elevated is circular. Used by
    run_report() only as a sensitivity check at the stricter threshold_unusual
    cut, reported separately from the baseline comparison table.
    """
    return abs(candidate["price"]["z_score"]) >= threshold


def predict_news_zscore_only(candidate: dict, threshold: float) -> bool:
    """Predict 'real move' iff the news-volume z-score clears *threshold* — a
    coincident burst of ticker-tagged article-count volume, independent of price
    and of what the coverage actually says.
    """
    return candidate["news"]["z_score"] >= threshold


# Hand-curated ticker -> alias table for the current watchlist, used by
# _headline_mentions_company. Bare tickers are included only where they aren't
# ambiguous with common words or short enough to false-match inside unrelated
# tokens (e.g. Ford's "F" would match inside "F-150"), which is why "F" and
# "PLUG" rely on the company name alone.
_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("NVIDIA", "Nvidia"),
    "TSLA": ("Tesla", "TSLA"),
    "META": ("Meta", "META"),
    "AMD": ("AMD", "Advanced Micro Devices"),
    "F": ("Ford",),
    "BAC": ("BAC", "Bank of America"),
    "GME": ("GME", "GameStop"),
    "PLUG": ("Plug Power",),
    "XOM": ("XOM", "Exxon", "ExxonMobil", "Exxon Mobil"),
    "UNH": ("UNH", "UnitedHealth"),
    "SPCX": ("SPCX", "SpaceX"),
}


def _headline_mentions_company(headline: str, ticker: str) -> bool:
    """Whether *headline* explicitly names the company, via _TICKER_ALIASES —
    not just "is in that ticker's Finnhub feed," which every headline here
    already is. Falls back to the bare ticker for symbols not in the table.
    """
    aliases = _TICKER_ALIASES.get(ticker, (ticker,))
    return any(re.search(rf"\b{re.escape(alias)}\b", headline) for alias in aliases)


def predict_name_mention(candidate: dict, min_fraction: float) -> bool:
    """Predict 'real move' iff at least *min_fraction* of that day's headlines
    explicitly name the company — a cheap, deterministic proxy for on-topic
    coverage (vs. roundup/market-wide headlines that merely appear in the
    ticker's news feed), with no sentiment or LLM reasoning involved.
    """
    articles = candidate["news"]["recent_articles"]
    if not articles:
        return False
    mentions = sum(
        1
        for a in articles
        if _headline_mentions_company(a["headline"], candidate["ticker"])
    )
    return (mentions / len(articles)) >= min_fraction


def _finbert_selected_articles(
    candidate: dict, enrich_cfg: EnrichmentConfig
) -> list[dict]:
    """finBERT-score that day's articles and apply the live selection filter
    (enrichment.select_articles) — the F component shared by predict_full_system
    and predict_news_zscore_with_sentiment.
    """
    articles = candidate["news"]["recent_articles"]
    if not articles:
        return []
    scored = score_headlines(articles, enrich_cfg.model_name, enrich_cfg.valid_labels)
    if not scored:
        return []
    return select_articles(
        scored,
        candidate["price"]["classification"],
        enrich_cfg.confidence_threshold,
        enrich_cfg.min_articles,
        enrich_cfg.max_articles,
    )


def predict_news_zscore_with_sentiment(
    candidate: dict, threshold: float, enrich_cfg: EnrichmentConfig
) -> bool:
    """Ablation arm Z+F (no LLM): predict 'real move' iff the news-volume z-score
    clears *threshold* AND at least one finBERT-selected article is confidently
    non-neutral (score >= confidence_threshold). Compared against
    predict_news_zscore_only at the same threshold, this isolates what finBERT
    adds to the news-volume signal without any LLM reasoning. One reasonable
    rule, defined after the fact — not a tuned optimum.
    """
    if not predict_news_zscore_only(candidate, threshold):
        return False
    return any(
        item["label"] in enrich_cfg.valid_labels
        and item["label"] != "neutral"
        and item["score"] >= enrich_cfg.confidence_threshold
        for item in _finbert_selected_articles(candidate, enrich_cfg)
    )


def predict_full_system(
    candidate: dict,
    enrich_cfg: EnrichmentConfig,
    use_finbert: bool = True,
    include_news_zscore: bool = True,
) -> tuple[bool, str]:
    """Baseline 3: finBERT-selected headlines + both z-scores fed to Claude for a
    single structured news-driven/noise judgment. Returns (is_news_driven, raw
    response text).

    The two flags are the component ablations (see run_ablation):
    use_finbert=False skips sentiment scoring entirely, passes the most recent
    max_articles headlines (recent_articles is already newest-first) and omits
    the [label score] tags from the prompt. include_news_zscore=False drops the
    news-volume z-score from both the user and system prompt; the price move and
    its z-score always stay, since the candidate pool is gated on them.
    """
    if use_finbert:
        selected = _finbert_selected_articles(candidate, enrich_cfg)
    else:
        selected = candidate["news"]["recent_articles"][: enrich_cfg.max_articles]

    headline_block = (
        "\n".join(
            "- "
            + (f"[{item['label']} {item['score']:.2f}] " if use_finbert else "")
            + item["headline"]
            + (f"\n  {item['summary']}" if item.get("summary") else "")
            for item in selected
        )
        or "(no headlines that day)"
    )

    news_line = (
        f"News volume: z={candidate['news']['z_score']:.2f}"
        f" ({candidate['news']['classification']})\n"
        if include_news_zscore
        else ""
    )
    prompt = (
        f"Ticker: {candidate['ticker']}\nDate: {candidate['date']}\n"
        f"Price move: {candidate['price']['return_pct']:+.2%}"
        f" (z={candidate['price']['z_score']:.2f},"
        f" {candidate['price']['classification']})\n"
        f"{news_line}"
        f"Headlines:\n{headline_block}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_judge_system_prompt(include_news_zscore),
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(b.text for b in response.content if b.type == "text")
    lowered = text.lower()
    if "verdict: news_driven" in lowered:
        return True, text
    if "verdict: noise" in lowered:
        return False, text
    logger.warning(
        "predict_full_system unparseable response ticker=%s date=%s text=%r",
        candidate["ticker"],
        candidate["date"],
        text,
    )
    return False, text


# ── metrics ───────────────────────────────────────────────────────────────────


def precision_recall_f1(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Precision/recall/F1/accuracy for a binary news_driven=True vs noise=False
    prediction task, plus the underlying confusion-matrix counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(y_true),
    }


def mcnemar_exact(y_true: list[bool], pred_a: list[bool], pred_b: list[bool]) -> dict:
    """Exact (binomial) two-sided McNemar test on paired predictions: whether
    arms A and B differ in accuracy beyond chance, using only the rows where
    exactly one of them is right. At N≈43 this is the honest way to read a
    one- or two-verdict difference between arms.
    """
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must be the same length")
    only_a = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a == t and b != t)
    only_b = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if b == t and a != t)
    n = only_a + only_b
    k = min(only_a, only_b)
    p_value = (
        min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    )
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p_value}


# ── report ────────────────────────────────────────────────────────────────────


def _labeled_rows(
    candidates: list[dict], labels: dict[str, Label]
) -> list[tuple[dict, Label]]:
    """Candidates with a confirmed, non-ambiguous label, paired with that label."""
    rows = []
    for c in candidates:
        label = labels.get(_label_key(c["ticker"], c["date"]))
        if label is None or label.label == "ambiguous":
            continue
        rows.append((c, label))
    return rows


def _enrich_config(cfg: Config) -> EnrichmentConfig:
    return EnrichmentConfig(
        model_name=cfg.sentiment.model_name,
        valid_labels=frozenset(cfg.sentiment.labels),
        confidence_threshold=cfg.briefing.confidence_threshold,
        min_articles=cfg.briefing.prompt_headlines_min,
        max_articles=cfg.briefing.prompt_headlines_max,
    )


def run_report(
    candidates: list[dict],
    labels: dict[str, Label],
    cfg: Config,
    price_zscore_sensitivity_thresholds: list[float] | None = None,
    news_zscore_thresholds: list[float] | None = None,
    name_mention_thresholds: list[float] | None = None,
) -> dict:
    """Run the three baselines plus the price-z-score sensitivity check against
    confirmed labels and return a report dict.
    """
    rows = _labeled_rows(candidates, labels)
    if not rows:
        raise ValueError("No confirmed (non-ambiguous) labels to evaluate against.")
    y_true = [label.label == "news_driven" for _, label in rows]

    # Sensitivity check only, not a baseline — see predict_zscore_only docstring.
    # threshold_elevated is excluded: every candidate already clears it by pool
    # construction, so evaluating there is circular (100% recall, precision ==
    # the pool's base rate).
    sensitivity_thresholds = price_zscore_sensitivity_thresholds or [
        cfg.price.threshold_unusual,
    ]
    price_zscore_sensitivity = {
        threshold: precision_recall_f1(
            y_true, [predict_zscore_only(c, threshold) for c, _ in rows]
        )
        for threshold in sensitivity_thresholds
    }

    news_thresholds = news_zscore_thresholds or [
        cfg.analysis.threshold_elevated,
        cfg.analysis.threshold_unusual,
    ]
    news_zscore_results = {
        threshold: precision_recall_f1(
            y_true, [predict_news_zscore_only(c, threshold) for c, _ in rows]
        )
        for threshold in news_thresholds
    }

    mention_thresholds = name_mention_thresholds or [0.25, 0.33, 0.5]
    name_mention_results = {
        threshold: precision_recall_f1(
            y_true, [predict_name_mention(c, threshold) for c, _ in rows]
        )
        for threshold in mention_thresholds
    }

    enrich_cfg = _enrich_config(cfg)
    system_pred, system_reasoning = [], []
    for c, _ in rows:
        is_real, text = predict_full_system(c, enrich_cfg)
        system_pred.append(is_real)
        system_reasoning.append(text)
    system_result = precision_recall_f1(y_true, system_pred)

    return {
        "n_labeled": len(rows),
        "price_zscore_sensitivity": price_zscore_sensitivity,
        "news_zscore_only": news_zscore_results,
        "name_mention": name_mention_results,
        "full_system": system_result,
        "full_system_reasoning": system_reasoning,
    }


def _print_report(report: dict) -> None:
    print()
    print("=" * 60)
    print(f"BASELINE COMPARISON — n={report['n_labeled']} labeled candidates")
    print("=" * 60)

    print("\nBaseline: news-volume z-score only")
    for threshold, result in report["news_zscore_only"].items():
        print(
            f"  threshold={threshold:.1f}  precision={result['precision']:.2f}"
            f" recall={result['recall']:.2f} f1={result['f1']:.2f}"
            f" accuracy={result['accuracy']:.2f}"
        )

    print("\nBaseline: name-mention fraction only")
    for threshold, result in report["name_mention"].items():
        print(
            f"  min_fraction={threshold:.2f}  precision={result['precision']:.2f}"
            f" recall={result['recall']:.2f} f1={result['f1']:.2f}"
            f" accuracy={result['accuracy']:.2f}"
        )

    r = report["full_system"]
    print("\nFull system (finBERT selection + both z-scores + LLM reasoning)")
    print(
        f"  precision={r['precision']:.2f} recall={r['recall']:.2f}"
        f" f1={r['f1']:.2f} accuracy={r['accuracy']:.2f}"
    )
    print("=" * 60)

    print(
        "\nSensitivity check (NOT an independent baseline — all candidates already"
        "\nclear threshold_elevated by pool construction; this only tests"
        "\ntightening the pool's own selection cut to threshold_unusual):"
    )
    print("  price z-score threshold")
    for threshold, result in report["price_zscore_sensitivity"].items():
        print(
            f"  threshold={threshold:.1f}  precision={result['precision']:.2f}"
            f" recall={result['recall']:.2f} f1={result['f1']:.2f}"
            f" accuracy={result['accuracy']:.2f}"
        )
    print()


# ── component ablation ────────────────────────────────────────────────────────


def _load_stored_verdicts(
    path: Path, verdict_key: str, reasoning_key: str
) -> dict[str, tuple[bool, str]]:
    """Per-row verdicts from an earlier Claude run, keyed like labels.json."""
    return {
        _label_key(row["ticker"], row["date"]): (
            row[verdict_key] == "news_driven",
            row[reasoning_key],
        )
        for row in json.loads(path.read_text())
    }


def run_ablation(
    candidates: list[dict],
    labels: dict[str, Label],
    cfg: Config,
    full_system_path: Path = _FULL_SYSTEM_VERDICTS_PATH,
    finbert_ablation_path: Path = _FINBERT_ABLATION_PATH,
) -> dict:
    """Component ablation of the full system: Z = news-volume z-score,
    F = finBERT selection, L = Claude. Price z and the raw move stay in every LLM
    prompt — the candidate pool is gated on price z, so it can't be ablated.

    Z+F+L and Z+L reuse stored verdicts (two identical Z+F+L runs flipped 0/43,
    see docs/evaluation.md) rather than re-calling Claude; F+L and L alone are
    run fresh (one Claude call per labeled row each); Z+F is deterministic and
    swept at the news-volume thresholds alongside Z alone for comparison. Every
    arm gets a McNemar test against Z+F+L.
    """
    rows = _labeled_rows(candidates, labels)
    if not rows:
        raise ValueError("No confirmed (non-ambiguous) labels to evaluate against.")
    keys = [_label_key(c["ticker"], c["date"]) for c, _ in rows]
    y_true = [label.label == "news_driven" for _, label in rows]

    stored = {
        "Z+F+L": _load_stored_verdicts(full_system_path, "predicted", "reasoning"),
        "Z+L": _load_stored_verdicts(
            finbert_ablation_path, "without_finbert", "without_finbert_reasoning"
        ),
    }
    for arm, verdicts in stored.items():
        missing = [k for k in keys if k not in verdicts]
        if missing:
            raise ValueError(
                f"Stored {arm} verdicts are missing labeled rows {missing} —"
                " re-run that arm before ablating, or the arms aren't comparable."
            )

    enrich_cfg = _enrich_config(cfg)
    preds: dict[str, list[bool]] = {}
    reasoning: dict[str, list[str]] = {}
    for arm, verdicts in stored.items():
        preds[arm] = [verdicts[k][0] for k in keys]
        reasoning[arm] = [verdicts[k][1] for k in keys]
    for arm, use_finbert in (("F+L", True), ("L", False)):
        preds[arm], reasoning[arm] = [], []
        for c, _ in rows:
            is_real, text = predict_full_system(
                c, enrich_cfg, use_finbert=use_finbert, include_news_zscore=False
            )
            preds[arm].append(is_real)
            reasoning[arm].append(text)
    for threshold in (cfg.analysis.threshold_elevated, cfg.analysis.threshold_unusual):
        preds[f"Z+F (θ={threshold:.1f})"] = [
            predict_news_zscore_with_sentiment(c, threshold, enrich_cfg)
            for c, _ in rows
        ]
        preds[f"Z (θ={threshold:.1f})"] = [
            predict_news_zscore_only(c, threshold) for c, _ in rows
        ]

    full = preds["Z+F+L"]
    return {
        "n_labeled": len(rows),
        "arms": {
            arm: {
                "metrics": precision_recall_f1(y_true, pred),
                "vs_full": (
                    None if arm == "Z+F+L" else mcnemar_exact(y_true, pred, full)
                ),
            }
            for arm, pred in preds.items()
        },
        "rows": [
            {
                "ticker": c["ticker"],
                "date": c["date"],
                "true_label": label.label,
                "verdicts": {
                    arm: "news_driven" if pred[i] else "noise"
                    for arm, pred in preds.items()
                },
                "reasoning": {arm: texts[i] for arm, texts in reasoning.items()},
            }
            for i, (c, label) in enumerate(rows)
        ],
    }


def _print_ablation(result: dict) -> None:
    print()
    print("=" * 78)
    print(f"COMPONENT ABLATION — n={result['n_labeled']} labeled candidates")
    print("Z=news-volume z  F=finBERT  L=Claude  (price z in every LLM prompt)")
    print("=" * 78)
    print(
        f"  {'arm':<14} {'prec':>5} {'rec':>5} {'f1':>5} {'acc':>5}"
        f"  {'tp/fp/fn/tn':<12} vs Z+F+L (only arm / only full, p)"
    )
    for arm, r in result["arms"].items():
        m, vs = r["metrics"], r["vs_full"]
        counts = f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
        paired = (
            "—"
            if vs is None
            else f"{vs['only_a_correct']} / {vs['only_b_correct']},"
            f" p={vs['p_value']:.3f}"
        )
        print(
            f"  {arm:<14} {m['precision']:>5.2f} {m['recall']:>5.2f}"
            f" {m['f1']:>5.2f} {m['accuracy']:>5.2f}  {counts:<12} {paired}"
        )
    print()


def main() -> int:
    cfg = load_config()
    args = sys.argv[1:]
    command = args[0] if args else "report"

    if command == "build-candidates":
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=90)  # ~60 trading days
        candidates = build_candidate_pool(
            cfg.monitor.tickers,
            start,
            end,
            fetch_delay_seconds=cfg.price.fetch_delay_seconds,
        )
        save_candidates(candidates)
        print(f"Wrote {len(candidates)} candidates to {_CANDIDATES_PATH}")
        return 0

    if command == "report":
        candidates = load_candidates()
        labels = load_labels()
        report = run_report(candidates, labels, cfg)
        _print_report(report)
        return 0

    if command == "ablate":
        result = run_ablation(load_candidates(), load_labels(), cfg)
        _print_ablation(result)
        _ABLATION_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote per-arm metrics and per-row verdicts to {_ABLATION_PATH}")
        return 0

    print(
        f"Unknown command: {command!r}. Use 'build-candidates', 'report' or 'ablate'."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
