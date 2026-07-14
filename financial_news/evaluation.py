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
whether it's genuinely about the ticker or explains the move — an axis orthogonal
to this task. It showed ~zero separation between news_driven and noise candidates
even conditioned on name-mention (see docs/evaluation.md).

See docs/evaluation.md for methodology and ground-truth caveats (small-N,
human-labeled by the author reading only price move + headlines).

Run via:
    FINNHUB_API_KEY=<key> uv run python -m financial_news.evaluation build-candidates
    FINNHUB_API_KEY=<key> ANTHROPIC_API_KEY=<key> \\
        uv run python -m financial_news.evaluation report
"""

import json
import logging
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

_LABEL_VALUES = frozenset({"news_driven", "noise", "ambiguous"})

_MODEL = "claude-opus-4-7"
_MAX_TOKENS = 300
_SYSTEM_PROMPT_JUDGE = (
    "You are auditing a single flagged price move for a financial monitoring"
    " system. You are given the price move and its z-score vs a volatility"
    " baseline, the news-volume z-score for the same day vs a 30-day baseline,"
    " and that day's headlines (with finBERT sentiment labels where available)."
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


def predict_full_system(
    candidate: dict, enrich_cfg: EnrichmentConfig
) -> tuple[bool, str]:
    """Baseline 3: finBERT-selected headlines + both z-scores fed to Claude for a
    single structured news-driven/noise judgment. Returns (is_news_driven, raw
    response text).
    """
    articles = candidate["news"]["recent_articles"]
    scored = (
        score_headlines(articles, enrich_cfg.model_name, enrich_cfg.valid_labels)
        if articles
        else []
    )
    selected = (
        select_articles(
            scored,
            candidate["price"]["classification"],
            enrich_cfg.confidence_threshold,
            enrich_cfg.min_articles,
            enrich_cfg.max_articles,
        )
        if scored
        else []
    )

    headline_block = (
        "\n".join(
            f"- [{item['label']} {item['score']:.2f}] {item['headline']}"
            + (f"\n  {item['summary']}" if item.get("summary") else "")
            for item in selected
        )
        or "(no headlines that day)"
    )

    prompt = (
        f"Ticker: {candidate['ticker']}\nDate: {candidate['date']}\n"
        f"Price move: {candidate['price']['return_pct']:+.2%}"
        f" (z={candidate['price']['z_score']:.2f},"
        f" {candidate['price']['classification']})\n"
        f"News volume: z={candidate['news']['z_score']:.2f}"
        f" ({candidate['news']['classification']})\n"
        f"Headlines:\n{headline_block}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT_JUDGE,
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

    valid_labels = frozenset(cfg.sentiment.labels)
    enrich_cfg = EnrichmentConfig(
        model_name=cfg.sentiment.model_name,
        valid_labels=valid_labels,
        confidence_threshold=cfg.briefing.confidence_threshold,
        min_articles=cfg.briefing.prompt_headlines_min,
        max_articles=cfg.briefing.prompt_headlines_max,
    )
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

    print(f"Unknown command: {command!r}. Use 'build-candidates' or 'report'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
