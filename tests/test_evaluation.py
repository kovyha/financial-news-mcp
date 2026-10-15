"""Tests for financial_news.evaluation."""

import json
import sys
from datetime import date
from unittest.mock import MagicMock

import anthropic
import pytest

from financial_news import evaluation
from financial_news.enrichment import EnrichmentConfig

# ── precision_recall_f1 ─────────────────────────────────────────────────────


def test_precision_recall_f1_empty_inputs():
    result = evaluation.precision_recall_f1([], [])
    assert result == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "n": 0,
    }


def test_precision_recall_f1_perfect_prediction():
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]
    result = evaluation.precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_precision_recall_f1_all_wrong():
    y_true = [True, True, False, False]
    y_pred = [False, False, True, True]
    result = evaluation.precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["accuracy"] == 0.0


def test_precision_recall_f1_all_positive_predictions():
    y_true = [True, False]
    y_pred = [True, True]
    result = evaluation.precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["tp"] == 1
    assert result["fp"] == 1


def test_precision_recall_f1_no_positive_predictions_recall_zero():
    y_true = [True, True]
    y_pred = [False, False]
    result = evaluation.precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_precision_recall_f1_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        evaluation.precision_recall_f1([True], [True, False])


# ── mcnemar_exact ─────────────────────────────────────────────────────────────


def _paired(only_a: int, only_b: int, both: int = 3) -> tuple[list, list, list]:
    """Build paired predictions with the given discordant counts (all truths
    True, so 'correct' == predicted True)."""
    y_true = [True] * (only_a + only_b + both)
    pred_a = [True] * only_a + [False] * only_b + [True] * both
    pred_b = [False] * only_a + [True] * only_b + [True] * both
    return y_true, pred_a, pred_b


@pytest.mark.parametrize(
    "only_a, only_b, expected_p",
    [
        (4, 5, 1.0),
        (0, 2, 0.5),
        (0, 0, 1.0),
        (0, 6, 2 / 64),
    ],
)
def test_mcnemar_exact(only_a, only_b, expected_p):
    result = evaluation.mcnemar_exact(*_paired(only_a, only_b))
    assert result["only_a_correct"] == only_a
    assert result["only_b_correct"] == only_b
    assert result["p_value"] == pytest.approx(expected_p)


def test_mcnemar_exact_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        evaluation.mcnemar_exact([True], [True], [True, False])


# ── Label ─────────────────────────────────────────────────────────────────────


def test_label_accepts_valid_values():
    for value in ("news_driven", "noise", "ambiguous"):
        evaluation.Label(ticker="NVDA", date="2026-01-15", label=value)


def test_label_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid label"):
        evaluation.Label(ticker="NVDA", date="2026-01-15", label="maybe")


# ── labels I/O ────────────────────────────────────────────────────────────────


def test_save_and_load_labels_round_trip(tmp_path):
    path = tmp_path / "labels.json"
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="news_driven", notes="earnings"
        ),
        "TSLA:2026-01-16": evaluation.Label(
            ticker="TSLA", date="2026-01-16", label="noise"
        ),
    }
    evaluation.save_labels(labels, path)
    loaded = evaluation.load_labels(path)
    assert loaded == labels


def test_load_labels_missing_file_returns_empty_dict(tmp_path):
    assert evaluation.load_labels(tmp_path / "missing.json") == {}


# ── candidates I/O ───────────────────────────────────────────────────────────


def _make_candidate(
    ticker: str = "NVDA",
    day: str = "2026-01-15",
    return_pct: float = 0.05,
    price_z: float = 3.0,
    price_classification: str = "unusual",
    news_z: float = 1.0,
    news_classification: str = "normal",
    articles: list[dict] | None = None,
) -> dict:
    articles = (
        articles
        if articles is not None
        else [{"headline": "Big news", "summary": "details", "time": "09:31:00"}]
    )
    return {
        "ticker": ticker,
        "date": day,
        "price": {
            "return_pct": return_pct,
            "z_score": price_z,
            "classification": price_classification,
        },
        "news": {
            "z_score": news_z,
            "classification": news_classification,
            "recent_articles": articles,
        },
    }


def test_save_and_load_candidates_round_trip(tmp_path):
    path = tmp_path / "candidates.json"
    candidates = [_make_candidate(), _make_candidate(ticker="TSLA", day="2026-01-16")]
    evaluation.save_candidates(candidates, path)
    assert evaluation.load_candidates(path) == candidates


def test_load_candidates_missing_file_returns_empty_list(tmp_path):
    assert evaluation.load_candidates(tmp_path / "missing.json") == []


# ── draft_label_inputs / unlabeled_candidates ───────────────────────────────


def test_draft_label_inputs_excludes_zscore_and_classification_fields():
    candidate = _make_candidate()
    inputs = evaluation.draft_label_inputs(candidate)
    assert inputs.keys() == {"ticker", "date", "return_pct", "headlines"}
    assert inputs["return_pct"] == candidate["price"]["return_pct"]
    assert inputs["headlines"] == [{"headline": "Big news", "summary": "details"}]


def test_unlabeled_candidates_filters_out_labeled_ones():
    candidates = [_make_candidate(ticker="NVDA"), _make_candidate(ticker="TSLA")]
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="noise"
        )
    }
    remaining = evaluation.unlabeled_candidates(candidates, labels)
    assert [c["ticker"] for c in remaining] == ["TSLA"]


# ── predict_zscore_only ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "z_score,threshold,expected",
    [(3.0, 3.0, True), (2.9, 3.0, False), (-3.5, 3.0, True)],
)
def test_predict_zscore_only(z_score, threshold, expected):
    candidate = _make_candidate(price_z=z_score)
    assert evaluation.predict_zscore_only(candidate, threshold) is expected


# ── predict_news_zscore_only ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "news_z,threshold,expected",
    [(3.0, 3.0, True), (2.9, 3.0, False), (-3.5, 3.0, False)],
)
def test_predict_news_zscore_only(news_z, threshold, expected):
    candidate = _make_candidate(news_z=news_z)
    assert evaluation.predict_news_zscore_only(candidate, threshold) is expected


# ── _headline_mentions_company / predict_name_mention ───────────────────────


@pytest.mark.parametrize(
    "ticker,headline,expected",
    [
        ("NVDA", "Why Nvidia (NVDA) Stock Is Soaring Today", True),
        ("NVDA", "Tech stocks lead market bloodbath", False),
        ("F", "Ford Falls Most in 15 Months", True),
        ("F", "Tesla Just Recalled Nearly 219,000 Cars", False),
        (
            "F",
            "Ford Recalls 42,784 Mustang Mach-E Vehicles Over Rear Differential",
            True,
        ),
        ("PLUG", "Fuel Cell Stocks Erupt: Bloom Energy Surges 23%", False),
        ("PLUG", "Plug Power Jumps 10% on Contract Win", True),
    ],
)
def test_headline_mentions_company(ticker, headline, expected):
    assert evaluation._headline_mentions_company(headline, ticker) is expected


def test_predict_name_mention_true_when_fraction_meets_threshold():
    candidate = _make_candidate(
        ticker="AMD",
        articles=[
            {"headline": "Why AMD Stock Popped Today", "summary": ""},
            {"headline": "AMD Beats Q1 Estimates", "summary": ""},
            {"headline": "Tech stocks lead market bloodbath", "summary": ""},
        ],
    )
    assert evaluation.predict_name_mention(candidate, 0.5) is True


def test_predict_name_mention_false_when_fraction_below_threshold():
    candidate = _make_candidate(
        ticker="AMD",
        articles=[
            {"headline": "Why AMD Stock Popped Today", "summary": ""},
            {"headline": "Tech stocks lead market bloodbath", "summary": ""},
            {
                "headline": "CNBC Daily Open: A ceasefire, but no Strait talk",
                "summary": "",
            },
        ],
    )
    assert evaluation.predict_name_mention(candidate, 0.5) is False


def test_predict_name_mention_no_articles_returns_false():
    candidate = _make_candidate(articles=[])
    assert evaluation.predict_name_mention(candidate, 0.25) is False


# ── predict_full_system ──────────────────────────────────────────────────────


def _enrich_cfg() -> EnrichmentConfig:
    return EnrichmentConfig(
        model_name="model",
        valid_labels=frozenset({"positive", "negative", "neutral"}),
        confidence_threshold=0.85,
        min_articles=1,
        max_articles=5,
    )


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def test_predict_full_system_parses_news_driven_verdict(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "score_headlines",
        lambda articles, *_: [
            {"headline": "h", "summary": "", "label": "positive", "score": 0.9}
        ],
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response(
        "VERDICT: news_driven | REASON: earnings beat"
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    is_real, text = evaluation.predict_full_system(_make_candidate(), _enrich_cfg())
    assert is_real is True
    assert "earnings beat" in text


def test_predict_full_system_parses_noise_verdict(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "score_headlines",
        lambda articles, *_: [
            {"headline": "h", "summary": "", "label": "neutral", "score": 0.5}
        ],
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response(
        "VERDICT: noise | REASON: thin liquidity"
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    is_real, _ = evaluation.predict_full_system(_make_candidate(), _enrich_cfg())
    assert is_real is False


def test_predict_full_system_unparseable_response_defaults_to_noise(monkeypatch):
    monkeypatch.setattr(evaluation, "score_headlines", lambda articles, *_: [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response("not a valid verdict")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    is_real, _ = evaluation.predict_full_system(
        _make_candidate(articles=[]), _enrich_cfg()
    )
    assert is_real is False


def test_predict_full_system_without_finbert_skips_scoring_and_tags(monkeypatch):
    def _fail(*_):
        raise AssertionError("score_headlines must not run when use_finbert=False")

    monkeypatch.setattr(evaluation, "score_headlines", _fail)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response(
        "VERDICT: news_driven | REASON: guidance cut"
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)
    articles = [
        {"headline": f"Headline {i}", "summary": "", "time": "09:31:00"}
        for i in range(7)
    ]

    is_real, _ = evaluation.predict_full_system(
        _make_candidate(articles=articles), _enrich_cfg(), use_finbert=False
    )

    assert is_real is True
    prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    # max_articles=5: most recent five kept, in order, with no sentiment tags
    assert "- Headline 0\n- Headline 1" in prompt
    assert "Headline 4" in prompt
    assert "Headline 5" not in prompt
    assert "[" not in prompt.split("Headlines:")[1]


def test_predict_full_system_default_prompt_includes_both_zscores(monkeypatch):
    monkeypatch.setattr(evaluation, "score_headlines", lambda articles, *_: [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response("VERDICT: noise")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    evaluation.predict_full_system(
        _make_candidate(articles=[], news_z=2.5, news_classification="elevated"),
        _enrich_cfg(),
    )

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["messages"][0]["content"] == (
        "Ticker: NVDA\nDate: 2026-01-15\n"
        "Price move: +5.00% (z=3.00, unusual)\n"
        "News volume: z=2.50 (elevated)\n"
        "Headlines:\n(no headlines that day)"
    )
    assert "news-volume z-score" in kwargs["system"]


def test_predict_full_system_without_news_zscore_keeps_price_zscore(monkeypatch):
    monkeypatch.setattr(evaluation, "score_headlines", lambda articles, *_: [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_response("VERDICT: noise")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    evaluation.predict_full_system(
        _make_candidate(articles=[], news_z=2.5),
        _enrich_cfg(),
        include_news_zscore=False,
    )

    kwargs = mock_client.messages.create.call_args.kwargs
    prompt = kwargs["messages"][0]["content"]
    assert "News volume" not in prompt
    assert "z=2.50" not in prompt
    assert "Price move: +5.00% (z=3.00, unusual)" in prompt
    assert "news-volume" not in kwargs["system"]
    assert "price move and its z-score" in kwargs["system"]


def test_judge_system_prompt_default_is_pinned():
    """Pinned verbatim: the stored Z+F+L and Z+L verdicts that run_ablation
    reuses (eval/baseline3_reasoning.json, eval/finbert_ablation.json) were
    produced with exactly this prompt. Changing it means re-running those arms.
    """
    assert evaluation._judge_system_prompt() == (
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


def test_judge_system_prompt_only_differs_by_news_clause():
    with_news = evaluation._judge_system_prompt()
    without_news = evaluation._judge_system_prompt(include_news_zscore=False)
    assert with_news.replace(evaluation._NEWS_ZSCORE_CLAUSE, "") == without_news


# ── predict_news_zscore_with_sentiment (Z+F ablation arm) ────────────────────


def _scored(label: str, score: float) -> list[dict]:
    return [{"headline": "h", "summary": "", "label": label, "score": score}]


@pytest.mark.parametrize(
    "news_z, scored, expected",
    [
        (1.0, _scored("positive", 0.95), False),  # below threshold
        (2.5, _scored("neutral", 0.70), False),  # only neutral selected
        (2.5, _scored("negative", 0.60), False),  # non-neutral but low confidence
        (2.5, _scored("unavailable", 0.0), False),  # finBERT not installed
        (2.5, _scored("negative", 0.90), True),
        (2.5, _scored("positive", 0.85), True),  # exactly at confidence threshold
    ],
)
def test_predict_news_zscore_with_sentiment(monkeypatch, news_z, scored, expected):
    monkeypatch.setattr(evaluation, "score_headlines", lambda *_: scored)
    candidate = _make_candidate(news_z=news_z)
    assert (
        evaluation.predict_news_zscore_with_sentiment(candidate, 2.0, _enrich_cfg())
        is expected
    )


def test_predict_news_zscore_with_sentiment_no_articles_is_false(monkeypatch):
    def _fail(*_):
        raise AssertionError("score_headlines must not run with no articles")

    monkeypatch.setattr(evaluation, "score_headlines", _fail)
    candidate = _make_candidate(news_z=5.0, articles=[])
    assert not evaluation.predict_news_zscore_with_sentiment(
        candidate, 2.0, _enrich_cfg()
    )


# ── build_candidate_pool ─────────────────────────────────────────────────────


def test_build_candidate_pool_filters_to_elevated_and_unusual(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_trading_days", lambda start, end: [date(2026, 1, 15)]
    )

    def fake_price_stats(symbol, as_of_date):
        classification = "unusual" if symbol == "NVDA" else "normal"
        return {"classification": classification, "z_score": 3.0, "return_pct": 0.05}

    def fake_volume_stats(symbol, as_of_date):
        return {"classification": "normal", "z_score": 0.5, "recent_articles": []}

    monkeypatch.setattr(evaluation, "compute_price_stats", fake_price_stats)
    monkeypatch.setattr(evaluation, "compute_volume_stats", fake_volume_stats)

    candidates = evaluation.build_candidate_pool(
        ["NVDA", "TSLA"], date(2026, 1, 15), date(2026, 1, 15)
    )
    assert [c["ticker"] for c in candidates] == ["NVDA"]


def test_build_candidate_pool_skips_ticker_on_price_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_trading_days", lambda start, end: [date(2026, 1, 15)]
    )

    def raise_error(symbol, as_of_date):
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(evaluation, "compute_price_stats", raise_error)

    candidates = evaluation.build_candidate_pool(
        ["NVDA"], date(2026, 1, 15), date(2026, 1, 15)
    )
    assert candidates == []


def test_build_candidate_pool_skips_day_on_news_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_trading_days", lambda start, end: [date(2026, 1, 15)]
    )
    monkeypatch.setattr(
        evaluation,
        "compute_price_stats",
        lambda symbol, as_of_date: {
            "classification": "unusual",
            "z_score": 3.0,
            "return_pct": 0.05,
        },
    )

    def raise_error(symbol, as_of_date):
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(evaluation, "compute_volume_stats", raise_error)

    candidates = evaluation.build_candidate_pool(
        ["NVDA"], date(2026, 1, 15), date(2026, 1, 15)
    )
    assert candidates == []


def test_build_candidate_pool_sleeps_between_calls_when_delay_configured(monkeypatch):
    monkeypatch.setattr(
        evaluation, "_trading_days", lambda start, end: [date(2026, 1, 15)]
    )
    monkeypatch.setattr(
        evaluation,
        "compute_price_stats",
        lambda symbol, as_of_date: {
            "classification": "normal",
            "z_score": 0.1,
            "return_pct": 0.001,
        },
    )
    sleep_calls = []
    monkeypatch.setattr(evaluation.time, "sleep", lambda s: sleep_calls.append(s))

    evaluation.build_candidate_pool(
        ["NVDA"], date(2026, 1, 15), date(2026, 1, 15), fetch_delay_seconds=1.5
    )
    assert sleep_calls == [1.5]


def test_trading_days_excludes_weekends():
    days = evaluation._trading_days(date(2026, 1, 12), date(2026, 1, 18))
    assert date(2026, 1, 17) not in days  # Saturday
    assert date(2026, 1, 18) not in days  # Sunday
    assert date(2026, 1, 12) in days  # Monday


# ── run_report / _labeled_rows ───────────────────────────────────────────────


def _fake_config():
    from financial_news.config import (
        AnalysisConfig,
        BriefingConfig,
        Config,
        MonitorConfig,
        PriceConfig,
        SentimentConfig,
    )

    return Config(
        analysis=AnalysisConfig(),
        price=PriceConfig(),
        monitor=MonitorConfig(tickers=["NVDA"]),
        briefing=BriefingConfig(),
        sentiment=SentimentConfig(),
    )


def test_labeled_rows_excludes_unlabeled_and_ambiguous():
    candidates = [
        _make_candidate(ticker="NVDA", day="2026-01-15"),
        _make_candidate(ticker="TSLA", day="2026-01-16"),
        _make_candidate(ticker="AMD", day="2026-01-17"),
    ]
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="news_driven"
        ),
        "TSLA:2026-01-16": evaluation.Label(
            ticker="TSLA", date="2026-01-16", label="ambiguous"
        ),
    }
    rows = evaluation._labeled_rows(candidates, labels)
    assert [c["ticker"] for c, _ in rows] == ["NVDA"]


def test_run_report_raises_when_no_confirmed_labels():
    with pytest.raises(ValueError, match="No confirmed"):
        evaluation.run_report([_make_candidate()], {}, _fake_config())


def test_run_report_combines_all_baselines(monkeypatch):
    candidates = [_make_candidate(ticker="NVDA", day="2026-01-15", price_z=3.5)]
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="news_driven"
        )
    }
    monkeypatch.setattr(evaluation, "predict_zscore_only", lambda c, t: True)
    monkeypatch.setattr(evaluation, "predict_news_zscore_only", lambda c, t: True)
    monkeypatch.setattr(evaluation, "predict_name_mention", lambda c, t: True)
    monkeypatch.setattr(
        evaluation, "predict_full_system", lambda c, cfg: (True, "reasoning text")
    )

    report = evaluation.run_report(candidates, labels, _fake_config())

    assert report["n_labeled"] == 1
    assert (
        report["news_zscore_only"][_fake_config().analysis.threshold_elevated][
            "precision"
        ]
        == 1.0
    )
    assert report["name_mention"][0.25]["precision"] == 1.0
    assert report["full_system"]["precision"] == 1.0
    assert report["full_system_reasoning"] == ["reasoning text"]


def test_run_report_price_zscore_sensitivity_excludes_threshold_elevated(
    monkeypatch,
):
    """threshold_elevated is the pool's own selection cut — every candidate
    already clears it, so evaluating there would be circular. Only
    threshold_unusual should appear by default.
    """
    candidates = [_make_candidate(ticker="NVDA", day="2026-01-15", price_z=3.5)]
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="news_driven"
        )
    }
    cfg = _fake_config()
    monkeypatch.setattr(
        evaluation, "predict_full_system", lambda c, cfg: (True, "reasoning text")
    )

    report = evaluation.run_report(candidates, labels, cfg)

    assert set(report["price_zscore_sensitivity"]) == {cfg.price.threshold_unusual}


# ── _print_report / main ─────────────────────────────────────────────────────


def _sample_report() -> dict:
    metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "accuracy": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "n": 1,
    }
    return {
        "n_labeled": 1,
        "price_zscore_sensitivity": {3.0: metrics},
        "news_zscore_only": {2.0: metrics},
        "name_mention": {0.5: metrics},
        "full_system": metrics,
        "full_system_reasoning": ["reasoning text"],
    }


def test_print_report_includes_all_baselines(capsys):
    evaluation._print_report(_sample_report())
    out = capsys.readouterr().out
    assert "BASELINE COMPARISON" in out
    assert "news-volume z-score only" in out
    assert "name-mention fraction only" in out
    assert "Full system" in out
    assert "Sensitivity check" in out
    assert "NOT an independent baseline" in out


# ── run_ablation / _print_ablation ───────────────────────────────────────────


def _ablation_fixture(tmp_path):
    candidates = [
        _make_candidate(ticker="NVDA", day="2026-01-15", news_z=3.5),
        _make_candidate(ticker="AMD", day="2026-01-16", news_z=0.5),
    ]
    labels = {
        "NVDA:2026-01-15": evaluation.Label(
            ticker="NVDA", date="2026-01-15", label="news_driven"
        ),
        "AMD:2026-01-16": evaluation.Label(
            ticker="AMD", date="2026-01-16", label="noise"
        ),
    }
    full_path = tmp_path / "baseline3_reasoning.json"
    full_path.write_text(
        json.dumps(
            [
                {
                    "ticker": "NVDA",
                    "date": "2026-01-15",
                    "predicted": "news_driven",
                    "reasoning": "full NVDA",
                },
                {
                    "ticker": "AMD",
                    "date": "2026-01-16",
                    "predicted": "noise",
                    "reasoning": "full AMD",
                },
            ]
        )
    )
    finbert_path = tmp_path / "finbert_ablation.json"
    finbert_path.write_text(
        json.dumps(
            [
                {
                    "ticker": t,
                    "date": d,
                    "without_finbert": "news_driven",
                    "without_finbert_reasoning": f"z+l {t}",
                }
                for t, d in (("NVDA", "2026-01-15"), ("AMD", "2026-01-16"))
            ]
        )
    )
    return candidates, labels, full_path, finbert_path


def test_run_ablation_assembles_all_arms(monkeypatch, tmp_path):
    candidates, labels, full_path, finbert_path = _ablation_fixture(tmp_path)
    calls = []

    def _fake_predict(c, cfg, use_finbert=True, include_news_zscore=True):
        calls.append((c["ticker"], use_finbert, include_news_zscore))
        # F+L right on both; L alone calls everything noise
        return (use_finbert and c["ticker"] == "NVDA"), f"{use_finbert} {c['ticker']}"

    monkeypatch.setattr(evaluation, "predict_full_system", _fake_predict)
    monkeypatch.setattr(
        evaluation, "score_headlines", lambda *_: _scored("positive", 0.95)
    )

    result = evaluation.run_ablation(
        candidates, labels, _fake_config(), full_path, finbert_path
    )

    # Stored arms are never re-run; new LLM arms always drop the news z-score.
    assert sorted(calls) == sorted(
        [(t, fb, False) for t in ("NVDA", "AMD") for fb in (True, False)]
    )
    arms = result["arms"]
    assert list(arms) == [
        "Z+F+L",
        "Z+L",
        "F+L",
        "L",
        "Z+F (θ=2.0)",
        "Z (θ=2.0)",
        "Z+F (θ=3.0)",
        "Z (θ=3.0)",
    ]
    assert arms["Z+F+L"]["vs_full"] is None
    assert arms["Z+F+L"]["metrics"]["accuracy"] == 1.0
    assert arms["Z+L"]["metrics"]["accuracy"] == 0.5
    assert arms["Z+L"]["vs_full"]["only_b_correct"] == 1
    assert arms["L"]["metrics"]["recall"] == 0.0
    assert arms["Z+F (θ=3.0)"]["metrics"]["accuracy"] == 1.0
    assert result["rows"][1] == {
        "ticker": "AMD",
        "date": "2026-01-16",
        "true_label": "noise",
        "verdicts": {
            "Z+F+L": "noise",
            "Z+L": "news_driven",
            "F+L": "noise",
            "L": "noise",
            "Z+F (θ=2.0)": "noise",
            "Z (θ=2.0)": "noise",
            "Z+F (θ=3.0)": "noise",
            "Z (θ=3.0)": "noise",
        },
        "reasoning": {
            "Z+F+L": "full AMD",
            "Z+L": "z+l AMD",
            "F+L": "True AMD",
            "L": "False AMD",
        },
    }


def test_run_ablation_raises_when_stored_verdicts_missing_a_row(monkeypatch, tmp_path):
    candidates, labels, full_path, finbert_path = _ablation_fixture(tmp_path)
    finbert_path.write_text(json.dumps([]))
    monkeypatch.setattr(
        evaluation,
        "predict_full_system",
        lambda *a, **k: pytest.fail("must fail before calling Claude"),
    )

    with pytest.raises(ValueError, match="Stored Z\\+L verdicts are missing"):
        evaluation.run_ablation(
            candidates, labels, _fake_config(), full_path, finbert_path
        )


def test_run_ablation_raises_when_no_confirmed_labels(tmp_path):
    with pytest.raises(ValueError, match="No confirmed"):
        evaluation.run_ablation([], {}, _fake_config(), tmp_path, tmp_path)


def _sample_ablation() -> dict:
    metrics = _sample_report()["full_system"]
    return {
        "n_labeled": 1,
        "arms": {
            "Z+F+L": {"metrics": metrics, "vs_full": None},
            "L": {
                "metrics": metrics,
                "vs_full": {"only_a_correct": 0, "only_b_correct": 2, "p_value": 0.5},
            },
        },
        "rows": [],
    }


def test_print_ablation_renders_arms_and_paired_test(capsys):
    evaluation._print_ablation(_sample_ablation())
    out = capsys.readouterr().out
    assert "COMPONENT ABLATION — n=1" in out
    assert "Z+F+L" in out
    assert "0 / 2, p=0.500" in out


def test_main_ablate_writes_result_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(evaluation, "load_candidates", lambda: [])
    monkeypatch.setattr(evaluation, "load_labels", lambda: {})
    out_path = tmp_path / "ablation.json"
    monkeypatch.setattr(evaluation, "_ABLATION_PATH", out_path)
    monkeypatch.setattr(evaluation, "run_ablation", lambda *a: _sample_ablation())
    monkeypatch.setattr(sys, "argv", ["evaluation.py", "ablate"])

    assert evaluation.main() == 0
    assert json.loads(out_path.read_text()) == _sample_ablation()
    assert "COMPONENT ABLATION" in capsys.readouterr().out


def test_main_build_candidates_writes_and_prints_count(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluation, "build_candidate_pool", lambda *a, **k: [_make_candidate()]
    )
    saved = []
    monkeypatch.setattr(evaluation, "save_candidates", lambda c: saved.append(c))
    monkeypatch.setattr(sys, "argv", ["evaluation.py", "build-candidates"])

    exit_code = evaluation.main()

    assert exit_code == 0
    assert "Wrote 1 candidates" in capsys.readouterr().out
    assert saved == [[_make_candidate()]]


def test_main_report_prints_comparison(monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "load_candidates", lambda: [_make_candidate()])
    monkeypatch.setattr(
        evaluation,
        "load_labels",
        lambda: {
            "NVDA:2026-01-15": evaluation.Label(
                ticker="NVDA", date="2026-01-15", label="news_driven"
            )
        },
    )
    monkeypatch.setattr(evaluation, "run_report", lambda *a, **k: _sample_report())
    monkeypatch.setattr(sys, "argv", ["evaluation.py", "report"])

    exit_code = evaluation.main()

    assert exit_code == 0
    assert "BASELINE COMPARISON" in capsys.readouterr().out


def test_main_unknown_command_returns_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluation.py", "bogus"])

    exit_code = evaluation.main()

    assert exit_code == 1
    assert "Unknown command" in capsys.readouterr().out


def test_main_defaults_to_report_command(monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "load_candidates", lambda: [_make_candidate()])
    monkeypatch.setattr(
        evaluation,
        "load_labels",
        lambda: {
            "NVDA:2026-01-15": evaluation.Label(
                ticker="NVDA", date="2026-01-15", label="news_driven"
            )
        },
    )
    monkeypatch.setattr(evaluation, "run_report", lambda *a, **k: _sample_report())
    monkeypatch.setattr(sys, "argv", ["evaluation.py"])

    exit_code = evaluation.main()

    assert exit_code == 0
    assert "BASELINE COMPARISON" in capsys.readouterr().out
