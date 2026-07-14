"""Tests for financial_news.evaluation."""

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
        "zscore_only": {3.0: metrics},
        "news_zscore_only": {2.0: metrics},
        "name_mention": {0.5: metrics},
        "full_system": metrics,
        "full_system_reasoning": ["reasoning text"],
    }


def test_print_report_includes_all_baselines(capsys):
    evaluation._print_report(_sample_report())
    out = capsys.readouterr().out
    assert "BASELINE COMPARISON" in out
    assert "price z-score only" in out
    assert "news-volume z-score only" in out
    assert "name-mention fraction only" in out
    assert "Full system" in out


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
