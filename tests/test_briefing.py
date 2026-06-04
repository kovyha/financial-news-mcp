"""Tests for the daily briefing agent."""

from unittest.mock import MagicMock, patch

import anthropic

from financial_news import briefing, sentiment
from financial_news.config import AnalysisConfig, BriefingConfig, EmailConfig

# ── Mock helpers ──────────────────────────────────────────────────────────────


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(block_id: str, ticker: str) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = block_id
    b.input = {"ticker": ticker}
    return b


def _response(stop_reason: str, content: list) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


def _sample_stats(ticker: str = "NVDA", classification: str = "unusual") -> dict:
    return {
        "ticker": ticker,
        "z_score": 3.5,
        "recent_count": 12,
        "mean": 3.0,
        "std": 1.0,
        "classification": classification,
        "headlines": ["Headline A", "Headline B"],
        "recent_headlines": ["Headline A", "Headline B", "Headline C"],
        "baseline_counts": [3.0] * 30,
    }


_CAP = AnalysisConfig.z_score_cap
_BASELINE = AnalysisConfig.baseline_days
_DAYS = BriefingConfig.headline_days
_MAX = BriefingConfig.max_headlines
_CONF = BriefingConfig.confidence_threshold
_PMIN = BriefingConfig.prompt_headlines_min
_PMAX = BriefingConfig.prompt_headlines_max

# ── _collect_stats ────────────────────────────────────────────────────────────


def test_collect_stats_success(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: _sample_stats(ticker),
    )
    results = briefing._collect_stats(["NVDA", "TSLA"], _CAP)
    assert len(results) == 2
    assert results[0]["ticker"] == "NVDA"
    assert results[0]["z_score"] == 3.5


def test_collect_stats_caps_z_score(monkeypatch):
    def high_z(ticker):
        s = _sample_stats(ticker)
        s["z_score"] = float("inf")
        return s

    monkeypatch.setattr("financial_news.briefing.compute_volume_stats", high_z)
    results = briefing._collect_stats(["NVDA"], _CAP)
    assert results[0]["z_score"] == _CAP


def test_collect_stats_error_path(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    results = briefing._collect_stats(["NVDA"], _CAP)
    assert "error" in results[0]
    assert results[0]["ticker"] == "NVDA"


# ── _enrich_stats_with_sentiment ─────────────────────────────────────────────


_TEST_MODEL = "test-model"
_TEST_LABELS = frozenset({"positive", "negative", "neutral"})


def _mock_score(monkeypatch, label: str = "positive", score: float = 0.90):
    """Patch sentiment.score_headlines to return a fixed label/score per headline."""

    def fake_score(headlines, model_name, valid_labels):
        return [{"headline": h, "label": label, "score": score} for h in headlines]

    monkeypatch.setattr(sentiment, "score_headlines", fake_score)


def _enrich(stats, monkeypatch=None, **kw):
    """Call _enrich_stats_with_sentiment with test defaults."""
    return briefing._enrich_stats_with_sentiment(
        stats,
        kw.get("model_name", _TEST_MODEL),
        kw.get("valid_labels", _TEST_LABELS),
    )


def test_enrich_adds_headline_sentiment(monkeypatch):
    _mock_score(monkeypatch)
    enriched = _enrich([_sample_stats("NVDA")])
    assert "headline_sentiment" in enriched[0]
    assert len(enriched[0]["headline_sentiment"]) > 0


def test_enrich_scores_from_recent_headlines(monkeypatch):
    """recent_headlines is preferred over the top-5 headlines field."""
    scored = []

    def capture_score(headlines, model_name, valid_labels):
        scored.extend(headlines)
        return [{"headline": h, "label": "neutral", "score": 0.5} for h in headlines]

    monkeypatch.setattr(sentiment, "score_headlines", capture_score)
    _enrich([_sample_stats("NVDA")])
    assert "Headline C" in scored


def test_enrich_scores_all_available_headlines(monkeypatch):
    """All available headlines are scored — no cap applied."""
    scored = []

    def capture_score(headlines, model_name, valid_labels):
        scored.extend(headlines)
        return [{"headline": h, "label": "positive", "score": 0.8} for h in headlines]

    monkeypatch.setattr(sentiment, "score_headlines", capture_score)
    s = {**_sample_stats("TSLA"), "recent_headlines": ["H1", "H2", "H3"]}
    _enrich([s])
    assert len(scored) == 3


def test_enrich_skips_error_entries(monkeypatch):
    _mock_score(monkeypatch)
    error_entry = {"ticker": "FAIL", "error": "fetch failed"}
    enriched = _enrich([error_entry])
    assert "headline_sentiment" not in enriched[0]


def test_enrich_zero_news_ticker_skips_scoring(monkeypatch):
    """Zero-news tickers get empty headline_sentiment without calling the scorer."""
    score_called = []

    def capture_score(headlines, model_name, valid_labels):
        score_called.append(headlines)
        return []

    monkeypatch.setattr(sentiment, "score_headlines", capture_score)
    s = {**_sample_stats("PLUG"), "headlines": [], "recent_headlines": []}
    enriched = _enrich([s])
    assert enriched[0]["headline_sentiment"] == []
    assert not score_called, "score_headlines must not be called for zero-news tickers"


def test_enrich_falls_back_to_headlines_when_no_recent(monkeypatch):
    """Falls back to top-5 headlines if recent_headlines is absent."""
    scored = []

    def capture_score(headlines, model_name, valid_labels):
        scored.extend(headlines)
        return [{"headline": h, "label": "neutral", "score": 0.5} for h in headlines]

    monkeypatch.setattr(sentiment, "score_headlines", capture_score)
    s = _sample_stats("AMD")
    del s["recent_headlines"]
    _enrich([s])
    assert "Headline A" in scored


# ── _select_prompt_headlines ──────────────────────────────────────────────────


def _scored(headlines_and_scores: list[tuple[str, float]]) -> list[dict]:
    return [
        {"headline": h, "label": "positive", "score": s}
        for h, s in headlines_and_scores
    ]


def test_select_prompt_headlines_confidence_filter():
    """Headlines at or above threshold are returned."""
    items = _scored([("H-high", 0.90), ("H-mid", 0.80), ("H-low", 0.60)])
    result = briefing._select_prompt_headlines(items, 0.85, 2, 10)
    headlines = [r["headline"] for r in result]
    assert "H-high" in headlines
    assert "H-low" not in headlines


def test_select_prompt_headlines_min_cap():
    """Falls back to top-min headlines when fewer than min meet the threshold."""
    items = _scored([("H1", 0.60), ("H2", 0.55), ("H3", 0.50), ("H4", 0.45)])
    result = briefing._select_prompt_headlines(items, 0.85, 3, 10)
    assert len(result) == 3
    assert result[0]["headline"] == "H1"  # sorted by score desc


def test_select_prompt_headlines_max_cap():
    """No more than max headlines returned even when many exceed threshold."""
    items = _scored([(f"H{i}", 0.90) for i in range(20)])
    result = briefing._select_prompt_headlines(items, 0.85, 5, 10)
    assert len(result) == 10


def test_select_prompt_headlines_sorted_by_score_desc():
    """Returned headlines are ordered highest score first."""
    items = _scored([("Lo", 0.88), ("Hi", 0.95), ("Mid", 0.91)])
    result = briefing._select_prompt_headlines(items, 0.85, 1, 10)
    assert [r["headline"] for r in result] == ["Hi", "Mid", "Lo"]


# ── _format_stats_for_prompt ──────────────────────────────────────────────────


def test_format_stats_normal():
    stats = [_sample_stats("NVDA", "unusual")]
    text = briefing._format_stats_for_prompt(stats, _CONF, _PMIN, _PMAX)
    assert "NVDA" in text
    assert "unusual" in text
    assert "Headline A" in text


def test_format_stats_with_sentiment():
    """Sentiment labels appear in the prompt when headline_sentiment is set."""
    s = _sample_stats("NVDA", "unusual")
    s["headline_sentiment"] = [
        {"headline": "Nvidia beats estimates", "label": "positive", "score": 0.97},
        {"headline": "Supply chain warning", "label": "negative", "score": 0.85},
    ]
    text = briefing._format_stats_for_prompt([s], _CONF, _PMIN, _PMAX)
    assert "[positive 0.97]" in text
    assert "[negative 0.85]" in text
    assert "Nvidia beats estimates" in text


def test_format_stats_falls_back_to_plain_when_all_unavailable():
    """Falls back to plain headline text if all sentiment labels are 'unavailable'."""
    s = _sample_stats("NVDA", "unusual")
    s["headline_sentiment"] = [
        {"headline": "Headline A", "label": "unavailable", "score": 0.0},
    ]
    text = briefing._format_stats_for_prompt([s], _CONF, _PMIN, _PMAX)
    assert "unavailable" not in text
    assert "Headline A" in text


def test_format_stats_no_headlines():
    s = _sample_stats("TSLA", "normal")
    s["headlines"] = []
    text = briefing._format_stats_for_prompt([s], _CONF, _PMIN, _PMAX)
    assert "no headlines today" in text


def test_format_stats_error_entry():
    stats = [{"ticker": "GME", "error": "fetch failed"}]
    text = briefing._format_stats_for_prompt(stats, _CONF, _PMIN, _PMAX)
    assert "GME" in text
    assert "ERROR" in text


# ── _fetch_headlines_for_tool ─────────────────────────────────────────────────


def test_fetch_headlines_success(monkeypatch):
    import time

    mock_news = [
        {
            "headline": "Big news",
            "source": "Reuters",
            "datetime": int(time.time()),
        }
    ]
    monkeypatch.setattr(
        "financial_news.briefing.fetch_news", lambda *a, **kw: mock_news
    )
    result = briefing._fetch_headlines_for_tool("NVDA", _DAYS, _MAX)
    assert "Big news" in result
    assert "Reuters" in result


def test_fetch_headlines_empty(monkeypatch):
    monkeypatch.setattr("financial_news.briefing.fetch_news", lambda *a, **kw: [])
    result = briefing._fetch_headlines_for_tool("NVDA", _DAYS, _MAX)
    assert "No news found" in result


def test_fetch_headlines_exception(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.fetch_news",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("timeout")),
    )
    result = briefing._fetch_headlines_for_tool("NVDA", _DAYS, _MAX)
    assert "Error fetching news" in result


# ── _run_briefing ─────────────────────────────────────────────────────────────


def test_run_briefing_end_turn(monkeypatch):
    resp = _response("end_turn", [_text_block("Watchlist is quiet today.")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "normal")], _BASELINE, _DAYS, _MAX, _CONF, _PMIN, _PMAX
    )
    assert "Watchlist is quiet today." in result


def test_run_briefing_prompt_uses_configured_baseline_days(monkeypatch):
    """baseline_days must appear in the prompt, not the literal '30'."""
    resp = _response("end_turn", [_text_block("ok")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    briefing._run_briefing(
        [_sample_stats()],
        baseline_days=60,
        headline_days=14,
        max_headlines=_MAX,
        confidence_threshold=_CONF,
        prompt_headlines_min=_PMIN,
        prompt_headlines_max=_PMAX,
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    prompt_text = call_kwargs["messages"][0]["content"]
    assert "60-day" in prompt_text
    assert "14-day" in prompt_text


def test_run_briefing_unexpected_stop_reason(monkeypatch):
    resp = _response("max_tokens", [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats()], _BASELINE, _DAYS, _MAX, _CONF, _PMIN, _PMAX
    )
    assert "max_tokens" in result


def test_run_briefing_tool_use_loop(monkeypatch):
    """Claude calls get_news_headlines then returns end_turn."""
    import time

    ts = int(time.time())
    mock_news = [
        {"headline": "NVIDIA H200 launch", "source": "Reuters", "datetime": ts}
    ]
    monkeypatch.setattr(
        "financial_news.briefing.fetch_news", lambda *a, **kw: mock_news
    )

    # First response: tool_use (with a non-tool block to cover the skip branch)
    tool_resp = _response(
        "tool_use",
        [_text_block("thinking..."), _tool_use_block("c1", "NVDA")],
    )
    end_resp = _response(
        "end_turn", [_text_block("NVDA: GPU launch driving activity.")]
    )

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, _MAX, _CONF, _PMIN, _PMAX
    )
    assert "NVDA" in result

    # Verify headline content was passed back as a tool result
    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "NVIDIA H200 launch" in tool_result_content


# ── main ──────────────────────────────────────────────────────────────────────


def test_main(monkeypatch, capsys):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda tickers, z_score_cap: [_sample_stats("NVDA", "normal")],
    )
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: "All quiet.",
    )
    exit_code = briefing.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DAILY BRIEFING" in out
    assert "All quiet." in out


def test_main_uses_snapshot_when_available(monkeypatch, capsys):
    snap_stats = [_sample_stats("NVDA", "normal")]
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: snap_stats)
    collect_called = []
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda *a, **kw: collect_called.append(True) or [],
    )
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: "From snapshot.",
    )
    briefing.main()
    assert not collect_called, (
        "_collect_stats must not be called when snapshot is fresh"
    )
    assert "From snapshot." in capsys.readouterr().out


def test_main_falls_back_to_collect_when_no_snapshot(monkeypatch, capsys):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    collect_called = []
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda tickers, z_score_cap: (
            collect_called.append(True) or [_sample_stats("NVDA")]
        ),
    )
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: "From collect.",
    )
    briefing.main()
    assert collect_called, "_collect_stats must be called when no snapshot is available"
    assert "From collect." in capsys.readouterr().out


# ── email integration ─────────────────────────────────────────────────────────


def _email_cfg() -> EmailConfig:
    return EmailConfig(recipients=["ops@example.com"], smtp_host="smtp.example.com")


def _patch_main(monkeypatch, stats, briefing_text="Analysis."):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats", lambda *a, **kw: stats
    )
    monkeypatch.setattr(
        "financial_news.briefing._enrich_stats_with_sentiment",
        lambda s, *a, **kw: s,
    )
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: briefing_text,
    )


def test_main_calls_send_run_summary_when_email_configured(monkeypatch, capsys):
    stats = [_sample_stats("NVDA", "unusual")]
    _patch_main(monkeypatch, stats)

    fake_cfg = MagicMock()
    fake_cfg.email = _email_cfg()
    fake_cfg.monitor.tickers = ["NVDA"]
    fake_cfg.monitor.snapshot_path = "/tmp/snap.json"

    with (
        patch("financial_news.briefing.load_config", return_value=fake_cfg),
        patch("financial_news.briefing.send_run_summary") as mock_email,
    ):
        briefing.main()

    mock_email.assert_called_once()
    cfg_arg, good_stats_arg, failed_arg, briefing_arg = mock_email.call_args[0]
    assert cfg_arg is fake_cfg.email
    assert len(good_stats_arg) == 1
    assert good_stats_arg[0]["ticker"] == "NVDA"
    assert failed_arg == []
    assert briefing_arg == "Analysis."


def test_main_does_not_call_send_run_summary_when_email_is_none(monkeypatch, capsys):
    stats = [_sample_stats("NVDA", "normal")]
    _patch_main(monkeypatch, stats)

    fake_cfg = MagicMock()
    fake_cfg.email = None
    fake_cfg.monitor.tickers = ["NVDA"]
    fake_cfg.monitor.snapshot_path = "/tmp/snap.json"

    with (
        patch("financial_news.briefing.load_config", return_value=fake_cfg),
        patch("financial_news.briefing.send_run_summary") as mock_email,
    ):
        briefing.main()

    mock_email.assert_not_called()


def test_main_separates_good_and_failed_stats_for_email(monkeypatch, capsys):
    stats = [
        _sample_stats("NVDA", "unusual"),
        {"ticker": "FAIL", "error": "fetch failed"},
    ]
    _patch_main(monkeypatch, stats)

    fake_cfg = MagicMock()
    fake_cfg.email = _email_cfg()
    fake_cfg.monitor.tickers = ["NVDA", "FAIL"]
    fake_cfg.monitor.snapshot_path = "/tmp/snap.json"

    with (
        patch("financial_news.briefing.load_config", return_value=fake_cfg),
        patch("financial_news.briefing.send_run_summary") as mock_email,
    ):
        briefing.main()

    _, good_stats_arg, failed_arg, _ = mock_email.call_args[0]
    assert good_stats_arg == [_sample_stats("NVDA", "unusual")]
    assert failed_arg == ["FAIL"]
