"""Tests for the daily briefing agent."""

from unittest.mock import MagicMock, patch

import anthropic

from financial_news import briefing
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


_BASELINE = AnalysisConfig.baseline_days
_DAYS = BriefingConfig.headline_days
_MAX = BriefingConfig.max_headlines

# ── _collect_stats ────────────────────────────────────────────────────────────


def test_collect_stats_success(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: _sample_stats(ticker),
    )
    results = briefing._collect_stats(["NVDA", "TSLA"])
    assert len(results) == 2
    assert results[0]["ticker"] == "NVDA"
    assert results[0]["z_score"] == 3.5


def test_collect_stats_error_path(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    results = briefing._collect_stats(["NVDA"])
    assert "error" in results[0]
    assert results[0]["ticker"] == "NVDA"


# ── _format_stats_for_prompt ──────────────────────────────────────────────────


def test_format_stats_normal():
    stats = [_sample_stats("NVDA", "unusual")]
    text = briefing._format_stats_for_prompt(stats)
    assert "NVDA" in text
    assert "unusual" in text
    assert "Headline A" in text


def test_format_stats_with_selected_articles():
    """Sentiment labels appear in the prompt when selected_articles is set."""
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        {
            "headline": "Nvidia beats estimates",
            "summary": "",
            "label": "positive",
            "score": 0.97,
        },  # noqa: E501
        {
            "headline": "Supply chain warning",
            "summary": "",
            "label": "negative",
            "score": 0.85,
        },  # noqa: E501
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "[positive 0.97]" in text
    assert "[negative 0.85]" in text
    assert "Nvidia beats estimates" in text


def test_format_stats_shows_summary_when_present():
    """Article summary appears in the prompt block when non-empty."""
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        {
            "headline": "Nvidia beats estimates",
            "summary": "EPS of $5.16 vs $4.59 expected.",
            "label": "positive",
            "score": 0.97,
        },
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "EPS of $5.16 vs $4.59 expected." in text


def test_format_stats_falls_back_to_plain_when_all_unavailable():
    """Falls back to plain headline text if all selected labels are 'unavailable'."""
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        {"headline": "Headline A", "summary": "", "label": "unavailable", "score": 0.0},
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "unavailable" not in text
    assert "Headline A" in text


def test_format_stats_no_headlines():
    s = _sample_stats("TSLA", "normal")
    s["headlines"] = []
    text = briefing._format_stats_for_prompt([s])
    assert "no headlines today" in text


def test_format_stats_error_entry():
    stats = [{"ticker": "GME", "error": "fetch failed"}]
    text = briefing._format_stats_for_prompt(stats)
    assert "GME" in text
    assert "ERROR" in text


# ── _fetch_headlines_for_tool ─────────────────────────────────────────────────


def test_fetch_headlines_success(monkeypatch):
    import time

    mock_news = [
        {
            "headline": "Big news",
            "summary": "Details on the acquisition.",
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
    assert "Details on the acquisition." in result


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
        [_sample_stats("NVDA", "normal")], _BASELINE, _DAYS, _MAX
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

    result = briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, _MAX)
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
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, _MAX
    )
    assert "NVDA" in result

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "NVIDIA H200 launch" in tool_result_content


# ── main ──────────────────────────────────────────────────────────────────────


def test_main(monkeypatch, capsys):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda tickers: [_sample_stats("NVDA", "normal")],
    )
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
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
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
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
        lambda tickers: (
            collect_called.append(True) or [_sample_stats("NVDA")]
        ),
    )
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
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
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
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
