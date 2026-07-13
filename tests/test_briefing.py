"""Tests for the daily briefing agent."""

from unittest.mock import MagicMock, patch

import anthropic

from financial_news import briefing
from financial_news.config import (  # noqa: F401
    AnalysisConfig,
    BriefingConfig,
    EmailConfig,
)

# ── Mock helpers ──────────────────────────────────────────────────────────────


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(
    block_id: str, ticker: str, name: str = "get_news_headlines"
) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = block_id
    b.input = {"ticker": ticker}
    b.name = name
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
_DAYS = AnalysisConfig.headline_days
_MAX = BriefingConfig.max_headlines

# ── _collect_stats ────────────────────────────────────────────────────────────


def _sample_price_stats(symbol: str = "NVDA", classification: str = "normal") -> dict:
    return {
        "symbol": symbol,
        "current_price": 100.0,
        "previous_close": 99.0,
        "return_pct": 0.0101,
        "mean": 0.0,
        "std": 0.01,
        "z_score": 1.0,
        "classification": classification,
        "baseline_days": 30,
    }


def test_collect_stats_success(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: _sample_stats(ticker),
    )
    monkeypatch.setattr(
        "financial_news.briefing.compute_price_stats",
        lambda ticker: _sample_price_stats(ticker),
    )
    results = briefing._collect_stats(["NVDA", "TSLA"])
    assert len(results) == 2
    assert results[0]["ticker"] == "NVDA"
    assert results[0]["z_score"] == 3.5
    assert results[0]["price"]["symbol"] == "NVDA"


def test_collect_stats_error_path(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("API down")),
    )
    results = briefing._collect_stats(["NVDA"])
    assert "error" in results[0]
    assert results[0]["ticker"] == "NVDA"


def test_collect_stats_price_failure_still_returns_news_stats(monkeypatch):
    """A yfinance outage must not discard the ticker's news stats."""
    monkeypatch.setattr(
        "financial_news.briefing.compute_volume_stats",
        lambda ticker: _sample_stats(ticker),
    )
    monkeypatch.setattr(
        "financial_news.briefing.compute_price_stats",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("yfinance down")),
    )
    results = briefing._collect_stats(["NVDA"])
    assert "error" not in results[0]
    assert results[0]["ticker"] == "NVDA"
    assert "price" not in results[0]


# ── _is_flagged / _format_intraday_bars / _build_intraday_cache ──────────────


def test_is_flagged_true_when_news_elevated():
    s = _sample_stats("NVDA", "elevated")
    assert briefing._is_flagged(s)


def test_is_flagged_true_when_price_unusual():
    s = _sample_stats("NVDA", "normal")
    s["price"] = _sample_price_stats("NVDA", "unusual")
    assert briefing._is_flagged(s)


def test_is_flagged_false_when_both_normal():
    s = _sample_stats("NVDA", "normal")
    s["price"] = _sample_price_stats("NVDA", "normal")
    assert not briefing._is_flagged(s)


def test_is_flagged_false_when_price_absent_and_news_normal():
    s = _sample_stats("NVDA", "normal")
    assert not briefing._is_flagged(s)


def test_format_intraday_bars_empty():
    assert briefing._format_intraday_bars([]) == "No intraday price data available."


def test_format_intraday_bars_formats_time_price_and_return():
    bars = [{"time": "09:30:00", "price": 100.5, "return_from_prev_close": 0.032}]
    text = briefing._format_intraday_bars(bars)
    assert "09:30:00" in text
    assert "$100.50" in text
    assert "+3.20%" in text


def test_build_intraday_cache_only_fetches_flagged_tickers(monkeypatch):
    fetch_calls = []

    def fake_fetch(ticker):
        fetch_calls.append(ticker)
        return {
            "date": "2026-07-13",
            "bars": [{"time": "09:30:00", "price": 1.0, "return_from_prev_close": 0.0}],
        }

    monkeypatch.setattr("financial_news.briefing.fetch_intraday_prices", fake_fetch)

    flagged = _sample_stats("NVDA", "unusual")
    normal = _sample_stats("TSLA", "normal")
    normal["price"] = _sample_price_stats("TSLA", "normal")
    error = {"ticker": "FAIL", "error": "fetch failed"}

    cache = briefing._build_intraday_cache([flagged, normal, error])

    assert fetch_calls == ["NVDA"]
    assert "NVDA" in cache
    assert "TSLA" not in cache
    assert "FAIL" not in cache


def test_build_intraday_cache_handles_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        "financial_news.briefing.fetch_intraday_prices",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("yfinance down")),
    )
    flagged = _sample_stats("NVDA", "unusual")

    cache = briefing._build_intraday_cache([flagged])
    assert cache["NVDA"] == "No intraday price data available."


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


def test_format_stats_shows_time_prefix_when_present():
    """Today's article time must be visible so Claude can attribute intraday moves."""
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        {
            "headline": "Nvidia beats estimates",
            "summary": "",
            "label": "positive",
            "score": 0.97,
            "time": "09:35:12",
        },
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "[09:35:12]" in text


def test_format_stats_omits_time_prefix_when_absent():
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        {
            "headline": "Nvidia beats estimates",
            "summary": "",
            "label": "positive",
            "score": 0.97,
        },
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "[positive 0.97] Nvidia beats estimates" in text


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


def test_format_stats_includes_price_move_when_present():
    s = _sample_stats("NVDA", "unusual")
    s["price"] = _sample_price_stats("NVDA", "elevated")
    text = briefing._format_stats_for_prompt([s])
    assert "price_move=+1.01%" in text
    assert "z=1.00" in text
    assert "elevated" in text


def test_format_stats_omits_price_move_when_absent():
    stats = [_sample_stats("NVDA", "normal")]
    text = briefing._format_stats_for_prompt(stats)
    assert "price_move" not in text


def test_system_prompt_instructs_cross_referencing_price_and_news():
    assert "price" in briefing._SYSTEM_PROMPT.lower()
    assert "news-driven" in briefing._SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_get_price_intraday():
    assert "get_price_intraday" in briefing._SYSTEM_PROMPT


def test_format_stats_error_entry():
    stats = [{"ticker": "GME", "error": "fetch failed"}]
    text = briefing._format_stats_for_prompt(stats)
    assert "GME" in text
    assert "ERROR" in text


# ── _format_headline_articles ─────────────────────────────────────────────────


def test_format_headline_articles_success():
    articles = [
        {
            "date": "2024-01-15",
            "headline": "Big news",
            "summary": "Details on the acquisition.",
            "source": "Reuters",
            "label": "positive",
            "score": 0.92,
        }
    ]
    result = briefing._format_headline_articles(articles, _MAX)
    assert "Big news" in result
    assert "Reuters" in result
    assert "Details on the acquisition." in result
    assert "2024-01-15" in result
    assert "[positive 0.92]" in result


def test_format_headline_articles_omits_unavailable_label():
    articles = [
        {
            "date": "2024-01-15",
            "headline": "Big news",
            "summary": "",
            "source": "Reuters",
            "label": "unavailable",
            "score": 0.0,
        }
    ]
    result = briefing._format_headline_articles(articles, _MAX)
    assert "unavailable" not in result
    assert "Big news" in result


def test_format_headline_articles_omits_label_when_absent():
    articles = [
        {"date": "2024-01-15", "headline": "Big news", "summary": "", "source": "AP"}
    ]
    result = briefing._format_headline_articles(articles, _MAX)
    assert "[" not in result.split("]", 1)[1] if "]" in result else True
    assert "Big news" in result


def test_format_headline_articles_empty():
    result = briefing._format_headline_articles([], _MAX)
    assert "No headline context available." in result


def test_format_headline_articles_truncates_to_max():
    articles = [
        {
            "date": "2024-01-15",
            "headline": f"Headline {i}",
            "summary": "",
            "source": "x",
        }
        for i in range(10)
    ]
    result = briefing._format_headline_articles(articles, max_headlines=3)
    assert "Headline 0" in result
    assert "Headline 2" in result
    assert "Headline 3" not in result


# ── _run_briefing ─────────────────────────────────────────────────────────────


def test_run_briefing_end_turn(monkeypatch):
    resp = _response("end_turn", [_text_block("Watchlist is quiet today.")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "normal")], _BASELINE, _DAYS, {}, {}
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
        headlines_cache={},
        intraday_cache={},
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

    result = briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {})
    assert "max_tokens" in result


def test_run_briefing_tool_use_loop(monkeypatch):
    """Claude calls get_news_headlines then returns end_turn."""
    headlines_cache = {
        "NVDA": "- [2024-06-01] NVIDIA H200 launch (Reuters)\n  New GPU announced."
    }

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
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, headlines_cache, {}
    )
    assert "NVDA" in result

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "NVIDIA H200 launch" in tool_result_content


def test_run_briefing_tool_use_loop_price_intraday(monkeypatch):
    """Claude calls get_price_intraday and gets the intraday cache, not headlines."""
    intraday_cache = {"NVDA": "- 09:30:00: $450.00 (+3.20% vs previous close)"}

    tool_resp = _response(
        "tool_use",
        [
            _text_block("checking intraday..."),
            _tool_use_block("c1", "NVDA", name="get_price_intraday"),
        ],
    )
    end_resp = _response("end_turn", [_text_block("NVDA: gapped up at the open.")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, {}, intraday_cache
    )
    assert "NVDA" in result

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "09:30:00" in tool_result_content
    assert tool_result_content == intraday_cache["NVDA"]


def test_run_briefing_tools_list_includes_price_intraday(monkeypatch):
    """The Claude tool schema must offer get_price_intraday, not just headlines."""
    resp = _response("end_turn", [_text_block("ok")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {})

    tool_names = {t["name"] for t in mock_client.messages.create.call_args[1]["tools"]}
    assert tool_names == {"get_news_headlines", "get_price_intraday"}


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
        lambda tickers: collect_called.append(True) or [_sample_stats("NVDA")],
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
