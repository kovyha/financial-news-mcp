"""Tests for the daily briefing agent."""

from unittest.mock import MagicMock, patch

import anthropic
import pytest
from claude_mocks import response as _response
from claude_mocks import text_block as _text_block
from claude_mocks import tool_use_block

from financial_news import briefing
from financial_news.config import (  # noqa: F401
    AnalysisConfig,
    BriefingConfig,
    EmailConfig,
    LLMConfig,
)
from financial_news.llm import ToolLoopResult


@pytest.fixture(autouse=True)
def _no_grafana(monkeypatch):
    """Keep main() from exporting to a real Grafana Cloud set in the shell."""
    monkeypatch.delenv("GRAFANA_CLOUD_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", raising=False)


# ── Mock helpers ──────────────────────────────────────────────────────────────


def _loop_result(text: str, cost_usd: float | None = 0.05) -> ToolLoopResult:
    usage = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 3000,
        "cache_read_input_tokens": 6000,
        "output_tokens": 500,
    }
    return ToolLoopResult(text, 3, usage, cost_usd)


def _tool_use_block(
    block_id: str, ticker: str, name: str = "get_news_headlines"
) -> MagicMock:
    return tool_use_block(block_id, name, {"ticker": ticker})


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


def _article(headline: str, label: str, score: float, summary: str = "") -> dict:
    return {"headline": headline, "summary": summary, "label": label, "score": score}


def test_format_stats_quiet_ticker_lists_top_headlines_non_neutral_first():
    s = _sample_stats("XOM", "normal")
    s["selected_articles"] = [
        _article("Dividend declared", "neutral", 0.97, "Board declared $0.99."),
        _article("Oil steady", "neutral", 0.95),
        _article("Shareholder meeting set", "neutral", 0.93),
        _article("Price target raised", "positive", 0.90, "Analyst lifts target."),
        _article("Refinery outage", "negative", 0.88),
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "Price target raised" in text
    assert "Refinery outage" in text
    assert "Dividend declared" in text
    assert "Oil steady" not in text
    assert "Shareholder meeting set" not in text
    assert "Analyst lifts target." not in text
    assert "Board declared $0.99." not in text
    assert text.index("Price target raised") < text.index("Refinery outage")
    assert text.index("Refinery outage") < text.index("Dividend declared")


def test_format_stats_quiet_ticker_ranks_unavailable_with_neutral():
    s = _sample_stats("XOM", "normal")
    s["selected_articles"] = [
        _article("Dividend declared", "neutral", 0.97),
        _article("Oil steady", "neutral", 0.95),
        _article("Garbled label", "unavailable", 0.0),
        _article("Price target raised", "positive", 0.90),
    ]
    text = briefing._format_stats_for_prompt([s])
    assert "Price target raised" in text
    assert "Garbled label" not in text


def test_format_stats_flagged_ticker_lists_every_selected_article_with_summary():
    s = _sample_stats("NVDA", "unusual")
    s["selected_articles"] = [
        _article(f"Story {i}", "neutral", 0.9, f"Summary {i}.") for i in range(5)
    ]
    text = briefing._format_stats_for_prompt([s])
    for i in range(5):
        assert f"Story {i}" in text
        assert f"Summary {i}." in text


def test_format_stats_price_flagged_ticker_counts_as_flagged():
    s = _sample_stats("NVDA", "normal")
    s["price"] = _sample_price_stats("NVDA", "unusual")
    s["selected_articles"] = [_article(f"Story {i}", "neutral", 0.9) for i in range(5)]
    text = briefing._format_stats_for_prompt([s])
    assert "Story 4" in text


def test_format_stats_quiet_ticker_plain_headlines_capped():
    s = _sample_stats("XOM", "normal")
    s["headlines"] = ["H1", "H2", "H3", "H4", "H5"]
    text = briefing._format_stats_for_prompt([s])
    assert "- H3" in text
    assert "H4" not in text


def test_run_briefing_prompt_says_quiet_tickers_are_trimmed(monkeypatch):
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = _response(
        "end_turn", [_text_block("ok")]
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {})

    prompt = mock_client.beta.messages.create.call_args[1]["messages"][0]["content"]
    assert "normal tickers list only their top 3 headlines" in prompt


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


def _scored(headline: str, label: str = "negative", **extra) -> dict:
    return {"headline": headline, "summary": "", "label": label, "score": 0.9, **extra}


def test_build_headlines_cache_drops_articles_already_in_prompt():
    stats = {
        **_sample_stats("NVDA", "unusual"),
        "selected_articles": [_scored("Today A"), _scored("Today B")],
        "selected_headline_articles": [
            _scored("Today A", date="2024-06-03", source="Reuters"),
            _scored("Earlier C", date="2024-06-01", source="Bloomberg"),
        ],
    }
    result = briefing._build_headlines_cache([stats], _MAX)["NVDA"]
    assert "Earlier C" in result
    assert "Today A" not in result


def test_build_headlines_cache_drops_plain_headlines_when_finbert_unavailable():
    stats = {
        **_sample_stats("NVDA", "unusual"),
        "selected_articles": [_scored("Headline A", label="unavailable")],
        "selected_headline_articles": [
            _scored("Headline A", date="2024-06-03", source="Reuters"),
            _scored("Earlier C", date="2024-06-01", source="Bloomberg"),
        ],
    }
    result = briefing._build_headlines_cache([stats], _MAX)["NVDA"]
    assert "Earlier C" in result
    assert "Headline A" not in result


def test_build_headlines_cache_says_so_when_everything_already_shown():
    stats = {
        **_sample_stats("NVDA", "unusual"),
        "selected_articles": [_scored("Today A")],
        "selected_headline_articles": [
            _scored("Today A", date="2024-06-03", source="Reuters")
        ],
    }
    assert briefing._build_headlines_cache([stats], _MAX)["NVDA"] == (
        "No headlines beyond those already listed in the watchlist statistics."
    )


def test_build_headlines_cache_quiet_ticker_keeps_articles_not_in_its_top_three():
    stats = {
        **_sample_stats("XOM", "normal"),
        "selected_articles": [_scored(f"Today {i}", label="neutral") for i in range(5)],
        "selected_headline_articles": [
            _scored(f"Today {i}", date="2024-06-03", source="Reuters") for i in range(5)
        ],
    }
    result = briefing._build_headlines_cache([stats], _MAX)["XOM"]
    assert "Today 0" not in result
    assert "Today 3" in result


def test_build_headlines_cache_skips_error_entries_and_handles_empty_window():
    stats = [
        {"ticker": "FAIL", "error": "fetch failed"},
        _sample_stats("NVDA", "unusual"),
    ]
    assert briefing._build_headlines_cache(stats, _MAX) == {
        "NVDA": "No headline context available."
    }


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
    mock_client.beta.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "normal")], _BASELINE, _DAYS, {}, {}
    )
    assert "Watchlist is quiet today." in result.text


def test_run_briefing_prompt_uses_configured_baseline_days(monkeypatch):
    """baseline_days must appear in the prompt, not the literal '30'."""
    resp = _response("end_turn", [_text_block("ok")])
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    briefing._run_briefing(
        [_sample_stats()],
        baseline_days=60,
        headline_days=14,
        headlines_cache={},
        intraday_cache={},
    )

    call_kwargs = mock_client.beta.messages.create.call_args[1]
    prompt_text = call_kwargs["messages"][0]["content"]
    assert "60-day" in prompt_text
    assert "14-day" in prompt_text


def test_run_briefing_unexpected_stop_reason(monkeypatch):
    resp = _response("max_tokens", [])
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {})
    assert "max_tokens" in result.text


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
    mock_client.beta.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, headlines_cache, {}
    )
    assert "NVDA" in result.text

    second_msgs = mock_client.beta.messages.create.call_args_list[1][1]["messages"]
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
    mock_client.beta.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = briefing._run_briefing(
        [_sample_stats("NVDA", "unusual")], _BASELINE, _DAYS, {}, intraday_cache
    )
    assert "NVDA" in result.text

    second_msgs = mock_client.beta.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "09:30:00" in tool_result_content
    assert tool_result_content == intraday_cache["NVDA"]


def test_run_briefing_tools_list_includes_price_intraday(monkeypatch):
    """The Claude tool schema must offer get_price_intraday, not just headlines."""
    resp = _response("end_turn", [_text_block("ok")])
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {})

    call_kwargs = mock_client.beta.messages.create.call_args[1]
    tool_names = {t["name"] for t in call_kwargs["tools"]}
    assert tool_names == {"get_news_headlines", "get_price_intraday"}


def test_run_briefing_uses_llm_config(monkeypatch):
    """Model, effort and fallback come from the [llm] config, not hard-coded."""
    resp = _response("end_turn", [_text_block("ok")])
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    llm = LLMConfig(model="claude-sonnet-5-5", effort="high", fallback=False)
    briefing._run_briefing([_sample_stats()], _BASELINE, _DAYS, {}, {}, llm)

    call_kwargs = mock_client.beta.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-5-5"
    assert call_kwargs["output_config"] == {"effort": "high"}
    assert "betas" not in call_kwargs
    assert "extra_body" not in call_kwargs


# ── main ──────────────────────────────────────────────────────────────────────


def test_main(monkeypatch, capsys):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda tickers: [_sample_stats("NVDA", "unusual")],
    )
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: _loop_result("NVDA is unusual."),
    )
    exit_code = briefing.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DAILY BRIEFING" in out
    assert "NVDA is unusual." in out


def test_main_skips_claude_when_nothing_flagged(monkeypatch, capsys):
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: None)
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda tickers: [
            {**_sample_stats("NVDA", "normal"), "price": _sample_price_stats()},
            {"ticker": "FAIL", "error": "fetch failed"},
        ],
    )
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    run_briefing = MagicMock()
    monkeypatch.setattr("financial_news.briefing._run_briefing", run_briefing)
    intraday = MagicMock()
    monkeypatch.setattr("financial_news.briefing.fetch_intraday_prices", intraday)

    assert briefing.main() == 0

    run_briefing.assert_not_called()
    intraday.assert_not_called()
    out = capsys.readouterr().out
    assert "All 1 tickers are normal: NVDA." in out
    assert "Fetch failed: FAIL." in out


def test_main_calls_claude_when_only_price_flagged(monkeypatch, capsys):
    stats = {**_sample_stats("NVDA", "normal"), "price": _sample_price_stats()}
    stats["price"]["classification"] = "unusual"
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: [stats])
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    monkeypatch.setattr(
        "financial_news.briefing.fetch_intraday_prices",
        lambda ticker: {"bars": []},
    )
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: _loop_result("Price move."),
    )

    briefing.main()

    assert "Price move." in capsys.readouterr().out


def test_main_exports_usage_with_flagged_count(monkeypatch, capsys):
    stats = [_sample_stats("NVDA", "unusual"), _sample_stats("F", "normal")]
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: stats)
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    monkeypatch.setattr(
        "financial_news.briefing.fetch_intraday_prices", lambda t: {"bars": []}
    )
    result = _loop_result("NVDA.")
    monkeypatch.setattr("financial_news.briefing._run_briefing", lambda *a: result)
    export = MagicMock()
    monkeypatch.setattr("financial_news.briefing._export_usage", export)

    briefing.main()

    export.assert_called_once_with(result, 1)


def test_main_exports_flagged_count_only_on_quiet_day(monkeypatch, capsys):
    stats = [_sample_stats("F", "normal")]
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: stats)
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    export = MagicMock()
    monkeypatch.setattr("financial_news.briefing._export_usage", export)

    briefing.main()

    export.assert_called_once_with(None, 0)


# ── _export_usage ─────────────────────────────────────────────────────────────


def _gauges(monkeypatch) -> tuple[dict[str, MagicMock], MagicMock, MagicMock]:
    """Capture the gauges _export_usage creates, keyed by metric name."""
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc")
    gauges: dict[str, MagicMock] = {}
    provider = MagicMock()
    provider.get_meter.return_value.create_gauge.side_effect = (
        lambda name, description: gauges.setdefault(name, MagicMock())
    )
    build = MagicMock(return_value=provider)
    monkeypatch.setattr("financial_news.briefing.build_meter_provider", build)
    return gauges, provider, build


def test_export_usage_sends_tokens_cost_and_ratios(monkeypatch):
    gauges, provider, build = _gauges(monkeypatch)

    briefing._export_usage(_loop_result("x", cost_usd=0.06), flagged=2)

    build.assert_called_once_with("financial-news-briefing")
    gauges["financial_news.briefing_flagged_tickers"].set.assert_called_once_with(2)
    gauges["financial_news.briefing_tokens"].set.assert_any_call(
        6000, {"type": "cache_read_input_tokens"}
    )
    assert gauges["financial_news.briefing_tokens"].set.call_count == 4
    gauges["financial_news.briefing_turns"].set.assert_called_once_with(3)
    gauges["financial_news.briefing_cache_hit_ratio"].set.assert_called_once_with(0.6)
    gauges["financial_news.briefing_cost_usd"].set.assert_called_once_with(0.06)
    gauges[
        "financial_news.briefing_cost_per_flagged_ticker_usd"
    ].set.assert_called_once_with(0.03)
    provider.force_flush.assert_called_once()
    provider.shutdown.assert_called_once()


def test_export_usage_quiet_day_sends_flagged_count_only(monkeypatch):
    gauges, _, _ = _gauges(monkeypatch)

    briefing._export_usage(None, flagged=0)

    assert set(gauges) == {"financial_news.briefing_flagged_tickers"}


def test_export_usage_omits_cost_when_model_price_unknown(monkeypatch):
    gauges, _, _ = _gauges(monkeypatch)

    briefing._export_usage(_loop_result("x", cost_usd=None), flagged=1)

    assert "financial_news.briefing_cost_usd" not in gauges
    assert "financial_news.briefing_cost_per_flagged_ticker_usd" not in gauges
    assert "financial_news.briefing_tokens" in gauges


def test_export_usage_omits_cache_ratio_when_no_input(monkeypatch):
    gauges, _, _ = _gauges(monkeypatch)
    result = ToolLoopResult("x", 1, dict.fromkeys(_loop_result("x").usage, 0), 0.0)

    briefing._export_usage(result, flagged=1)

    assert "financial_news.briefing_cache_hit_ratio" not in gauges


def test_export_usage_skipped_without_grafana_config(monkeypatch):
    build = MagicMock()
    monkeypatch.setattr("financial_news.briefing.build_meter_provider", build)

    briefing._export_usage(_loop_result("x"), flagged=1)

    build.assert_not_called()


def test_export_usage_failure_does_not_raise(monkeypatch, caplog):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc")
    monkeypatch.setattr(
        "financial_news.briefing.build_meter_provider",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    briefing._export_usage(_loop_result("x"), flagged=1)

    assert "briefing usage export failed" in caplog.text


# ── _quiet_briefing ───────────────────────────────────────────────────────────


def test_quiet_briefing_lists_normal_tickers():
    stats = [
        {**_sample_stats("NVDA", "normal"), "price": _sample_price_stats()},
        {**_sample_stats("F", "normal"), "price": _sample_price_stats("F")},
    ]
    assert briefing._quiet_briefing(stats) == (
        "No elevated or unusual news-volume or price activity today."
        " All 2 tickers are normal: NVDA, F."
    )


def test_quiet_briefing_notes_missing_price_and_failures():
    stats = [
        _sample_stats("NVDA", "normal"),
        {"ticker": "FAIL", "error": "fetch failed"},
    ]
    text = briefing._quiet_briefing(stats)
    assert "Price signal unavailable (news volume only): NVDA." in text
    assert "Fetch failed: FAIL." in text


def test_quiet_briefing_all_failed():
    text = briefing._quiet_briefing([{"ticker": "FAIL", "error": "fetch failed"}])
    assert text == "Fetch failed: FAIL."


def test_quiet_briefing_empty():
    assert briefing._quiet_briefing([]) == "No watchlist data available."


def test_main_uses_snapshot_when_available(monkeypatch, capsys):
    snap_stats = [_sample_stats("NVDA", "unusual")]
    monkeypatch.setattr("financial_news.snapshot.read", lambda *a, **kw: snap_stats)
    collect_called = []
    monkeypatch.setattr(
        "financial_news.briefing._collect_stats",
        lambda *a, **kw: collect_called.append(True) or [],
    )
    monkeypatch.setattr("financial_news.briefing.enrich_stats", lambda s, *a, **kw: s)
    monkeypatch.setattr(
        "financial_news.briefing._run_briefing",
        lambda *a, **kw: _loop_result("From snapshot."),
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
        lambda *a, **kw: _loop_result("From collect."),
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
        lambda *a, **kw: _loop_result(briefing_text),
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
