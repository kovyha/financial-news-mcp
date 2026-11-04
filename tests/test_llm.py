import logging
from unittest.mock import MagicMock

import anthropic
import pytest
from claude_mocks import response, text_block, tool_use_block

from financial_news.config import LLMConfig
from financial_news.llm import ToolLoopResult, run_tool_loop


def _run(monkeypatch, responses: list) -> ToolLoopResult:
    mock_client = MagicMock()
    mock_client.beta.messages.create.side_effect = responses
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)
    return run_tool_loop("system", "prompt", [], lambda b: "ok", LLMConfig(), "Test")


def _two_turns() -> list:
    return [
        response(
            "tool_use",
            [tool_use_block("c1", "get_news_headlines", {"ticker": "NVDA"})],
            input_tokens=1000,
            cache_creation_input_tokens=200,
            output_tokens=30,
        ),
        response(
            "end_turn",
            [text_block("done")],
            input_tokens=50,
            cache_read_input_tokens=1200,
            output_tokens=400,
        ),
    ]


def test_logs_usage_and_cost_per_turn_and_total(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="financial_news.llm"):
        _run(monkeypatch, _two_turns())

    messages = [r.getMessage() for r in caplog.records]
    # Opus 5.5: 1000*$4 + 200*$5 + 30*$20 per MTok = $0.0056
    assert (
        "llm_usage label=Test turn=1 model=claude-opus-5-5 stop_reason=tool_use"
        " input_tokens=1000 cache_creation_input_tokens=200"
        " cache_read_input_tokens=0 output_tokens=30 cost_usd=0.0056"
    ) in messages
    # Turn 2: 50*$4 + 1200*$0.20 + 400*$20 = $0.00844; total $0.01404
    assert (
        "llm_usage_total label=Test turns=2 input_tokens=1050"
        " cache_creation_input_tokens=200 cache_read_input_tokens=1200"
        " output_tokens=430 cost_usd=0.0140"
    ) in messages


def test_returns_text_turns_usage_and_cost(monkeypatch):
    result = _run(monkeypatch, _two_turns())

    assert result.text == "done"
    assert result.turns == 2
    assert result.usage == {
        "input_tokens": 1050,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 1200,
        "output_tokens": 430,
    }
    assert result.cost_usd == pytest.approx(0.01404)


def test_cache_hit_ratio_is_cache_reads_over_all_input():
    usage = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 3000,
        "cache_read_input_tokens": 6000,
        "output_tokens": 99999,
    }
    assert ToolLoopResult("x", 3, usage, None).cache_hit_ratio == 0.6


def test_cache_hit_ratio_none_without_input():
    usage = dict.fromkeys(
        [
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        ],
        0,
    )
    assert ToolLoopResult("x", 1, usage, None).cache_hit_ratio is None


def test_cost_unknown_when_a_turn_runs_on_an_unpriced_model(monkeypatch, caplog):
    turns = _two_turns()
    turns[0].model = "claude-unpriced"
    with caplog.at_level(logging.INFO, logger="financial_news.llm"):
        result = _run(monkeypatch, turns)

    assert result.cost_usd is None
    assert "cost_usd=unknown" in caplog.text


def test_caches_the_conversation_not_just_the_static_prefix(monkeypatch):
    mock_client = MagicMock()
    mock_client.beta.messages.create.return_value = response(
        "end_turn", [text_block("done")]
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    run_tool_loop("system", "prompt", [], lambda b: "ok", LLMConfig(), "Test")

    kwargs = mock_client.beta.messages.create.call_args[1]
    assert kwargs["cache_control"] == {"type": "ephemeral"}


def test_missing_cache_counts_log_as_zero(monkeypatch, caplog):
    turns = [
        response(
            "end_turn",
            [text_block("done")],
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
    ]
    with caplog.at_level(logging.INFO, logger="financial_news.llm"):
        _run(monkeypatch, turns)

    assert "cache_creation_input_tokens=0 cache_read_input_tokens=0" in caplog.text


def test_logs_total_on_unexpected_stop(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="financial_news.llm"):
        result = _run(monkeypatch, [response("max_tokens", [])])

    assert "max_tokens" in result.text
    assert "llm_usage_total label=Test turns=1" in caplog.text
