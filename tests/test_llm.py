import logging
from unittest.mock import MagicMock

import anthropic
from claude_mocks import response, text_block, tool_use_block

from financial_news.config import LLMConfig
from financial_news.llm import run_tool_loop


def _run(monkeypatch, responses: list) -> str:
    mock_client = MagicMock()
    mock_client.beta.messages.create.side_effect = responses
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)
    return run_tool_loop("system", "prompt", [], lambda b: "ok", LLMConfig(), "Test")


def test_logs_usage_per_turn_and_total(monkeypatch, caplog):
    turns = [
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
    with caplog.at_level(logging.INFO, logger="financial_news.llm"):
        assert _run(monkeypatch, turns) == "done"

    messages = [r.getMessage() for r in caplog.records]
    assert (
        "llm_usage label=Test turn=1 model=claude-opus-5-5 stop_reason=tool_use"
        " input_tokens=1000 cache_creation_input_tokens=200"
        " cache_read_input_tokens=0 output_tokens=30"
    ) in messages
    assert (
        "llm_usage_total label=Test turns=2 input_tokens=1050"
        " cache_creation_input_tokens=200 cache_read_input_tokens=1200"
        " output_tokens=430"
    ) in messages


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

    assert "max_tokens" in result
    assert "llm_usage_total label=Test turns=1" in caplog.text
