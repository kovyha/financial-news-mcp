"""Claude tool-use loop shared by the LLM-layer agents (briefing and diagnostic).

LLM layer only: the deterministic entry points (server, monitor) must never import
this module, which tests/test_boundary.py enforces.
"""

import logging
from collections.abc import Callable
from typing import Any

import anthropic

from financial_news.config import LLMConfig

logger = logging.getLogger(__name__)

_MAX_TOKENS = 16000
_THINKING = {"type": "adaptive"}
# Billed separately, each at its own rate, so they're logged separately.
_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _log_usage(label: str, turn: int, response, totals: dict[str, int]) -> None:
    """Log one response's token usage and add it to the run's totals."""
    counts = {f: getattr(response.usage, f, None) or 0 for f in _USAGE_FIELDS}
    for field, count in counts.items():
        totals[field] += count
    logger.info(
        "llm_usage label=%s turn=%d model=%s stop_reason=%s %s",
        label,
        turn,
        response.model,
        response.stop_reason,
        " ".join(f"{f}={c}" for f, c in counts.items()),
    )


def run_tool_loop(
    system_prompt: str,
    prompt: str,
    tools: list[dict],
    answer_tool: Callable[[Any], str],
    llm: LLMConfig,
    label: str,
) -> str:
    """Call Claude until it finishes, answering each tool call with answer_tool.

    Returns Claude's final text, or "<label> ended unexpectedly (...)" if it stops
    for any reason other than finishing or calling a tool.
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    totals = dict.fromkeys(_USAGE_FIELDS, 0)
    turn = 0
    while True:
        turn += 1
        response = client.beta.messages.create(
            **llm.request_params(),
            max_tokens=_MAX_TOKENS,
            thinking=_THINKING,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=tools,
            messages=messages,
        )
        _log_usage(label, turn, response, totals)
        if response.stop_reason != "tool_use":
            logger.info(
                "llm_usage_total label=%s turns=%d %s",
                label,
                turn,
                " ".join(f"{f}={c}" for f, c in totals.items()),
            )
        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "tool_use":
            return f"{label} ended unexpectedly (stop_reason={response.stop_reason})."
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": answer_tool(b)}
            for b in response.content
            if b.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})
