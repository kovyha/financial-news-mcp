"""Claude tool-use loop shared by the LLM-layer agents (briefing and diagnostic).

LLM layer only: the deterministic entry points (server, monitor) must never import
this module, which tests/test_boundary.py enforces.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
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
# USD per million tokens, in _USAGE_FIELDS order: input, cache write (5-minute
# cache), cache read, output. Used only to estimate cost for monitoring; the bill
# is the source of truth. Checked against
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-04; add the
# model here when changing [llm] model, or cost is reported as unknown. Cache
# reads are not a fixed share of input across models (0.05x on Opus/Sonnet 5.5,
# 0.025x on Fable 5.1), so take each value from the page.
_PRICES_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 2.50, 0.10, 10.00),
    "claude-fable-5-1": (10.00, 12.50, 0.25, 50.00),
}


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    turns: int
    usage: dict[str, int]
    # None if any turn ran on a model missing from _PRICES_PER_MTOK.
    cost_usd: float | None

    @property
    def cache_hit_ratio(self) -> float | None:
        """Share of input tokens read from cache; None if there was no input."""
        input_total = sum(self.usage[f] for f in _USAGE_FIELDS if f != "output_tokens")
        if not input_total:
            return None
        return self.usage["cache_read_input_tokens"] / input_total


def _cost_usd(model: str, counts: dict[str, int]) -> float | None:
    prices = _PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return sum(counts[f] * p for f, p in zip(_USAGE_FIELDS, prices)) / 1_000_000


def _format_cost(cost: float | None) -> str:
    return "unknown" if cost is None else f"{cost:.4f}"


def run_tool_loop(
    system_prompt: str,
    prompt: str,
    tools: list[dict],
    answer_tool: Callable[[Any], str],
    llm: LLMConfig,
    label: str,
) -> ToolLoopResult:
    """Call Claude until it finishes, answering each tool call with answer_tool.

    The result's text is Claude's final text, or "<label> ended unexpectedly (...)"
    if it stops for any reason other than finishing or calling a tool. Token usage
    and estimated cost are logged per turn and per run, and returned.
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    totals = dict.fromkeys(_USAGE_FIELDS, 0)
    total_cost: float | None = 0.0
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
            # Moves a breakpoint to the end of each request, so the next turn reads
            # everything already sent (prompt, earlier tool results) from cache.
            cache_control={"type": "ephemeral"},
        )

        counts = {f: getattr(response.usage, f, None) or 0 for f in _USAGE_FIELDS}
        for field, count in counts.items():
            totals[field] += count
        cost = _cost_usd(response.model, counts)
        total_cost = None if cost is None or total_cost is None else total_cost + cost
        logger.info(
            "llm_usage label=%s turn=%d model=%s stop_reason=%s %s cost_usd=%s",
            label,
            turn,
            response.model,
            response.stop_reason,
            " ".join(f"{f}={c}" for f, c in counts.items()),
            _format_cost(cost),
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": answer_tool(b)}
                for b in response.content
                if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
            continue

        logger.info(
            "llm_usage_total label=%s turns=%d %s cost_usd=%s",
            label,
            turn,
            " ".join(f"{f}={c}" for f, c in totals.items()),
            _format_cost(total_cost),
        )
        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if b.type == "text")
        else:
            text = f"{label} ended unexpectedly (stop_reason={response.stop_reason})."
        return ToolLoopResult(text, turn, totals, total_cost)
