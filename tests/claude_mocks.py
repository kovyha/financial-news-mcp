"""Mock Anthropic response objects shared by the briefing and diagnostic tests."""

from unittest.mock import MagicMock


def text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def tool_use_block(block_id: str, name: str, tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = tool_input
    return block


def response(
    stop_reason: str,
    content: list,
    input_tokens: int = 100,
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    output_tokens: int = 50,
) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    resp.model = "claude-opus-5-5"
    resp.usage.input_tokens = input_tokens
    resp.usage.cache_creation_input_tokens = cache_creation_input_tokens
    resp.usage.cache_read_input_tokens = cache_read_input_tokens
    resp.usage.output_tokens = output_tokens
    return resp
