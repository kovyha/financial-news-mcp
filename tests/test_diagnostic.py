"""Tests for the diagnostic module."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import anthropic

from financial_news import diagnostic

# ── Helpers for building mock Anthropic response objects ──────────────────────


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(block_id: str, path: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.input = {"path": path}
    return block


def _response(stop_reason: str, content: list) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


def test_read_error_log_for_date_empty_file():
    """Test reading from an empty log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_file.write_text("")
        result = diagnostic.read_error_log_for_date(log_file, date(2026, 4, 20))
        assert result == []


def test_read_error_log_for_date_matching_lines():
    """Test that matching date lines are returned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_content = (
            "20-04-2026 10:00:00 ERROR error 1\n"
            "20-04-2026 11:00:00 ERROR error 2\n"
            "19-04-2026 09:00:00 ERROR error 3\n"
        )
        log_file.write_text(log_content)
        result = diagnostic.read_error_log_for_date(log_file, date(2026, 4, 20))
        assert len(result) == 2
        assert "error 1" in result[0]
        assert "error 2" in result[1]


def test_read_error_log_for_date_nonexistent_file():
    """Test handling of nonexistent log file."""
    result = diagnostic.read_error_log_for_date(
        Path("/nonexistent/path.log"), date(2026, 4, 20)
    )
    assert result == []


def test_read_error_log_for_date_different_date():
    """Test that lines from different dates are filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_content = (
            "19-04-2026 10:00:00 ERROR error 1\n"
            "20-04-2026 11:00:00 ERROR error 2\n"
            "21-04-2026 09:00:00 ERROR error 3\n"
        )
        log_file.write_text(log_content)
        result = diagnostic.read_error_log_for_date(log_file, date(2026, 4, 20))
        assert len(result) == 1
        assert "error 2" in result[0]


def test_diagnose_no_errors():
    """Test diagnostic when no errors are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_file.write_text("")
        report = diagnostic.diagnose(log_path=log_file, target_date=date(2026, 4, 20))
        assert "✅ Diagnostic report" in report
        assert "No errors found" in report


def test_diagnose_with_errors():
    """Test diagnostic when errors are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_content = (
            "20-04-2026 10:00:00 ERROR something went wrong\n"
            "20-04-2026 11:00:00 ERROR another problem\n"
        )
        log_file.write_text(log_content)
        report = diagnostic.diagnose(
            log_path=log_file, target_date=date(2026, 4, 20), use_llm=False
        )
        assert "🔍 Diagnostic report" in report
        assert "Errors found: 2" in report
        assert "something went wrong" in report
        assert "another problem" in report


def test_diagnose_limits_output():
    """Test that diagnostic limits output to first 20 errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        # Create 30 errors
        lines = [f"20-04-2026 {i:02d}:00:00 ERROR error {i}" for i in range(30)]
        log_file.write_text("\n".join(lines) + "\n")
        report = diagnostic.diagnose(
            log_path=log_file, target_date=date(2026, 4, 20), use_llm=False
        )
        assert "error 0" in report
        assert "error 19" in report
        assert "... and 10 more errors" in report


def test_read_error_log_for_date_file_read_error(monkeypatch):
    """Test handling of file read errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_file.write_text("20-04-2026 10:00:00 ERROR test")

        def mock_open(*args, **kwargs):
            raise IOError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open)
        result = diagnostic.read_error_log_for_date(log_file, date(2026, 4, 20))
        assert result == []


def test_diagnose_with_default_parameters(monkeypatch):
    """Test diagnose with default parameters (uses config and today's date)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        log_file.write_text("")

        # Mock config to return our temp directory
        class MockConfig:
            class Logging:
                log_dir = tmpdir
                filename = "test"

            logging = Logging()

        monkeypatch.setattr(
            "financial_news.diagnostic.load_config", lambda: MockConfig()
        )
        report = diagnostic.diagnose()
        assert "Diagnostic report" in report


def test_main_no_errors(monkeypatch, capsys):
    """Test main function when no errors are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock config to return our temp directory
        class MockConfig:
            class Logging:
                log_dir = tmpdir
                filename = "test"

            logging = Logging()

        monkeypatch.setattr(
            "financial_news.diagnostic.load_config", lambda: MockConfig()
        )

        # Create empty log file
        log_file = Path(tmpdir) / "test.log"
        log_file.write_text("")

        exit_code = diagnostic.main()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Diagnostic report" in captured.out


def test_main_with_errors(monkeypatch, capsys):
    """Test main function when errors are found."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock config to return our temp directory
        class MockConfig:
            class Logging:
                log_dir = tmpdir
                filename = "test"

            logging = Logging()

        monkeypatch.setattr(
            "financial_news.diagnostic.load_config", lambda: MockConfig()
        )

        # Create log file with the expected pattern
        log_file = Path(tmpdir) / "test.error.log"
        log_file.write_text(
            f"{date.today().strftime('%d-%m-%Y')} 10:00:00 ERROR test error\n"
        )

        exit_code = diagnostic.main()
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Diagnostic report" in captured.out


# ── _analyze_with_llm tests ───────────────────────────────────────────────────


def test_analyze_with_llm_end_turn(monkeypatch, tmp_path):
    """end_turn immediately — no tool calls needed."""
    resp = _response("end_turn", [_text_block("Root cause: invalid API key.")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = diagnostic._analyze_with_llm(["16-05-2026 ERROR boom"], tmp_path)
    assert "Root cause: invalid API key." in result


def test_analyze_with_llm_unexpected_stop_reason(monkeypatch, tmp_path):
    """Non-tool_use, non-end_turn stop_reason returns an error string."""
    resp = _response("max_tokens", [])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = diagnostic._analyze_with_llm(["error"], tmp_path)
    assert "max_tokens" in result


def test_analyze_with_llm_tool_use_valid_file(monkeypatch, tmp_path):
    """tool_use loop: Claude reads a real file then returns end_turn."""
    (tmp_path / "analysis.py").write_text("# z-score logic")

    # First call returns tool_use (with a non-tool block too, to cover the skip branch)
    tool_resp = _response(
        "tool_use",
        [_text_block("thinking..."), _tool_use_block("c1", "analysis.py")],
    )
    end_resp = _response("end_turn", [_text_block("Fix: update EWM span.")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    result = diagnostic._analyze_with_llm(["error"], tmp_path)
    assert "Fix: update EWM span." in result

    # Verify file content was passed back as tool result
    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "z-score logic" in tool_result_content


def test_analyze_with_llm_path_traversal_rejected(monkeypatch, tmp_path):
    """Path traversal outside project root is blocked."""
    tool_resp = _response("tool_use", [_tool_use_block("c1", "../../../etc/passwd")])
    end_resp = _response("end_turn", [_text_block("done")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    diagnostic._analyze_with_llm(["error"], tmp_path)

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "path traversal not allowed" in tool_result_content


def test_analyze_with_llm_file_not_found(monkeypatch, tmp_path):
    """Missing file path returns a clear error in the tool result."""
    tool_resp = _response("tool_use", [_tool_use_block("c1", "nonexistent.py")])
    end_resp = _response("end_turn", [_text_block("done")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    diagnostic._analyze_with_llm(["error"], tmp_path)

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "File not found" in tool_result_content


def test_analyze_with_llm_file_read_error(monkeypatch, tmp_path):
    """IOError during file read is returned as an error string in the tool result."""
    target = tmp_path / "bad.py"
    target.write_text("content")

    def broken_read_text(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", broken_read_text)

    tool_resp = _response("tool_use", [_tool_use_block("c1", "bad.py")])
    end_resp = _response("end_turn", [_text_block("done")])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_resp, end_resp]
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    diagnostic._analyze_with_llm(["error"], tmp_path)

    second_msgs = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_content = second_msgs[-1]["content"][0]["content"]
    assert "Error reading" in tool_result_content


# ── diagnose() LLM branch tests ───────────────────────────────────────────────


def test_diagnose_with_llm_success(monkeypatch, tmp_path):
    """diagnose() calls LLM when ANTHROPIC_API_KEY is present and use_llm=True."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    resp = _response("end_turn", [_text_block("LLM found the bug.")])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    log_file = tmp_path / "test.error.log"
    log_file.write_text("20-04-2026 10:00:00 ERROR something failed\n")

    report = diagnostic.diagnose(
        log_path=log_file, target_date=date(2026, 4, 20), use_llm=True
    )
    assert "LLM root-cause analysis" in report
    assert "LLM found the bug." in report


def test_diagnose_llm_failure_falls_back(monkeypatch, tmp_path):
    """diagnose() falls back gracefully when the LLM call raises."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API unavailable")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)

    log_file = tmp_path / "test.error.log"
    log_file.write_text("20-04-2026 10:00:00 ERROR something failed\n")

    report = diagnostic.diagnose(
        log_path=log_file, target_date=date(2026, 4, 20), use_llm=True
    )
    assert "LLM analysis failed" in report
