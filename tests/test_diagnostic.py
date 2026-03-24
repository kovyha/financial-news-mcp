"""Tests for the diagnostic module."""

import tempfile
from datetime import date
from pathlib import Path

from financial_news import diagnostic


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
        report = diagnostic.diagnose(log_path=log_file, target_date=date(2026, 4, 20))
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
        report = diagnostic.diagnose(log_path=log_file, target_date=date(2026, 4, 20))
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
