import gzip
import logging
from datetime import datetime

from financial_news.server import _TimestampRotatingFileHandler


def test_should_rollover_false_when_under_limit(tmp_path):
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    handler = _TimestampRotatingFileHandler(
        tmp_path / "test_01012026_120000.log",
        max_bytes=10 * 1024 * 1024,
        created_at=created_at,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, "", 0, "short message", (), None)
    assert not handler.shouldRollover(record)
    handler.close()


def test_should_rollover_true_when_over_limit(tmp_path):
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    handler = _TimestampRotatingFileHandler(
        tmp_path / "test_01012026_120000.log", max_bytes=1, created_at=created_at
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(
        "test", logging.INFO, "", 0, "message longer than 1 byte", (), None
    )
    assert handler.shouldRollover(record)
    handler.close()


def test_do_rollover_compresses_and_truncates(tmp_path):
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    log_file = tmp_path / "test_01012026_120000.log"
    handler = _TimestampRotatingFileHandler(
        log_file,
        max_bytes=10 * 1024 * 1024,
        created_at=created_at,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.stream.write("hello world\n")
    handler.stream.flush()

    handler.doRollover()

    # Original file should now be gzipped
    gz_files = list(tmp_path.glob("test_01012026_120000.log.gz"))
    assert len(gz_files) == 1

    with gzip.open(gz_files[0], "rt") as f:
        assert "hello world" in f.read()

    # New active file should exist with a different (later) timestamp
    new_logs = list(tmp_path.glob("test_*.log"))
    assert len(new_logs) == 1
    new_log_file = new_logs[0]
    assert new_log_file.name != log_file.name  # Different filename (timestamp)
    assert new_log_file.stat().st_size == 0  # Empty

    handler.close()


def test_emit_flushes_stream(tmp_path):
    """Test that emit method flushes the stream after writing."""
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    log_file = tmp_path / "test_01012026_120000.log"
    handler = _TimestampRotatingFileHandler(
        log_file,
        max_bytes=10 * 1024 * 1024,
        created_at=created_at,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Create a record and emit it
    record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
    handler.emit(record)

    # Check that the file was written to (stream should be flushed)
    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content

    handler.close()


def test_handler_created_at_none(tmp_path):
    """Test handler initialization when created_at is None."""
    log_file = tmp_path / "test.log"
    handler = _TimestampRotatingFileHandler(
        log_file,
        max_bytes=10 * 1024 * 1024,
        created_at=None,
    )

    # Should use current time when created_at is None
    assert handler._created_at is not None
    assert isinstance(handler._created_at, datetime)

    handler.close()
