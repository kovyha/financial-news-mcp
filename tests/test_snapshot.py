"""Tests for financial_news.snapshot."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from financial_news import snapshot


def _write_raw(path: Path, snap_date: date, stats: list[dict]) -> None:
    path.write_text(json.dumps({"date": snap_date.isoformat(), "stats": stats}))


def test_write_creates_file_with_correct_structure(tmp_path):
    path = tmp_path / "snap.json"
    stats = [{"ticker": "NVDA", "z_score": 1.5}]
    snapshot.write(stats, path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["date"] == date.today().isoformat()
    assert payload["stats"] == stats


def test_write_overwrites_existing_file(tmp_path):
    path = tmp_path / "snap.json"
    snapshot.write([{"ticker": "NVDA"}], path)
    snapshot.write([{"ticker": "TSLA"}], path)
    payload = json.loads(path.read_text())
    assert payload["stats"][0]["ticker"] == "TSLA"


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "snap.json"
    snapshot.write([{"ticker": "AMD"}], path)
    assert path.exists()


def test_write_is_atomic_no_tmp_file_left(tmp_path):
    path = tmp_path / "snap.json"
    snapshot.write([{"ticker": "NVDA"}], path)
    assert not path.with_suffix(".tmp").exists()


def test_read_returns_stats_for_todays_snapshot(tmp_path):
    path = tmp_path / "snap.json"
    stats = [{"ticker": "NVDA", "z_score": 2.1}]
    _write_raw(path, date.today(), stats)
    assert snapshot.read(path) == stats


def test_read_returns_none_when_file_missing(tmp_path):
    assert snapshot.read(tmp_path / "nonexistent.json") is None


def test_read_returns_none_for_yesterday_snapshot(tmp_path):
    path = tmp_path / "snap.json"
    _write_raw(path, date.today() - timedelta(days=1), [{"ticker": "NVDA"}])
    assert snapshot.read(path) is None


def test_read_returns_none_for_future_dated_snapshot(tmp_path):
    path = tmp_path / "snap.json"
    _write_raw(path, date.today() + timedelta(days=1), [{"ticker": "NVDA"}])
    assert snapshot.read(path) is None


def test_roundtrip_preserves_all_stat_fields(tmp_path):
    path = tmp_path / "snap.json"
    stats = [
        {
            "ticker": "NVDA",
            "z_score": 3.5,
            "recent_count": 12,
            "mean": 3.0,
            "std": 1.0,
            "classification": "unusual",
            "headlines": ["Headline A", "Headline B"],
            "baseline_counts": [3.0] * 30,
        }
    ]
    snapshot.write(stats, path)
    assert snapshot.read(path) == stats


def test_read_uses_todays_date_not_file_mtime(tmp_path):
    """Date check is semantic (payload field), not filesystem-based."""
    path = tmp_path / "snap.json"
    yesterday = date.today() - timedelta(days=1)
    with patch("financial_news.snapshot.datetime") as mock_dt:
        mock_dt.now.return_value.date.return_value = yesterday
        snapshot.write([{"ticker": "NVDA"}], path)
    # File was written as yesterday's snapshot — today's read should reject it
    assert snapshot.read(path) is None


def test_read_returns_none_for_corrupt_json(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text("{not valid json")
    assert snapshot.read(path) is None


def test_read_returns_none_for_missing_stats_key(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"date": date.today().isoformat()}))
    assert snapshot.read(path) is None
