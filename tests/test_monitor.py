"""Tests for financial_news.monitor."""

import math
from unittest.mock import MagicMock, patch

import pytest

from financial_news import analysis
from financial_news.analysis import GAUGE_SPECS
from financial_news.config import AnalysisConfig
from financial_news.monitor import _build_meter_provider, main, run


@pytest.fixture()
def mock_meter():
    meter = MagicMock()
    meter.create_gauge.return_value = MagicMock()
    meter.create_counter.return_value = MagicMock()
    return meter


@pytest.fixture()
def mock_provider():
    return MagicMock()


@pytest.fixture()
def good_stats():
    return {
        "z_score": 1.5,
        "recent_count": 6,
        "mean": 4.0,
        "std": 1.0,
        "classification": "normal",
        "headlines": [],
        "baseline_counts": [],
    }


def _run_with_mocks(mock_meter, mock_provider, tickers, stats_by_ticker):
    from pathlib import Path

    def compute_side_effect(ticker):
        if isinstance(stats_by_ticker, Exception):
            raise stats_by_ticker
        result = stats_by_ticker.get(ticker)
        if isinstance(result, Exception):
            raise result
        return dict(result)

    with (
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch(
            "financial_news.monitor.compute_volume_stats",
            side_effect=compute_side_effect,
        ),
        patch("financial_news.monitor.snapshot"),
    ):
        mock_metrics.get_meter.return_value = mock_meter
        return run(
            mock_provider, tickers, snapshot_path=Path("/tmp/test_snapshot.json")
        )


def test_run_returns_0_on_full_success(mock_meter, mock_provider, good_stats):
    stats_map = {"NVDA": good_stats, "TSLA": good_stats}
    result = _run_with_mocks(mock_meter, mock_provider, ["NVDA", "TSLA"], stats_map)
    assert result == 0


def test_run_creates_one_gauge_per_spec_and_one_counter(
    mock_meter, mock_provider, good_stats
):
    _run_with_mocks(mock_meter, mock_provider, ["NVDA"], {"NVDA": good_stats})

    assert mock_meter.create_gauge.call_count == len(GAUGE_SPECS)
    assert mock_meter.create_counter.call_count == 1


def test_run_sets_all_gauges_for_each_ticker(mock_meter, mock_provider, good_stats):
    tickers = ["NVDA", "TSLA", "AMD"]
    _run_with_mocks(
        mock_meter,
        mock_provider,
        tickers,
        {t: good_stats for t in tickers},
    )
    gauge = mock_meter.create_gauge.return_value
    assert gauge.set.call_count == len(GAUGE_SPECS) * len(tickers)


def test_run_includes_classification_in_gauge_attrs(
    mock_meter, mock_provider, good_stats
):
    _run_with_mocks(mock_meter, mock_provider, ["NVDA"], {"NVDA": good_stats})
    gauge = mock_meter.create_gauge.return_value
    for call in gauge.set.call_args_list:
        attrs = call[0][1]
        assert "classification" in attrs
        assert attrs["classification"] == "normal"
        assert attrs["ticker"] == "NVDA"


def test_run_caps_infinite_z_score(mock_meter, mock_provider, good_stats):
    stats = {**good_stats, "z_score": float("inf"), "classification": "unusual"}
    _run_with_mocks(mock_meter, mock_provider, ["TSLA"], {"TSLA": stats})

    gauge = mock_meter.create_gauge.return_value
    for call in gauge.set.call_args_list:
        value = call[0][0]
        assert math.isfinite(value)
        assert value <= AnalysisConfig.z_score_cap


def test_run_returns_1_when_all_tickers_fail(mock_meter, mock_provider):
    def compute_side_effect(ticker):
        raise RuntimeError("API error")

    with (
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch(
            "financial_news.monitor.compute_volume_stats",
            side_effect=compute_side_effect,
        ),
    ):
        mock_metrics.get_meter.return_value = mock_meter
        result = run(mock_provider, ["NVDA", "TSLA"])

    assert result == 1
    assert mock_meter.create_counter.return_value.add.call_count == 2


def test_run_returns_1_on_partial_failure(mock_meter, mock_provider, good_stats):
    result = _run_with_mocks(
        mock_meter,
        mock_provider,
        ["NVDA", "FAIL", "TSLA"],
        {"NVDA": good_stats, "FAIL": RuntimeError("boom"), "TSLA": good_stats},
    )
    assert result == 1
    assert mock_meter.create_counter.return_value.add.call_count == 1


def test_run_always_flushes_and_shuts_down(mock_meter, mock_provider):
    def compute_side_effect(ticker):
        raise RuntimeError("always fails")

    with (
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch(
            "financial_news.monitor.compute_volume_stats",
            side_effect=compute_side_effect,
        ),
    ):
        mock_metrics.get_meter.return_value = mock_meter
        run(mock_provider, ["NVDA"])

    mock_provider.force_flush.assert_called_once()
    mock_provider.shutdown.assert_called_once()


def test_build_meter_provider_appends_v1_metrics_path(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com/")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")

    with (
        patch("financial_news.monitor.OTLPMetricExporter") as mock_exporter_cls,
        patch("financial_news.monitor.PeriodicExportingMetricReader"),
        patch("financial_news.monitor.MeterProvider") as mock_provider_cls,
    ):
        result = _build_meter_provider()

    mock_exporter_cls.assert_called_once_with(
        endpoint="https://otlp.example.com/v1/metrics",
        headers={"Authorization": "Basic abc123"},
    )
    assert result is mock_provider_cls.return_value


def test_main_passes_config_tickers_to_run(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic token")

    fake_cfg = MagicMock()
    fake_cfg.monitor.tickers = ["AAPL", "GOOG"]
    fake_cfg.monitor.snapshot_path = "/tmp/financial_news_snapshot.json"
    fake_provider = MagicMock()

    with (
        patch("financial_news.monitor.load_config", return_value=fake_cfg),
        patch(
            "financial_news.monitor._build_meter_provider",
            return_value=fake_provider,
        ),
        patch("financial_news.monitor.metrics"),
        patch("financial_news.monitor.run", return_value=0) as mock_run,
    ):
        result = main()

    mock_run.assert_called_once_with(
        fake_provider,
        ["AAPL", "GOOG"],
        fake_cfg.analysis.z_score_cap,
        Path("/tmp/financial_news_snapshot.json"),
    )
    assert result == 0


def test_run_sets_correct_gauge_values(mock_provider, good_stats):
    """Each gauge receives the right value from stats, not just any value."""
    meter = MagicMock()
    meter.create_counter.return_value = MagicMock()
    per_key_gauge: dict = {}

    def _create_gauge(otel_name, **_):
        m = MagicMock()
        per_key_gauge[otel_name] = m
        return m

    meter.create_gauge.side_effect = _create_gauge
    _run_with_mocks(meter, mock_provider, ["NVDA"], {"NVDA": good_stats})

    for key, otel_name, _ in GAUGE_SPECS:
        per_key_gauge[otel_name].set.assert_called_once_with(
            good_stats[key], {"ticker": "NVDA", "classification": "normal"}
        )


def test_run_error_counter_tagged_with_ticker(mock_meter, mock_provider, good_stats):
    """Error counter must record the specific failing ticker, not a blank attribute."""
    _run_with_mocks(
        mock_meter,
        mock_provider,
        ["NVDA", "FAIL", "TSLA"],
        {"NVDA": good_stats, "FAIL": RuntimeError("boom"), "TSLA": good_stats},
    )
    mock_meter.create_counter.return_value.add.assert_called_once_with(
        1, {"ticker": "FAIL"}
    )


def test_run_empty_tickers_returns_0_and_flushes(mock_meter, mock_provider):
    result = _run_with_mocks(mock_meter, mock_provider, [], {})
    assert result == 0
    mock_provider.force_flush.assert_called_once()
    mock_provider.shutdown.assert_called_once()


def test_build_meter_provider_without_trailing_slash(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")

    with (
        patch("financial_news.monitor.OTLPMetricExporter") as mock_exporter_cls,
        patch("financial_news.monitor.PeriodicExportingMetricReader"),
        patch("financial_news.monitor.MeterProvider"),
    ):
        _build_meter_provider()

    mock_exporter_cls.assert_called_once_with(
        endpoint="https://otlp.example.com/v1/metrics",
        headers={"Authorization": "Basic abc123"},
    )


def test_build_meter_provider_raises_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("GRAFANA_CLOUD_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")
    with pytest.raises(KeyError):
        _build_meter_provider()


def test_build_meter_provider_raises_when_auth_unset(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.delenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", raising=False)
    with pytest.raises(KeyError):
        _build_meter_provider()


def test_main_sets_meter_provider(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic token")

    fake_cfg = MagicMock()
    fake_cfg.monitor.tickers = ["AAPL"]
    fake_provider = MagicMock()

    with (
        patch("financial_news.monitor.load_config", return_value=fake_cfg),
        patch(
            "financial_news.monitor._build_meter_provider", return_value=fake_provider
        ),
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch("financial_news.monitor.run", return_value=0),
    ):
        main()

    mock_metrics.set_meter_provider.assert_called_once_with(fake_provider)


def test_run_writes_snapshot_for_successful_tickers(
    mock_meter, mock_provider, good_stats
):
    from pathlib import Path

    def compute_side_effect(ticker):
        if ticker == "FAIL":
            raise RuntimeError("boom")
        return dict(good_stats)

    snap_path = Path("/tmp/test_snapshot.json")
    with (
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch(
            "financial_news.monitor.compute_volume_stats",
            side_effect=compute_side_effect,
        ),
        patch("financial_news.monitor.snapshot") as mock_snapshot,
    ):
        mock_metrics.get_meter.return_value = mock_meter
        run(mock_provider, ["NVDA", "FAIL"], snapshot_path=snap_path)

    mock_snapshot.write.assert_called_once()
    written_stats, written_path = mock_snapshot.write.call_args[0]
    assert len(written_stats) == 1
    assert written_stats[0]["ticker"] == "NVDA"
    assert written_path == snap_path


def test_run_does_not_write_snapshot_when_all_tickers_fail(mock_meter, mock_provider):
    from pathlib import Path

    with (
        patch("financial_news.monitor.metrics") as mock_metrics,
        patch(
            "financial_news.monitor.compute_volume_stats",
            side_effect=RuntimeError("down"),
        ),
        patch("financial_news.monitor.snapshot") as mock_snapshot,
    ):
        mock_metrics.get_meter.return_value = mock_meter
        run(mock_provider, ["NVDA"], snapshot_path=Path("/tmp/test.json"))

    mock_snapshot.write.assert_not_called()


class TestGaugeSpecsContract:
    """Verify GAUGE_SPECS stays in sync with compute_volume_stats and monitor output."""

    def test_all_gauge_keys_exist_in_compute_volume_stats_output(self):
        """Catch renames: every stats_key in GAUGE_SPECS must be a real output key."""
        with patch("financial_news.analysis.fetch_news", return_value=[]):
            stats = analysis.compute_volume_stats("TEST")
        missing = {key for key, _, _ in GAUGE_SPECS} - stats.keys()
        assert not missing, (
            "GAUGE_SPECS references keys absent from compute_volume_stats "
            f"output: {missing}"
        )

    def test_gauge_specs_has_no_duplicate_keys(self):
        """Catch accidental double-registration of the same field."""
        keys = [key for key, _, _ in GAUGE_SPECS]
        dupes = [k for k in keys if keys.count(k) > 1]
        assert len(keys) == len(set(keys)), (
            f"Duplicate stats_key in GAUGE_SPECS: {dupes}"
        )

    def test_gauge_specs_has_no_duplicate_otel_names(self):
        """Catch two specs sharing an OTel metric name — causes silent overwrite."""
        otel_names = [name for _, name, _ in GAUGE_SPECS]
        dupes = [n for n in otel_names if otel_names.count(n) > 1]
        assert len(otel_names) == len(set(otel_names)), (
            f"Duplicate OTel metric name in GAUGE_SPECS: {dupes}"
        )

    def test_gauge_specs_descriptions_are_non_empty(self):
        """Empty descriptions produce undocumented metrics in Grafana."""
        for key, name, desc in GAUGE_SPECS:
            assert desc.strip(), (
                f"GAUGE_SPECS entry '{key}' (metric: {name!r}) has an empty description"
            )
