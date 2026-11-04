"""Tests for financial_news.telemetry."""

from unittest.mock import patch

import pytest

from financial_news.telemetry import build_meter_provider, grafana_configured


def test_build_meter_provider_appends_v1_metrics_path(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com/")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")

    with (
        patch("financial_news.telemetry.OTLPMetricExporter") as mock_exporter_cls,
        patch("financial_news.telemetry.PeriodicExportingMetricReader"),
        patch("financial_news.telemetry.MeterProvider") as mock_provider_cls,
    ):
        result = build_meter_provider("svc")

    mock_exporter_cls.assert_called_once_with(
        endpoint="https://otlp.example.com/v1/metrics",
        headers={"Authorization": "Basic abc123"},
    )
    assert result is mock_provider_cls.return_value


def test_build_meter_provider_without_trailing_slash(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")

    with (
        patch("financial_news.telemetry.OTLPMetricExporter") as mock_exporter_cls,
        patch("financial_news.telemetry.PeriodicExportingMetricReader"),
        patch("financial_news.telemetry.MeterProvider"),
    ):
        build_meter_provider("svc")

    mock_exporter_cls.assert_called_once_with(
        endpoint="https://otlp.example.com/v1/metrics",
        headers={"Authorization": "Basic abc123"},
    )


def test_build_meter_provider_raises_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("GRAFANA_CLOUD_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic abc123")
    with pytest.raises(KeyError):
        build_meter_provider("svc")


def test_build_meter_provider_raises_when_auth_unset(monkeypatch):
    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", "https://otlp.example.com")
    monkeypatch.delenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", raising=False)
    with pytest.raises(KeyError):
        build_meter_provider("svc")


@pytest.mark.parametrize(
    ("endpoint", "auth", "expected"),
    [
        ("https://otlp.example.com", "Basic abc", True),
        (None, "Basic abc", False),
        ("https://otlp.example.com", None, False),
        ("", "Basic abc", False),
    ],
)
def test_grafana_configured(monkeypatch, endpoint, auth, expected):
    for var, value in [
        ("GRAFANA_CLOUD_OTLP_ENDPOINT", endpoint),
        ("GRAFANA_CLOUD_BASIC_AUTH_HEADER", auth),
    ]:
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    assert grafana_configured() is expected
