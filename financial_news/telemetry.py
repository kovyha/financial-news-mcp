"""Grafana Cloud metrics connection shared by the monitor and the briefing.

Deterministic: no LLM imports, so the monitor (a deterministic entry point) can
use it. The connection comes from GRAFANA_CLOUD_OTLP_ENDPOINT and
GRAFANA_CLOUD_BASIC_AUTH_HEADER.
"""

import os

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

_ENDPOINT_VAR = "GRAFANA_CLOUD_OTLP_ENDPOINT"
_AUTH_VAR = "GRAFANA_CLOUD_BASIC_AUTH_HEADER"


def grafana_configured() -> bool:
    """True if both Grafana Cloud variables are set and non-empty."""
    return bool(os.environ.get(_ENDPOINT_VAR) and os.environ.get(_AUTH_VAR))


def build_meter_provider() -> MeterProvider:
    """Meter provider exporting to Grafana Cloud. Raises KeyError if unconfigured."""
    endpoint = os.environ[_ENDPOINT_VAR].rstrip("/") + "/v1/metrics"
    auth = os.environ[_AUTH_VAR]
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers={"Authorization": auth},
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5_000)
    return MeterProvider(metric_readers=[reader])
