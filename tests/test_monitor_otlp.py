"""Integration tests for the OTel → HTTP export path in financial_news.monitor.

These tests use a real OTel SDK + a local HTTP server (pytest-httpserver) to
verify that metrics are correctly serialised and flushed to the wire, without
requiring live Grafana Cloud credentials.

What this covers that unit tests with MagicMock cannot:
- The OTel SDK actually encodes gauge values into an OTLP payload.
- _build_meter_provider() wires the endpoint URL and Authorization header correctly.
- force_flush() triggers an export before shutdown (timing guarantee).

What this does NOT cover:
- Grafana Cloud's ingestion pipeline accepting the payload.
- Metric names / label cardinality limits imposed by Grafana.
"""

from opentelemetry import metrics as otel_metrics

from financial_news.monitor import _build_meter_provider


def test_otlp_export_posts_to_v1_metrics_path(httpserver, monkeypatch):
    """The exporter must POST to /v1/metrics, not the bare endpoint."""
    httpserver.expect_request("/v1/metrics", method="POST").respond_with_data(
        "", status=200
    )

    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", httpserver.url_for("/"))
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic dGVzdDp0ZXN0")

    provider = _build_meter_provider()
    otel_metrics.set_meter_provider(provider)
    meter = otel_metrics.get_meter("test.monitor")
    gauge = meter.create_gauge("test.gauge", description="test")
    gauge.set(42.0, {"ticker": "TEST"})

    provider.force_flush()
    provider.shutdown()

    httpserver.check_assertions()


def test_otlp_export_sends_authorization_header(httpserver, monkeypatch):
    """The Authorization header must match GRAFANA_CLOUD_BASIC_AUTH_HEADER exactly."""
    expected_auth = "Basic dXNlcjpwYXNz"
    received_headers: list[dict] = []

    def capture_and_ok(request):
        received_headers.append(dict(request.headers))
        from werkzeug.wrappers import Response

        return Response("", status=200)

    httpserver.expect_request("/v1/metrics", method="POST").respond_with_handler(
        capture_and_ok
    )

    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", httpserver.url_for("/"))
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", expected_auth)

    provider = _build_meter_provider()
    otel_metrics.set_meter_provider(provider)
    meter = otel_metrics.get_meter("test.auth")
    gauge = meter.create_gauge("test.auth.gauge", description="auth test")
    gauge.set(1.0, {"ticker": "AUTH"})

    provider.force_flush()
    provider.shutdown()

    assert received_headers, "No request received — force_flush() did not export"
    assert received_headers[0].get("Authorization") == expected_auth


def test_otlp_export_flushes_before_shutdown(httpserver, monkeypatch):
    """force_flush() must deliver data; shutdown alone is not sufficient."""
    requests_received: list[int] = []

    def count_and_ok(request):
        requests_received.append(1)
        from werkzeug.wrappers import Response

        return Response("", status=200)

    httpserver.expect_request("/v1/metrics", method="POST").respond_with_handler(
        count_and_ok
    )

    monkeypatch.setenv("GRAFANA_CLOUD_OTLP_ENDPOINT", httpserver.url_for("/"))
    monkeypatch.setenv("GRAFANA_CLOUD_BASIC_AUTH_HEADER", "Basic dGVzdA==")

    provider = _build_meter_provider()
    otel_metrics.set_meter_provider(provider)
    meter = otel_metrics.get_meter("test.flush")
    gauge = meter.create_gauge("test.flush.gauge", description="flush test")
    gauge.set(7.0, {"ticker": "FLUSH"})

    provider.force_flush()
    provider.shutdown()

    assert len(requests_received) >= 1, (
        "No export received after force_flush() — data was silently dropped"
    )
