"""Daily monitoring agent for financial-news-mcp.

Fetches news volume stats for a configured watchlist of tickers, then pushes
z-score, recent count, and EWM mean as OpenTelemetry metrics to Grafana Cloud.

Run via GitHub Actions (monitor.yaml) or manually:
    FINNHUB_API_KEY=<key> \\
    GRAFANA_CLOUD_OTLP_ENDPOINT=<url> \\
    GRAFANA_CLOUD_BASIC_AUTH_HEADER="Basic <token>" \\
    uv run python -m financial_news.monitor
"""

import logging
import os
import sys

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from financial_news.analysis import GAUGE_SPECS, compute_volume_stats
from financial_news.config import load_config

logger = logging.getLogger(__name__)

# inf z-scores (zero-baseline spike) are capped so OTel gauges stay finite.
Z_SCORE_CAP = 99.0


def _build_meter_provider() -> MeterProvider:
    endpoint = os.environ["GRAFANA_CLOUD_OTLP_ENDPOINT"].rstrip("/") + "/v1/metrics"
    auth = os.environ["GRAFANA_CLOUD_BASIC_AUTH_HEADER"]
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers={"Authorization": auth},
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5_000)
    return MeterProvider(metric_readers=[reader])


def run(provider: MeterProvider, tickers: list[str]) -> int:
    meter = metrics.get_meter("financial_news.monitor")

    gauges = {
        key: meter.create_gauge(name, description=desc)
        for key, name, desc in GAUGE_SPECS
    }
    error_counter = meter.create_counter(
        "financial_news.monitor_errors",
        description="Finnhub API errors during monitoring run",
    )

    errors = 0
    for ticker in tickers:
        try:
            stats = compute_volume_stats(ticker)
            stats["z_score"] = min(stats["z_score"], Z_SCORE_CAP)
            attrs = {"ticker": ticker, "classification": stats["classification"]}
            for key, gauge in gauges.items():
                gauge.set(stats[key], attrs)
            logger.info(
                "ticker=%s recent=%d z=%.2f classification=%s",
                ticker,
                stats["recent_count"],
                stats["z_score"],
                stats["classification"],
            )
        except Exception:
            logger.exception("ticker=%s failed", ticker)
            error_counter.add(1, {"ticker": ticker})
            errors += 1

    provider.force_flush()
    provider.shutdown()
    return 1 if errors else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    cfg = load_config()
    provider = _build_meter_provider()
    metrics.set_meter_provider(provider)
    return run(provider, cfg.monitor.tickers)


if __name__ == "__main__":
    sys.exit(main())
