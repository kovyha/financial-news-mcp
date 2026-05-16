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
from pathlib import Path

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# log_setup must be imported before analysis so handlers are attached
# to the financial_news logger before analysis module-level code runs.
from financial_news import (
    log_setup,  # noqa: F401
    snapshot,
)
from financial_news.analysis import GAUGE_SPECS, compute_volume_stats
from financial_news.config import load_config

logger = logging.getLogger(__name__)


def _build_meter_provider() -> MeterProvider:
    endpoint = os.environ["GRAFANA_CLOUD_OTLP_ENDPOINT"].rstrip("/") + "/v1/metrics"
    auth = os.environ["GRAFANA_CLOUD_BASIC_AUTH_HEADER"]
    exporter = OTLPMetricExporter(
        endpoint=endpoint,
        headers={"Authorization": auth},
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5_000)
    return MeterProvider(metric_readers=[reader])


def run(
    provider: MeterProvider,
    tickers: list[str],
    z_score_cap: float = 99.0,
    snapshot_path: Path = Path("/tmp/financial_news_snapshot.json"),
) -> int:
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
    snapshot_stats: list[dict] = []
    for ticker in tickers:
        try:
            stats = compute_volume_stats(ticker)
            stats["z_score"] = min(stats["z_score"], z_score_cap)
            stats["ticker"] = ticker
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
            snapshot_stats.append(stats)
        except Exception:
            logger.exception("ticker=%s failed", ticker)
            error_counter.add(1, {"ticker": ticker})
            errors += 1

    if snapshot_stats:
        snapshot.write(snapshot_stats, snapshot_path)

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
    return run(
        provider,
        cfg.monitor.tickers,
        cfg.analysis.z_score_cap,
        Path(cfg.monitor.snapshot_path),
    )


if __name__ == "__main__":
    sys.exit(main())
