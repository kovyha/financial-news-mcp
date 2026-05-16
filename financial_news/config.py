"""Configuration loader for financial-news-mcp.

Reads config.toml from the current working directory if present.
Falls back to built-in defaults when the file does not exist.
FINNHUB_API_KEY is intentionally excluded — secrets stay in the environment.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    filename: str = "financial_news"
    level: str = "INFO"  # INFO | DEBUG


@dataclass
class AnalysisConfig:
    baseline_days: int = 30
    threshold_elevated: float = 2.0
    threshold_unusual: float = 3.0
    z_score_cap: float = 99.0


_DEFAULT_MONITOR_TICKERS: list[str] = [
    "NVDA",
    "TSLA",
    "META",
    "AMD",
    "F",
    "BAC",
    "GME",
    "PLUG",
    "XOM",
    "UNH",
]


@dataclass
class MonitorConfig:
    tickers: list[str] = field(default_factory=lambda: list(_DEFAULT_MONITOR_TICKERS))
    snapshot_path: str = "/tmp/financial_news_snapshot.json"


@dataclass
class BriefingConfig:
    headline_days: int = 7
    max_headlines: int = 50


@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from *path*.

    Returns a Config with default values if the file does not exist.
    Raises ValueError for unrecognised keys or invalid values so
    misconfiguration is visible immediately on startup.
    """
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    log_data = data.get("logging", {})

    unknown = log_data.keys() - LoggingConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [logging] contains unrecognised keys: {sorted(unknown)}"
        )

    level = log_data.get("level", LoggingConfig.level).upper()
    if level not in ("INFO", "DEBUG"):
        raise ValueError(
            f"config.toml [logging] level must be INFO or DEBUG, got: {level!r}"
        )

    analysis_data = data.get("analysis", {})

    unknown = analysis_data.keys() - AnalysisConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [analysis] contains unrecognised keys: {sorted(unknown)}"
        )

    baseline_days = int(
        analysis_data.get("baseline_days", AnalysisConfig.baseline_days)
    )
    threshold_elevated = float(
        analysis_data.get("threshold_elevated", AnalysisConfig.threshold_elevated)
    )
    threshold_unusual = float(
        analysis_data.get("threshold_unusual", AnalysisConfig.threshold_unusual)
    )
    z_score_cap = float(analysis_data.get("z_score_cap", AnalysisConfig.z_score_cap))

    if baseline_days <= 0:
        raise ValueError(
            f"config.toml [analysis] baseline_days must be > 0, got: {baseline_days}"
        )
    if threshold_elevated >= threshold_unusual:
        raise ValueError(
            f"config.toml [analysis] threshold_elevated ({threshold_elevated}) "
            f"must be less than threshold_unusual ({threshold_unusual})"
        )
    if z_score_cap <= 0:
        raise ValueError(
            f"config.toml [analysis] z_score_cap must be > 0, got: {z_score_cap}"
        )

    monitor_data = data.get("monitor", {})

    unknown = monitor_data.keys() - MonitorConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [monitor] contains unrecognised keys: {sorted(unknown)}"
        )

    tickers = list(monitor_data.get("tickers", _DEFAULT_MONITOR_TICKERS))
    if not tickers:
        raise ValueError("config.toml [monitor] tickers must not be empty")

    snapshot_path = str(monitor_data.get("snapshot_path", MonitorConfig.snapshot_path))

    briefing_data = data.get("briefing", {})

    unknown = briefing_data.keys() - BriefingConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [briefing] contains unrecognised keys: {sorted(unknown)}"
        )

    headline_days = int(
        briefing_data.get("headline_days", BriefingConfig.headline_days)
    )
    max_headlines = int(
        briefing_data.get("max_headlines", BriefingConfig.max_headlines)
    )

    if headline_days <= 0:
        raise ValueError(
            f"config.toml [briefing] headline_days must be > 0, got: {headline_days}"
        )
    if max_headlines <= 0:
        raise ValueError(
            f"config.toml [briefing] max_headlines must be > 0, got: {max_headlines}"
        )

    return Config(
        logging=LoggingConfig(
            log_dir=log_data.get("log_dir", LoggingConfig.log_dir),
            max_bytes=log_data.get("max_bytes", LoggingConfig.max_bytes),
            filename=log_data.get("filename", LoggingConfig.filename),
            level=level,
        ),
        analysis=AnalysisConfig(
            baseline_days=baseline_days,
            threshold_elevated=threshold_elevated,
            threshold_unusual=threshold_unusual,
            z_score_cap=z_score_cap,
        ),
        monitor=MonitorConfig(tickers=tickers, snapshot_path=snapshot_path),
        briefing=BriefingConfig(
            headline_days=headline_days,
            max_headlines=max_headlines,
        ),
    )
