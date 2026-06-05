"""Configuration loader for financial-news-mcp.

Reads config.toml from the current working directory if present.
Falls back to built-in defaults when the file does not exist.
FINNHUB_API_KEY is intentionally excluded — secrets stay in the environment.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_EMAIL_KNOWN_KEYS = {"recipients", "smtp_host", "smtp_port", "smtp_from"}

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


_DEFAULT_SENTIMENT_LABELS: list[str] = ["positive", "negative", "neutral"]


@dataclass
class SentimentConfig:
    model_name: str = "ProsusAI/finbert"
    labels: list[str] = field(default_factory=lambda: list(_DEFAULT_SENTIMENT_LABELS))


@dataclass
class BriefingConfig:
    headline_days: int = 7
    max_headlines: int = 50
    confidence_threshold: float = 0.85
    prompt_headlines_min: int = 5
    prompt_headlines_max: int = 50


@dataclass
class EmailConfig:
    recipients: list[str]
    smtp_host: str
    smtp_port: int = 587
    smtp_from: str = ""


@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    email: EmailConfig | None = None


def _logging_config_from_env() -> LoggingConfig:
    """Override log level from LOG_LEVEL env var (INFO or DEBUG)."""
    level = os.environ.get("LOG_LEVEL", "").strip().upper()
    if level in ("INFO", "DEBUG"):
        return LoggingConfig(level=level)
    return LoggingConfig()


def _email_config_from_env() -> EmailConfig | None:
    """Build EmailConfig from environment variables, or return None if not configured.

    EMAIL_RECIPIENTS (comma-separated) and SMTP_HOST are required.
    SMTP_PORT defaults to 587; SMTP_FROM defaults to SMTP_USER.
    """
    recipients_raw = os.environ.get("EMAIL_RECIPIENTS", "").strip()
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    if not recipients_raw or not smtp_host:
        return None
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not recipients:
        return None
    smtp_port_raw = os.environ.get("SMTP_PORT", "").strip()
    smtp_port = int(smtp_port_raw) if smtp_port_raw else EmailConfig.smtp_port
    smtp_from = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", "")).strip()
    return EmailConfig(
        recipients=recipients,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_from=smtp_from,
    )


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from *path*.

    Returns a Config with default values if the file does not exist.
    Raises ValueError for unrecognised keys or invalid values so
    misconfiguration is visible immediately on startup.
    """
    if not path.exists():
        sentiment_model_env = os.environ.get("SENTIMENT_MODEL_NAME", "").strip()
        sentiment = (
            SentimentConfig(model_name=sentiment_model_env)
            if sentiment_model_env
            else SentimentConfig()
        )
        log_cfg = _logging_config_from_env()
        return Config(
            logging=log_cfg, email=_email_config_from_env(), sentiment=sentiment
        )

    with open(path, "rb") as f:
        data = tomllib.load(f)

    log_data = data.get("logging", {})

    unknown = log_data.keys() - LoggingConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [logging] contains unrecognised keys: {sorted(unknown)}"
        )

    level = (
        os.environ.get("LOG_LEVEL", "").strip().upper()
        or log_data.get("level", LoggingConfig.level).upper()
    )
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

    confidence_threshold = float(
        briefing_data.get("confidence_threshold", BriefingConfig.confidence_threshold)
    )
    prompt_headlines_min = int(
        briefing_data.get("prompt_headlines_min", BriefingConfig.prompt_headlines_min)
    )
    prompt_headlines_max = int(
        briefing_data.get("prompt_headlines_max", BriefingConfig.prompt_headlines_max)
    )

    if headline_days <= 0:
        raise ValueError(
            f"config.toml [briefing] headline_days must be > 0, got: {headline_days}"
        )
    if max_headlines <= 0:
        raise ValueError(
            f"config.toml [briefing] max_headlines must be > 0, got: {max_headlines}"
        )
    if not (0.0 < confidence_threshold <= 1.0):
        raise ValueError(
            "config.toml [briefing] confidence_threshold must be in (0, 1],"
            f" got: {confidence_threshold}"
        )
    if prompt_headlines_min <= 0:
        raise ValueError(
            "config.toml [briefing] prompt_headlines_min must be > 0,"
            f" got: {prompt_headlines_min}"
        )
    if prompt_headlines_max < prompt_headlines_min:
        raise ValueError(
            f"config.toml [briefing] prompt_headlines_max ({prompt_headlines_max})"
            f" must be >= prompt_headlines_min ({prompt_headlines_min})"
        )

    sentiment_data = data.get("sentiment", {})

    unknown = sentiment_data.keys() - SentimentConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"config.toml [sentiment] contains unrecognised keys: {sorted(unknown)}"
        )

    sentiment_model_name = str(
        sentiment_data.get("model_name", SentimentConfig.model_name)
    )
    if not sentiment_model_name:
        raise ValueError("config.toml [sentiment] model_name must not be empty")

    sentiment_labels = list(sentiment_data.get("labels", _DEFAULT_SENTIMENT_LABELS))
    if not sentiment_labels:
        raise ValueError("config.toml [sentiment] labels must not be empty")
    blank = [lbl for lbl in sentiment_labels if not str(lbl).strip()]
    if blank:
        raise ValueError("config.toml [sentiment] labels must not contain blank values")

    email_data = data.get("email")
    email_cfg: EmailConfig | None = (
        _email_config_from_env() if email_data is None else None
    )
    if email_data is not None:
        unknown = email_data.keys() - _EMAIL_KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"config.toml [email] contains unrecognised keys: {sorted(unknown)}"
            )
        recipients = list(email_data.get("recipients", []))
        if not recipients:
            raise ValueError("config.toml [email] recipients must not be empty")
        smtp_host = str(email_data.get("smtp_host", "")).strip()
        if not smtp_host:
            raise ValueError("config.toml [email] smtp_host is required")
        smtp_port = int(email_data.get("smtp_port", EmailConfig.smtp_port))
        if not (1 <= smtp_port <= 65535):
            raise ValueError(
                f"config.toml [email] smtp_port must be 1–65535, got: {smtp_port}"
            )
        smtp_from = str(email_data.get("smtp_from", ""))
        email_cfg = EmailConfig(
            recipients=recipients,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_from=smtp_from,
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
            confidence_threshold=confidence_threshold,
            prompt_headlines_min=prompt_headlines_min,
            prompt_headlines_max=prompt_headlines_max,
        ),
        sentiment=SentimentConfig(
            model_name=sentiment_model_name, labels=sentiment_labels
        ),
        email=email_cfg,
    )
