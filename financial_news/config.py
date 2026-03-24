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
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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

    return Config(
        logging=LoggingConfig(
            log_dir=log_data.get("log_dir", LoggingConfig.log_dir),
            max_bytes=log_data.get("max_bytes", LoggingConfig.max_bytes),
            filename=log_data.get("filename", LoggingConfig.filename),
            level=level,
        )
    )
