import gzip
import logging
import logging.handlers
import shutil
from datetime import datetime
from pathlib import Path

from financial_news.config import load_config

_config = load_config()
LOG_DIR = Path(_config.logging.log_dir)
_LOG_MAX_BYTES = _config.logging.max_bytes
_ACTIVE_LOG_FILENAME = _config.logging.filename
_LOG_LEVEL = logging.DEBUG if _config.logging.level == "DEBUG" else logging.INFO


class _TimestampRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """Rolling file handler that names rolled files with their creation
    timestamp and gzip-compresses them immediately on rollover.

    Each rolled filename takes the form:
        <stem>_<YYYYMMDD_HHMMSS><ext>.gz
    where the timestamp reflects when the *current* file was opened,
    not when it was rolled off.
    """

    def __init__(
        self,
        filename: Path,
        max_bytes: int = _LOG_MAX_BYTES,
        created_at: datetime | None = None,
    ) -> None:
        super().__init__(str(filename), mode="a", encoding="utf-8")
        self.maxBytes = max_bytes
        self._created_at: datetime = (
            created_at if created_at is not None else datetime.now()
        )

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.stream is None:
            self.stream = self._open()
        msg = self.format(record) + "\n"
        self.stream.seek(0, 2)
        current_size = self.stream.tell()
        message_size = len(msg.encode("utf-8"))
        return current_size + message_size >= self.maxBytes

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record, ensuring writes are flushed to disk immediately."""
        try:
            super().emit(record)
            if self.stream:
                self.stream.flush()
        except Exception:
            self.handleError(record)

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        path = Path(self.baseFilename)
        dest = path.with_suffix(path.suffix + ".gz")

        with open(path, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        path.unlink()

        new_ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        stem_parts = path.stem.rsplit("_", 2)
        base_stem = "_".join(stem_parts[:-2]) if len(stem_parts) >= 3 else path.stem
        new_path = path.parent / f"{base_stem}_{new_ts}{path.suffix}"

        self.baseFilename = str(new_path)
        self._created_at = datetime.now()
        self.stream = self._open()


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("financial_news")
    if log.handlers:
        return log

    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    startup_ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    startup_dt = datetime.now()

    app_handler = _TimestampRotatingFileHandler(
        LOG_DIR / f"{_ACTIVE_LOG_FILENAME}_{startup_ts}.log",
        created_at=startup_dt,
    )
    app_handler.setLevel(_LOG_LEVEL)
    app_handler.setFormatter(fmt)

    error_handler = _TimestampRotatingFileHandler(
        LOG_DIR / f"{_ACTIVE_LOG_FILENAME}_{startup_ts}.error.log",
        created_at=startup_dt,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    log.setLevel(logging.DEBUG)
    log.addHandler(app_handler)
    log.addHandler(error_handler)

    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_LOG_LEVEL)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    return log


logger = _setup_logger()
