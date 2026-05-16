"""Pass monitor stats to the briefing step within the same workflow run.

The monitor writes a snapshot after its OTel export; the briefing reads it
instead of re-fetching the same data from Finnhub.  Both steps run in the
same GitHub Actions job, so they share /tmp and the snapshot is always fresh.

The date check guards against stale files left over from a previous local run.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


def write(stats: list[dict], path: Path) -> None:
    """Atomically write stats to a dated snapshot file."""
    payload = {"date": date.today().isoformat(), "stats": stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    logger.info("snapshot written path=%s tickers=%d", path, len(stats))


def read(path: Path) -> list[dict] | None:
    """Return stats if the snapshot is from today, else None."""
    if not path.exists():
        logger.info("snapshot not found path=%s", path)
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("snapshot corrupt, ignoring path=%s: %s", path, exc)
        return None
    if payload.get("date") != date.today().isoformat():
        logger.info("snapshot date mismatch path=%s", path)
        return None
    stats = payload.get("stats")
    if stats is None:
        logger.warning("snapshot missing 'stats' key path=%s", path)
        return None
    logger.info("snapshot loaded path=%s tickers=%d", path, len(stats))
    return stats
