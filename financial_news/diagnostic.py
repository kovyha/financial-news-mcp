"""
Diagnostic agent for financial-news-mcp.

This module investigates anomalous behavior: unexpected LLM output, bad Finnhub data,
or errors surfaced in the error log. It can be run:
  - Automatically on a schedule (via .github/workflows/diagnostic.yaml)
  - Manually for urgent or complex incidents

The diagnostic agent reads error logs, produces a written diagnosis, and proposes
fixes where causes are identified. All changes require explicit human approval.
"""

import logging
import sys
from datetime import date
from datetime import datetime as dt
from pathlib import Path

from financial_news.config import load_config

logger = logging.getLogger(__name__)


def _find_active_error_log(log_dir: Path, filename_stem: str) -> Path | None:
    """Find the most recently modified active error log.

    Args:
        log_dir: Directory containing log files
        filename_stem: Base filename (e.g., "financial_news")

    Returns:
        Path to the most recent *.error.log file, or None if not found.
    """
    candidates = sorted(
        log_dir.glob(f"{filename_stem}*.error.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_error_log_for_date(log_path: Path, target_date: date) -> list[str]:
    """Read error log and extract lines for a specific date.

    Args:
        log_path: Path to error log file (e.g., logs/financial_news.error.log)
        target_date: Date to filter (DD-MM-YYYY format in logs)

    Returns:
        List of error log lines matching the target date.
    """
    if not log_path.exists():
        logger.warning("Error log not found at %s", log_path)
        return []

    target_str = target_date.strftime("%d-%m-%Y")
    matching_lines = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if target_str in line:
                    matching_lines.append(line.rstrip())
    except Exception as e:
        logger.error("Failed to read error log: %s", e)
        return []

    return matching_lines


def diagnose(log_path: Path | None = None, target_date: date | None = None) -> str:
    """Run diagnostic check on error log.

    Args:
        log_path: Override path to error log (default: auto-detected from config)
        target_date: Date to check (default: today)

    Returns:
        Diagnostic report as a string.
    """
    config = load_config()

    if log_path is None:
        log_dir = Path(config.logging.log_dir)
        log_path = _find_active_error_log(log_dir, config.logging.filename)
        if log_path is None:
            return (
                f"✅ Diagnostic report for {date.today().isoformat()}\n"
                f"No error log found in {log_dir}\n"
                f"Log file appears healthy."
            )

    if target_date is None:
        target_date = date.today()

    logger.info("Running diagnostic check for %s", target_date.isoformat())

    # Read error log
    error_lines = read_error_log_for_date(log_path, target_date)

    if not error_lines:
        report = (
            f"✅ Diagnostic report for {target_date.isoformat()}\n"
            f"No errors found in {log_path}\n"
            f"Log file appears healthy."
        )
        logger.info(report)
        return report

    # Build report
    report_lines = [
        f"🔍 Diagnostic report for {target_date.isoformat()}",
        f"Checked: {log_path}",
        f"Errors found: {len(error_lines)}",
        "",
        "Error log entries:",
    ]
    report_lines.extend(f"  {line}" for line in error_lines[:20])  # Limit to 20 lines

    if len(error_lines) > 20:
        report_lines.append(f"  ... and {len(error_lines) - 20} more errors")

    report_lines.append("")
    report_lines.append(f"Checked at: {dt.now().isoformat()}")
    report_lines.append("")
    report_lines.append("Next steps:")
    report_lines.append("  1. Review the errors above")
    report_lines.append("  2. Identify the root cause from the error messages")
    report_lines.append("  3. Check relevant code sections (financial_news/server.py)")
    report_lines.append("  4. Propose a fix if the cause is identified")
    report_lines.append("  5. Request human review before merging any changes")

    report = "\n".join(report_lines)
    logger.warning("Errors detected: %s", report)
    return report


def main() -> int:
    """Entry point for diagnostic agent.

    Returns:
        0 if no errors found, 1 if errors were found
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    report = diagnose()
    print(report)

    # Return exit code based on whether errors were found
    config = load_config()
    log_dir = Path(config.logging.log_dir)
    log_path = _find_active_error_log(log_dir, config.logging.filename)
    if log_path is None:
        return 0
    error_lines = read_error_log_for_date(log_path, date.today())

    return 1 if error_lines else 0


if __name__ == "__main__":
    sys.exit(main())
