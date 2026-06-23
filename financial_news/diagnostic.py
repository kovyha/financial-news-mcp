"""
Diagnostic agent for financial-news-mcp.

This module investigates anomalous behavior: unexpected LLM output, bad Finnhub data,
or errors surfaced in the error log. It can be run:
  - Automatically after monitor failures (via .github/workflows/monitor.yaml)
  - Manually for urgent or complex incidents

The diagnostic agent reads error logs, produces a written diagnosis, and proposes
fixes where causes are identified. All changes require explicit human approval.
"""

import logging
import os
import sys
from datetime import date
from datetime import datetime as dt
from pathlib import Path

import anthropic

from financial_news.config import load_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are diagnosing errors in financial-news-mcp, an MCP server that"
    " fetches Finnhub news and computes 30-day EWM z-scores to detect"
    " unusual stock ticker activity.\n\n"
    "Key files: financial_news/analysis.py (z-score logic),"
    " financial_news/server.py (MCP tools),"
    " financial_news/config.py (thresholds),"
    " financial_news/monitor.py (daily watchlist runner).\n\n"
    "Read the relevant source files, identify the root cause,"
    " and propose a specific fix. If changes require human review"
    " before merging, say so."
)


def _analyze_with_llm(error_lines: list[str], project_root: Path) -> str:
    """Call Claude to reason about error lines, reading source files as needed."""
    client = anthropic.Anthropic()
    tools = [
        {
            "name": "read_file",
            "description": (
                "Read a source file from the financial-news-mcp project "
                "to understand the code that produced the errors."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path from the project root, "
                            "e.g. 'financial_news/analysis.py'"
                        ),
                    }
                },
                "required": ["path"],
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]
    error_text = "\n".join(error_lines)
    prompt = f"Error log entries from today:\n\n{error_text}"
    messages = [{"role": "user", "content": prompt}]
    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            return "\n".join(
                block.text for block in response.content if block.type == "text"
            )
        if response.stop_reason != "tool_use":
            return f"Analysis ended unexpectedly (stop_reason={response.stop_reason})."
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            raw_path = block.input.get("path", "")
            target = (project_root / raw_path).resolve()
            if not target.is_relative_to(project_root):
                content = "Error: path traversal not allowed."
            elif not target.exists():
                content = f"File not found: {raw_path}"
            else:
                try:
                    content = target.read_text(encoding="utf-8")
                except Exception as exc:
                    content = f"Error reading {raw_path}: {exc}"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )
        messages.append({"role": "user", "content": tool_results})


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


def diagnose(
    log_path: Path | None = None,
    target_date: date | None = None,
    use_llm: bool = True,
) -> str:
    """Run diagnostic check on error log.

    Args:
        log_path: Override path to error log (default: auto-detected from config)
        target_date: Date to check (default: today)
        use_llm: Whether to call Claude for root-cause analysis (default: True)

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

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        report_lines.append("LLM root-cause analysis:")
        report_lines.append("")
        try:
            analysis = _analyze_with_llm(error_lines[:20], _PROJECT_ROOT)
            report_lines.append(analysis)
        except Exception as exc:
            logger.exception("LLM analysis failed")
            report_lines.append(
                f"LLM analysis failed ({exc}). Review errors above manually."
            )
    else:
        report_lines.append("Set ANTHROPIC_API_KEY to enable LLM root-cause analysis.")

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
