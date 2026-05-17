"""SMTP email summary for the financial-news monitor + briefing run.

Credentials come from environment variables:
  SMTP_USER     — login username (usually the sender address)
  SMTP_PASSWORD — login password / app password

The [email] section of config.toml controls recipients, host, port, and an
optional smtp_from override. If the section is absent, send_run_summary() is
a no-op.

Email structure (primary → secondary):
  1. One-line classification summary: Unusual: X | Elevated: Y | Normal: Z
  2. Tickers grouped by classification
  3. Analyst briefing (LLM reasoning, if available)
  4. Errors, if any
"""

import logging
import os
import smtplib
from datetime import date
from email.message import EmailMessage

from financial_news.config import EmailConfig

logger = logging.getLogger(__name__)

_DIVIDER = "─" * 50


def _count_by_classification(ticker_stats: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"unusual": 0, "elevated": 0, "normal": 0}
    for s in ticker_stats:
        key = s.get("classification", "normal")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_subject(ticker_stats: list[dict], failed_tickers: list[str]) -> str:
    today = date.today().isoformat()
    c = _count_by_classification(ticker_stats)
    summary = (
        f"Unusual: {c['unusual']}, Elevated: {c['elevated']}, Normal: {c['normal']}"
    )
    error_tag = ""
    if failed_tickers:
        n = len(failed_tickers)
        error_tag = f" [{n} error{'s' if n > 1 else ''}]"
    return f"Financial News Monitor — {today} — {summary}{error_tag}"


def _build_body(
    ticker_stats: list[dict],
    failed_tickers: list[str],
    briefing_text: str | None = None,
) -> str:
    c = _count_by_classification(ticker_stats)
    lines: list[str] = []

    # One-line classification summary
    summary = (
        f"Unusual: {c['unusual']}  |  Elevated: {c['elevated']}"
        f"  |  Normal: {c['normal']}"
    )
    if failed_tickers:
        summary += f"  |  Errors: {len(failed_tickers)}"
    lines.append(summary)
    lines.append("")

    # Tickers grouped by classification (Unusual first)
    for classification, label in [
        ("unusual", "UNUSUAL"),
        ("elevated", "ELEVATED"),
        ("normal", "NORMAL"),
    ]:
        group = [s for s in ticker_stats if s.get("classification") == classification]
        if not group:
            continue
        lines.append(f"--- {label} ---")
        for s in group:
            lines.append(
                f"  {s['ticker']:<8}  z={s['z_score']:>6.2f}"
                f"  recent={s['recent_count']:>4}"
            )
        lines.append("")

    # LLM analyst briefing (primary purpose when available)
    if briefing_text and briefing_text.strip():
        lines.append(_DIVIDER)
        lines.append("ANALYST BRIEFING")
        lines.append(_DIVIDER)
        lines.append(briefing_text.strip())
        lines.append("")

    # Errors (secondary)
    if failed_tickers:
        lines.append(_DIVIDER)
        lines.append(f"ERRORS — {len(failed_tickers)} ticker(s) failed")
        lines.append(_DIVIDER)
        for t in failed_tickers:
            lines.append(f"  - {t}")

    return "\n".join(lines)


def send_run_summary(
    cfg: EmailConfig,
    ticker_stats: list[dict],
    failed_tickers: list[str],
    briefing_text: str | None = None,
) -> None:
    """Send a monitor/briefing run summary via SMTP. Logs and returns on any error."""
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    sender = cfg.smtp_from or smtp_user

    msg = EmailMessage()
    msg["Subject"] = _build_subject(ticker_stats, failed_tickers)
    msg["From"] = sender
    msg["To"] = ", ".join(cfg.recipients)
    msg.set_content(_build_body(ticker_stats, failed_tickers, briefing_text))

    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        logger.info("monitor summary email sent to %s", cfg.recipients)
    except Exception:
        logger.exception("failed to send monitor summary email")
