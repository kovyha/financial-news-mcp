"""SMTP email summary for the financial-news monitor + briefing run.

Credentials come from environment variables:
  SMTP_USER     — login username (usually the sender address)
  SMTP_PASSWORD — login password / app password

The [email] section of config.toml controls recipients, host, port, and an
optional smtp_from override. If the section is absent, send_run_summary() is
a no-op.

Email structure (primary → secondary):
  1. One-line classification summary: Unusual: X | Elevated: Y | Normal: Z
     (a ticker counts under whichever of news/price is more severe)
  2. Tickers grouped by classification, each showing both news and price
     z-scores, with a divergence tag when only one signal is flagged
  3. Analyst briefing (LLM reasoning, if available)
  4. Errors, if any
"""

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from financial_news.config import EmailConfig

logger = logging.getLogger(__name__)

_DIVIDER = "─" * 50
_SEVERITY = {"normal": 0, "elevated": 1, "unusual": 2}


def _price_classification(s: dict) -> str:
    price = s.get("price")
    return price.get("classification", "normal") if price else "normal"


def _combined_classification(s: dict) -> str:
    """The more severe of news and price classification for this ticker.

    A ticker with an unusual price move but normal news volume (or vice versa)
    must still surface as unusual — that divergence is exactly the signal this
    project exists to catch, and a news-only count would hide it entirely.
    """
    news_cls = s.get("classification", "normal")
    price_cls = _price_classification(s)
    return news_cls if _SEVERITY[news_cls] >= _SEVERITY[price_cls] else price_cls


def _divergence_tag(s: dict) -> str:
    """Flag when only one of the two signals is flagged elevated/unusual."""
    if "price" not in s:
        return ""
    news_flagged = s.get("classification", "normal") != "normal"
    price_flagged = _price_classification(s) != "normal"
    if price_flagged and not news_flagged:
        return "  [significant price move, no matching news anomaly]"
    if news_flagged and not price_flagged:
        return "  [news anomaly, no matching significant price move]"
    return ""


def _count_by_classification(ticker_stats: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"unusual": 0, "elevated": 0, "normal": 0}
    for s in ticker_stats:
        key = _combined_classification(s)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_subject(ticker_stats: list[dict], failed_tickers: list[str]) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
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
        group = [
            s for s in ticker_stats if _combined_classification(s) == classification
        ]
        if not group:
            continue
        lines.append(f"--- {label} ---")
        for s in group:
            price = s.get("price")
            price_part = (
                f"  price_move={price['return_pct']:>+7.2%} (z={price['z_score']:.2f})"
                if price
                else ""
            )
            lines.append(
                f"  {s['ticker']:<8}  z_news={s['z_score']:>6.2f}"
                f"  recent={s['recent_count']:>4}{price_part}{_divergence_tag(s)}"
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
