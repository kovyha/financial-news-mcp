"""Tests for financial_news.email_report."""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from financial_news.config import EmailConfig
from financial_news.email_report import (
    _build_body,
    _build_subject,
    _count_by_classification,
    send_run_summary,
)


@pytest.fixture()
def cfg():
    return EmailConfig(
        recipients=["a@example.com", "b@example.com"],
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_from="monitor@example.com",
    )


@pytest.fixture()
def sample_stats():
    return [
        {
            "ticker": "NVDA",
            "classification": "unusual",
            "z_score": 4.5,
            "recent_count": 20,
        },
        {
            "ticker": "TSLA",
            "classification": "elevated",
            "z_score": 2.3,
            "recent_count": 8,
        },
        {
            "ticker": "META",
            "classification": "normal",
            "z_score": 0.5,
            "recent_count": 3,
        },
    ]


# --- _count_by_classification ---


def test_count_by_classification_correct(sample_stats):
    counts = _count_by_classification(sample_stats)
    assert counts["unusual"] == 1
    assert counts["elevated"] == 1
    assert counts["normal"] == 1


def test_count_by_classification_all_normal():
    stats = [{"classification": "normal"}, {"classification": "normal"}]
    counts = _count_by_classification(stats)
    assert counts["unusual"] == 0
    assert counts["elevated"] == 0
    assert counts["normal"] == 2


def test_count_by_classification_empty():
    counts = _count_by_classification([])
    assert counts == {"unusual": 0, "elevated": 0, "normal": 0}


# --- _build_subject ---


def test_subject_contains_classification_counts(sample_stats):
    subj = _build_subject(sample_stats, [])
    assert "Unusual: 1" in subj
    assert "Elevated: 1" in subj
    assert "Normal: 1" in subj


def test_subject_contains_today_date(sample_stats):
    from datetime import date

    assert date.today().isoformat() in _build_subject(sample_stats, [])


def test_subject_no_error_tag_when_no_failures(sample_stats):
    assert "error" not in _build_subject(sample_stats, []).lower()


def test_subject_error_tag_singular(sample_stats):
    subj = _build_subject(sample_stats, ["FAIL"])
    assert "[1 error]" in subj


def test_subject_error_tag_plural(sample_stats):
    subj = _build_subject(sample_stats, ["FAIL1", "FAIL2"])
    assert "[2 errors]" in subj


# --- _build_body ---


def test_body_one_line_summary(sample_stats):
    body = _build_body(sample_stats, [])
    first_line = body.splitlines()[0]
    assert "Unusual: 1" in first_line
    assert "Elevated: 1" in first_line
    assert "Normal: 1" in first_line


def test_body_errors_in_summary_line(sample_stats):
    body = _build_body(sample_stats, ["GME"])
    first_line = body.splitlines()[0]
    assert "Errors: 1" in first_line


def test_body_unusual_section_appears_first(sample_stats):
    body = _build_body(sample_stats, [])
    assert body.index("UNUSUAL") < body.index("ELEVATED") < body.index("NORMAL")


def test_body_contains_ticker_under_correct_section(sample_stats):
    body = _build_body(sample_stats, [])
    unusual_idx = body.index("UNUSUAL")
    elevated_idx = body.index("ELEVATED")
    nvda_idx = body.index("NVDA")
    tsla_idx = body.index("TSLA")
    assert unusual_idx < nvda_idx < elevated_idx
    assert elevated_idx < tsla_idx


def test_body_z_score_and_count_present(sample_stats):
    body = _build_body(sample_stats, [])
    assert "4.50" in body
    assert "20" in body


def test_body_briefing_text_included():
    stats = [
        {"ticker": "X", "classification": "normal", "z_score": 0.1, "recent_count": 1}
    ]
    body = _build_body(stats, [], briefing_text="X is quiet due to low volume.")
    assert "ANALYST BRIEFING" in body
    assert "X is quiet due to low volume." in body


def test_body_briefing_section_after_ticker_groups(sample_stats):
    body = _build_body(sample_stats, [], briefing_text="Some analysis here.")
    assert body.index("NORMAL") < body.index("ANALYST BRIEFING")


def test_body_no_briefing_section_when_text_none(sample_stats):
    body = _build_body(sample_stats, [], briefing_text=None)
    assert "ANALYST BRIEFING" not in body


def test_body_errors_section_last(sample_stats):
    body = _build_body(sample_stats, ["FAIL"], briefing_text="Analysis.")
    assert body.index("ANALYST BRIEFING") < body.index("ERRORS")


def test_body_failed_tickers_listed(sample_stats):
    body = _build_body(sample_stats, ["AMD", "GME"])
    assert "AMD" in body
    assert "GME" in body
    assert "ERRORS" in body


def test_body_no_errors_section_when_no_failures(sample_stats):
    assert "ERRORS" not in _build_body(sample_stats, [])


def test_body_skips_missing_classification_sections():
    stats = [
        {
            "ticker": "NVDA",
            "classification": "unusual",
            "z_score": 5.0,
            "recent_count": 10,
        }
    ]
    body = _build_body(stats, [])
    assert "UNUSUAL" in body
    assert "ELEVATED" not in body
    assert "NORMAL" not in body


# --- send_run_summary ---


def _make_smtp_mock():
    smtp_instance = MagicMock()
    smtp_cm = MagicMock()
    smtp_cm.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_cm.__exit__ = MagicMock(return_value=False)
    return smtp_cm, smtp_instance


def test_send_uses_configured_host_and_port(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    smtp_cm, _ = _make_smtp_mock()

    smtp_patch = patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm)
    with smtp_patch as mock_smtp:
        send_run_summary(cfg, sample_stats, [])

    mock_smtp.assert_called_once_with("smtp.example.com", 587)


def test_send_calls_starttls_and_login(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(cfg, sample_stats, [])

    smtp_instance.starttls.assert_called_once()
    smtp_instance.login.assert_called_once_with("user@example.com", "secret")


def test_send_uses_smtp_from_when_set(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(cfg, sample_stats, [])

    sent_msg = smtp_instance.send_message.call_args[0][0]
    assert sent_msg["From"] == "monitor@example.com"


def test_send_falls_back_to_smtp_user_when_smtp_from_empty(sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    cfg_no_from = EmailConfig(
        recipients=["a@example.com"],
        smtp_host="smtp.example.com",
        smtp_from="",
    )
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(cfg_no_from, sample_stats, [])

    sent_msg = smtp_instance.send_message.call_args[0][0]
    assert sent_msg["From"] == "user@example.com"


def test_send_addresses_all_recipients(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(cfg, sample_stats, [])

    sent_msg = smtp_instance.send_message.call_args[0][0]
    assert "a@example.com" in sent_msg["To"]
    assert "b@example.com" in sent_msg["To"]


def test_send_passes_briefing_text_to_body(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(
            cfg, sample_stats, [], briefing_text="NVDA driven by GPU demand."
        )

    sent_msg = smtp_instance.send_message.call_args[0][0]
    assert "NVDA driven by GPU demand." in sent_msg.get_content()


def test_send_does_not_raise_on_smtp_error(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    with patch(
        "financial_news.email_report.smtplib.SMTP",
        side_effect=smtplib.SMTPException("connection refused"),
    ):
        send_run_summary(cfg, sample_stats, [])  # must not raise


def test_send_logs_error_on_smtp_failure(cfg, sample_stats, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    with patch(
        "financial_news.email_report.smtplib.SMTP",
        side_effect=smtplib.SMTPException("boom"),
    ):
        with caplog.at_level(logging.ERROR, logger="financial_news.email_report"):
            send_run_summary(cfg, sample_stats, [])

    assert any("failed to send" in r.message for r in caplog.records)


def test_send_subject_reflects_classification_counts(cfg, sample_stats, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    smtp_cm, smtp_instance = _make_smtp_mock()

    with patch("financial_news.email_report.smtplib.SMTP", return_value=smtp_cm):
        send_run_summary(cfg, sample_stats, [])

    sent_msg = smtp_instance.send_message.call_args[0][0]
    assert "Unusual: 1" in sent_msg["Subject"]
    assert "Elevated: 1" in sent_msg["Subject"]
