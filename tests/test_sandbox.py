"""Unit tests for the network sandbox launcher and startup check."""

import errno
import json
import socket
import sys
from pathlib import Path

import pytest

from financial_news import sandbox


def test_server_allowlist_is_finnhub_and_yahoo_only():
    assert sandbox.allowed_hosts("server", {}) == [
        "api.finnhub.io",
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
        "fc.yahoo.com",
    ]


def test_monitor_allowlist_adds_grafana_host_from_env():
    env = {"GRAFANA_CLOUD_OTLP_ENDPOINT": "https://otlp.grafana.net/otlp"}
    hosts = sandbox.allowed_hosts("monitor", env)
    assert hosts[-1] == "otlp.grafana.net"
    assert hosts[:-1] == sandbox.ALLOWED_HOSTS["monitor"]


def test_monitor_allowlist_without_grafana_env():
    assert sandbox.allowed_hosts("monitor", {}) == sandbox.ALLOWED_HOSTS["monitor"]


def test_build_settings_is_strict_and_blocks_local_ports():
    settings = sandbox.build_settings("server", {})
    assert settings["network"]["strictAllowlist"] is True
    assert settings["network"]["allowLocalBinding"] is False
    assert "~/.ssh" in settings["filesystem"]["denyRead"]


def test_build_settings_monitor_can_write_snapshot_dir():
    settings = sandbox.build_settings("monitor", {})
    snapshot_path = Path(sandbox.load_config().monitor.snapshot_path)
    assert str(snapshot_path.resolve().parent) in settings["filesystem"]["allowWrite"]


def test_child_environ_strips_llm_credentials_and_forces_offline_models():
    env = {"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y", "FINNHUB_API_KEY": "z"}
    child = sandbox.child_environ(env)
    assert "ANTHROPIC_API_KEY" not in child
    assert "OPENAI_API_KEY" not in child
    assert child["FINNHUB_API_KEY"] == "z"
    assert child["HF_HUB_OFFLINE"] == "1"


def _connect_raising(exc):
    def fake(*_args, **_kwargs):
        raise exc

    return fake


@pytest.mark.parametrize("code", [errno.EPERM, errno.EACCES, errno.ENETUNREACH])
def test_require_sandbox_passes_when_network_is_blocked(monkeypatch, code):
    monkeypatch.setattr(
        socket, "create_connection", _connect_raising(OSError(code, "blocked"))
    )
    sandbox.require_sandbox()


def test_require_sandbox_strips_llm_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(
        socket, "create_connection", _connect_raising(PermissionError(errno.EPERM, ""))
    )
    sandbox.require_sandbox()
    assert "ANTHROPIC_API_KEY" not in sandbox.os.environ


@pytest.mark.parametrize(
    "exc", [TimeoutError("timed out"), OSError(errno.EHOSTUNREACH, "unreachable")]
)
def test_require_sandbox_exits_when_network_is_open(monkeypatch, exc):
    monkeypatch.setattr(socket, "create_connection", _connect_raising(exc))
    with pytest.raises(SystemExit, match="Refusing to start"):
        sandbox.require_sandbox()


def test_require_sandbox_exits_when_probe_connects(monkeypatch):
    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Conn())
    with pytest.raises(SystemExit, match="Refusing to start"):
        sandbox.require_sandbox()


@pytest.mark.parametrize("argv", [[], ["briefing"], ["server", "extra"]])
def test_main_rejects_unknown_targets(argv):
    assert sandbox.main(argv) == 2


def test_main_fails_without_npx(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    assert sandbox.main(["server"]) == 1


def test_main_execs_pinned_srt_with_scrubbed_env(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/npx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    captured = {}

    def fake_execve(path, args, env):
        captured.update(path=path, args=args, env=env)

    monkeypatch.setattr(sandbox.os, "execve", fake_execve)
    sandbox.main(["server"])

    args = captured["args"]
    assert args[:3] == ["/usr/bin/npx", "-y", sandbox.SRT_PACKAGE]
    assert args[-3:] == [sys.executable, "-m", "financial_news.server"]
    settings_path = Path(args[args.index("--settings") + 1])
    assert settings_path.name == "financial-news-srt-server.json"
    settings = json.loads(settings_path.read_text())
    assert settings["network"]["allowedDomains"] == sandbox.ALLOWED_HOSTS["server"]
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_main_prefetch_model(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox, "prefetch_model", lambda: calls.append(True))
    assert sandbox.main(["prefetch-model"]) == 0
    assert calls == [True]


def test_prefetch_model_downloads_pinned_revisions(monkeypatch, capsys):
    pytest.importorskip("huggingface_hub")
    calls = []

    def fake_download(name, revision):
        calls.append((name, revision))
        return f"/cache/{name}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    sandbox.prefetch_model()
    assert calls == list(sandbox.APPROVED_SENTIMENT_MODELS.items())
