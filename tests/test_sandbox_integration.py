"""Tries real ways out of the sandbox and checks each one is blocked.

Runs the actual sandbox runtime (srt), so it needs Node.js, network access to npm
and the allowlisted hosts, and on Linux bubblewrap, socat and ripgrep. Skipped unless
RUN_SANDBOX_TESTS=1 (CI runs it in its own job).
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from financial_news import sandbox

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SANDBOX_TESTS") != "1", reason="set RUN_SANDBOX_TESTS=1"
)

_PROBE = textwrap.dedent(
    """
    import json, socket, subprocess, sys, urllib.error, urllib.request

    LISTENER_PORT = int(sys.argv[1])

    def blocked(fn):
        try:
            fn()
        except urllib.error.HTTPError:
            return False  # The server answered, so the request got out.
        except Exception:
            return True
        return False

    def http(url):
        return lambda: urllib.request.urlopen(url, timeout=10)

    def curl_cffi(url, **kwargs):
        from curl_cffi import requests
        return lambda: requests.get(url, timeout=10, **kwargs)

    def curl(*args):
        cmd = ["curl", "-sS", "-m", "10", "-o", "/dev/null", *args]
        return lambda: subprocess.run(cmd, check=True, capture_output=True)

    def sock(host, port):
        return lambda: socket.create_connection((host, port), timeout=5).close()

    def reachable(url):
        try:
            urllib.request.urlopen(url, timeout=15)
        except urllib.error.HTTPError:
            pass  # Any HTTP response means the host was reached.
        return True

    print(json.dumps({
        "openai via urllib": blocked(http("https://api.openai.com/v1/models")),
        "anthropic via urllib": blocked(http("https://api.anthropic.com")),
        "unknown provider": blocked(http("https://api.mistral.ai")),
        "hugging face": blocked(http("https://huggingface.co")),
        "openai via curl_cffi": blocked(curl_cffi("https://api.openai.com")),
        "curl_cffi with proxy off": blocked(
            curl_cffi("https://api.openai.com", proxies={"https": "", "http": ""})
        ),
        "curl subprocess": blocked(curl("https://api.openai.com")),
        "curl ignoring proxy": blocked(curl("--noproxy", "*", "https://api.openai.com")),
        "raw socket to IP": blocked(sock("1.1.1.1", 443)),
        "localhost model server": blocked(sock("127.0.0.1", LISTENER_PORT)),
        "finnhub reachable": reachable("https://api.finnhub.io/api/v1/"),
    }))
    """
)


def _run_in_sandbox(tmp_path: Path, target: str, *command: str):
    settings = tmp_path / "srt.json"
    settings.write_text(json.dumps(sandbox.build_settings(target, dict(os.environ))))
    return subprocess.run(
        sandbox.srt_command(shutil.which("npx"), settings, *command),
        capture_output=True,
        text=True,
        timeout=300,
        env=sandbox.child_environ(dict(os.environ)),
    )


def _probe(tmp_path: Path, *wrapper: str) -> dict[str, bool]:
    """Run the probe, with a real listener on this machine standing in for a
    local model server."""
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = str(listener.getsockname()[1])
        command = [sys.executable, "-I", str(probe), port]
        if wrapper:
            result = _run_in_sandbox(tmp_path, wrapper[0], *command)
        else:
            result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_probe_detects_open_routes_without_sandbox(tmp_path):
    """Control: unsandboxed, the routes are open, so the probe isn't vacuous."""
    outcomes = _probe(tmp_path)
    escape_routes = {k: v for k, v in outcomes.items() if k != "finnhub reachable"}
    assert not any(escape_routes.values()), outcomes


@pytest.mark.parametrize("target", ["server", "monitor"])
def test_every_escape_route_is_blocked(tmp_path, target):
    outcomes = _probe(tmp_path, target)
    assert all(outcomes.values()), outcomes


def test_require_sandbox_passes_inside_sandbox(tmp_path):
    # srt runs the command through a shell, so pass code as a file, not via -c.
    check = tmp_path / "check.py"
    check.write_text(
        "from financial_news.sandbox import require_sandbox\nrequire_sandbox()\n"
    )
    result = _run_in_sandbox(tmp_path, "server", sys.executable, str(check))
    assert result.returncode == 0, result.stderr


def test_server_refuses_to_start_unsandboxed():
    result = subprocess.run(
        [sys.executable, "-m", "financial_news.server"],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 1
    assert "Refusing to start outside the network sandbox" in result.stderr
