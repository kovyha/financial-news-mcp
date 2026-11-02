"""Network sandbox for the deterministic entry points (MCP server and monitor).

These processes must never reach an LLM. A hosted LLM call has to leave the machine
over the network, so they run inside Anthropic's sandbox runtime (srt), which blocks
every destination not listed below at the operating-system level. That includes
traffic from compiled libraries (yfinance's curl_cffi) and subprocesses, which a
Python-level hook can't see.

Launch through this module, never directly:

    uv run python -m financial_news.sandbox server
    uv run python -m financial_news.sandbox monitor

Each entry point calls require_sandbox() at startup and refuses to run unsandboxed.
Download the approved finBERT model once, outside the sandbox, before first use:

    uv run python -m financial_news.sandbox prefetch-model

Adding a host here widens what the deterministic layer can reach; it needs review.
"""

import errno
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from financial_news.approved_models import APPROVED_SENTIMENT_MODELS
from financial_news.config import load_config

# Pinned: srt is pre-1.0, so a new version is a reviewed change, not an upgrade.
SRT_PACKAGE = "@anthropic-ai/sandbox-runtime@0.0.81"

FINNHUB_HOSTS = ["api.finnhub.io"]
# fc.yahoo.com issues the session cookie yfinance needs before querying.
YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com", "fc.yahoo.com"]
ALLOWED_HOSTS: dict[str, list[str]] = {
    "server": FINNHUB_HOSTS + YAHOO_HOSTS,
    "monitor": FINNHUB_HOSTS + YAHOO_HOSTS,  # plus the Grafana OTLP host, from env
}

# Credentials that would only be useful for reaching an LLM; never passed in.
LLM_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "HF_TOKEN",
)

DENY_READ = ["~/.ssh", "~/.aws", "~/.config/gh", "~/.git-credentials", "~/.netrc"]

# TEST-NET-1 (RFC 5737) is never routable. Inside the sandbox a connection attempt
# fails at once; outside it, the attempt hangs until the timeout.
_PROBE_ADDRESS = ("192.0.2.1", 9)
_SANDBOXED_ERRNOS = {errno.EPERM, errno.EACCES, errno.ENETUNREACH}


def allowed_hosts(target: str, environ: dict[str, str]) -> list[str]:
    hosts = list(ALLOWED_HOSTS[target])
    if target == "monitor":
        grafana_host = urlparse(environ.get("GRAFANA_CLOUD_OTLP_ENDPOINT", "")).hostname
        if grafana_host:
            hosts.append(grafana_host)
    return hosts


def build_settings(target: str, environ: dict[str, str]) -> dict:
    """Return the srt settings for target: strict host allowlist, minimal writes."""
    cfg = load_config()
    allow_write = [
        str(Path(cfg.logging.log_dir).resolve()),
        str(Path.home() / ".cache" / "huggingface"),
        tempfile.gettempdir(),
    ]
    if target == "monitor":
        allow_write.append(str(Path(cfg.monitor.snapshot_path).resolve().parent))
    return {
        "network": {
            "allowedDomains": allowed_hosts(target, environ),
            "deniedDomains": [],
            "strictAllowlist": True,
            "allowLocalBinding": False,
        },
        "filesystem": {
            "denyRead": DENY_READ,
            "allowRead": [],
            "allowWrite": allow_write,
            "denyWrite": [],
        },
    }


def write_settings(target: str, environ: dict[str, str]) -> Path:
    """Write target's srt settings to a fixed temp path, replaced on each launch."""
    path = Path(tempfile.gettempdir()) / f"financial-news-srt-{target}.json"
    path.write_text(json.dumps(build_settings(target, environ)))
    return path


def srt_command(npx: str, settings: Path, *command: str) -> list[str]:
    """Return the argv that runs command inside the pinned sandbox runtime."""
    return [npx, "-y", SRT_PACKAGE, "--settings", str(settings), *command]


def child_environ(environ: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in LLM_CREDENTIAL_VARS}
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def require_sandbox() -> None:
    """Exit unless this process is running inside the network sandbox.

    An offline machine also passes, which is harmless: it can't reach an LLM either.
    """
    for var in LLM_CREDENTIAL_VARS:
        os.environ.pop(var, None)
    try:
        socket.create_connection(_PROBE_ADDRESS, timeout=1).close()
    except OSError as exc:
        if exc.errno in _SANDBOXED_ERRNOS:
            return
    sys.exit(
        "Refusing to start outside the network sandbox. Launch with: "
        "uv run python -m financial_news.sandbox {server,monitor}"
    )


def prefetch_model() -> None:
    """Download every approved sentiment model at its pinned revision."""
    from huggingface_hub import snapshot_download  # type: ignore[import]

    for model_name, revision in APPROVED_SENTIMENT_MODELS.items():
        path = snapshot_download(model_name, revision=revision)
        print(f"{model_name}@{revision} -> {path}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["prefetch-model"]:
        prefetch_model()
        return 0
    if len(args) != 1 or args[0] not in ALLOWED_HOSTS:
        print(
            "usage: python -m financial_news.sandbox {server,monitor,prefetch-model}",
            file=sys.stderr,
        )
        return 2
    target = args[0]
    npx = shutil.which("npx")
    if npx is None:
        print("npx not found; install Node.js to run the sandbox", file=sys.stderr)
        return 1
    settings = write_settings(target, dict(os.environ))
    command = srt_command(
        npx, settings, sys.executable, "-m", f"financial_news.{target}"
    )
    os.execve(npx, command, child_environ(dict(os.environ)))


if __name__ == "__main__":
    sys.exit(main())
