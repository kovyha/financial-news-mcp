# Agents & Tools Catalog

This file catalogs the agents and tools used with this repository,
with their defined scope, triggers, and permissions.

Audience: Developers & Automation

Behavioral rules, constraints, and workflow expectations belong in `docs/SKILL.md`.
This file is the catalog only.

## Terminology

- **Agent:** An autonomous role that makes independent decisions, proposes changes, and requires human approval before landing on main. Agents have decision-making authority within their defined scope.

- **Tool:** A pure function exposed via MCP that Claude invokes on demand. Tools have no decision-making autonomy; they return structured data for the caller to interpret.

This distinction is important because agents require additional governance constraints (self-approval prevention, scope boundaries) while tools inherit their governance from the context where they're used.

---

## Autonomous Agents

### Explore

**Purpose:** Fast, read-only codebase exploration and Q&A. Use to get a quick overview,
locate symbols or files, or answer questions about how the code works.

**Trigger:** Human instruction.

**Scope:** Read-only. May read any file in the repository. Must not write, edit, or
delete files.

**Constraints:**
- Never proposes or makes changes. If a change is needed, hand off to the Contributor
  agent.
- Read `docs/developer-infra.md` before exploring unfamiliar parts of the project layout.

---

### Contributor

**Purpose:** Makes focused changes to the repository — implementation, tests,
documentation, or configuration — at the direction of a human.

**Trigger:** Human instruction (one-off or recurring task).

**Scope:** Same file-access as a human contributor. May read and write any file in the
repository, including `financial_news/server.py`, `financial_news/config.py`, tests, docs,
and configuration files.

**Constraints:**
- All changes must pass the full validation workflow (`ruff check`, `ruff format`,
  `pytest` with coverage enforcement) before being proposed for merge.
- All changes require human review and explicit approval before landing on `main`.
- Must not self-approve or merge its own changes.
- Must not modify `ci.yaml` or `uv.lock` without explicit human instruction.
- Follow `docs/engineering-standards.md` for code quality expectations.
- See `docs/SKILL.md` for behavioral rules and `CONTRIBUTING.md` for the validation
  workflow.

---

### PR Reviewer (Planned)

**Status:** Role defined; implementation pending. Will be implemented once multi-file diff parsing infrastructure is in place.

**Purpose:** Reviews an open pull request against the repository's engineering standards
and produces structured review output for a human to assess and post.

**Trigger:** Human instruction, given an open PR to review.

**Scope:** Read-only. Reads the PR diff, affected files, tests, and
`docs/engineering-standards.md`.

**Constraints:**
- Must not post comments, approvals, or any output directly to GitHub.
- Produces review output for a human to read, edit, and post.
- Checks: adherence to `docs/engineering-standards.md`, test coverage for changed
  behaviour, and whether the deterministic/LLM boundary in `server.py` has been touched
  without clear justification.

---

### Monitor

**Purpose:** Runs daily after US market close. Fetches news volume stats and price-move
stats for a configured ticker watchlist, then pushes z-score, recent count, and EWM mean
(for both signals) as OpenTelemetry gauges to Grafana Cloud via OTLP. On failure, triggers
the Diagnostic agent automatically.

The monitor is the first step of `.github/workflows/monitor.yaml`; the Briefing agent
runs as the second step of the same job (sharing the runner and `/tmp`).

**Triggers:**
1. **Scheduled (automatic):** Daily at 21:00 UTC (Mon–Fri), via `.github/workflows/monitor.yaml`.
   Disabled by default until secrets (`FINNHUB_API_KEY`, `GRAFANA_CLOUD_OTLP_ENDPOINT`,
   `GRAFANA_CLOUD_BASIC_AUTH_HEADER`) are configured in GitHub Actions.
2. **Manual:** `workflow_dispatch` on the `monitor.yaml` workflow (set `run_briefing: false`
   to skip the briefing step), or `uv run python -m financial_news.monitor` locally.

**Scope:** Read-only against the Finnhub API and the local config. Writes metrics to
Grafana Cloud only. Does not modify any file in the repository.

**Constraints:**
- Deterministic only within `run()`. Calls `compute_volume_stats()` from `analysis.py`
  and `compute_price_stats()` from `price.py`, recording both results via OTel gauges —
  no LLM inference occurs.
- Inf/large z-scores (zero-baseline spikes, extreme price moves) are capped so OTel
  gauges stay finite.
- News and price fetches fail independently per ticker (a yfinance outage does not
  block that ticker's news stats, and vice versa), each tracked by its own OTel error
  counter (`financial_news.monitor_errors` / `financial_news.monitor_price_errors`).
- A configurable delay (`[price] fetch_delay_seconds` in `config.toml`, default 1.0s)
  is applied between per-ticker yfinance calls — yfinance is an unofficial scraper of
  Yahoo's internal endpoints that is more sensitive to rapid back-to-back requests
  than Finnhub's documented rate limit.
- Ticker watchlist is driven by `[monitor]` in `config.toml` (default: 10-ticker list
  in `MonitorConfig`). Must not be hardcoded inside `monitor.py`.
- All proposed changes require explicit human approval before merge.

---

### Diagnostic

**Purpose:** Investigates anomalous behaviour — unexpected LLM output, bad Finnhub data,
or errors surfaced in the error log — and proposes fixes where the cause is identified.

**Triggers:**
1. **Scheduled (automatic):** Runs automatically when the monitor job fails, via
   `.github/workflows/monitor.yaml`. Reads the error log as configured in `config.toml`
   (default: `logs/financial_news.error.log`). If any error lines are present for the
   current day, begins investigation automatically.
2. **Ad hoc (human):** Triggered manually for urgent or complex incidents.

**Scope:** Full read access across the repository, including source code, configuration,
and log files in the configured log directory. May propose changes to any file, including
`financial_news/server.py`.

**Constraints:**
- All proposed changes require explicit human approval before merge. Must not commit or
  merge its own fixes.
- Must read the error log and produce a written diagnosis before proposing any fix.
  Must not propose fixes without a prior diagnosis.
- If the root cause cannot be determined from the error log and codebase alone, must
  surface that clearly rather than speculating.
- Follow `docs/engineering-standards.md` for any code changes proposed.

---

### Dependency Updater

**Purpose:** Reviews and proposes updates to runtime and development dependencies in
`pyproject.toml`, keeping the project current with upstream releases.

**Trigger:** Human instruction (periodic review).

**Scope:** Read access across the repository. Write access limited to `pyproject.toml`.

**Constraints:**
- Must not update `uv.lock` directly. Lockfile changes are a consequence of dependency
  changes and must be reviewed by a human running `uv lock` locally.
- Must not add new dependencies. Scope is updates to existing ones only; additions
  require a human decision and a rationale against `docs/engineering-standards.md`.
- Each proposed update must state the current version, the proposed version, and a
  brief reason (e.g. security fix, API change, routine update).
- All changes require human review before merge.

---

### Documentation Sync

**Purpose:** Ensures that documentation stays accurate and consistent with the code after
changes — specifically that architectural claims in docs (such as the deterministic/LLM
boundary) match the current implementation.

**Trigger:** Human instruction, typically after a non-trivial code change.

**Scope:** Read access across the full repository. Write access to files in `docs/` and
`README.md` only.

**Constraints:**
- Must not modify source code (`financial_news/`) or tests.
- Must not modify `AGENTS.md` or `docs/SKILL.md` — those are governance documents
  maintained by humans.
- When flagging a documentation gap, must cite the specific code location and the
  specific doc location that diverge before proposing any fix.
- All changes require human review before merge.

---

## Non-Autonomous Tools

### `get_news_volume` (MCP Tool)

**Type:** Deterministic signal tool (not an agent). Invoked by Claude, no autonomous decisions.

**Purpose:** Detects statistically unusual news volume for a given stock symbol and returns
a structured signal for Claude to reason over.

**Trigger:** Invoked by Claude via MCP in response to a user query.

**Scope:** Read-only against the Finnhub API. No write access to any system.

**Constraints:**
- Deterministic only. Fetches Finnhub data, computes article counts, mean, standard
  deviation (via pandas EWM) and z-score, classifies the result against configurable thresholds,
  and returns a structured string.
- No model inference occurs inside this tool. All LLM reasoning begins after the tool
  returns its output.
- Requires `FINNHUB_API_KEY` in the environment.
- Classification thresholds (normal / elevated / unusual) must not be changed without
  human review.

### `get_price_move` (MCP Tool)

**Type:** Deterministic signal tool (not an agent). Invoked by Claude, no autonomous decisions.

**Purpose:** Detects a statistically unusual price move for a given stock symbol and
returns a structured signal for Claude to reason over, expressed in the same z-score
units as `get_news_volume` so the two can be directly compared.

**Trigger:** Invoked by Claude via MCP in response to a user query.

**Scope:** Read-only against the yfinance API. No write access to any system.

**Constraints:**
- Deterministic only. Fetches yfinance daily price history and a live quote, computes
  a return z-score against an EWM baseline of prior returns (via `pandas`), and
  classifies the result against configurable thresholds.
- No model inference occurs inside this tool. All LLM reasoning begins after the tool
  returns its output.
- Requires no API key (yfinance is unauthenticated).
- Classification thresholds (normal / elevated / unusual) must not be changed without
  human review.

### `get_price_intraday` (MCP Tool)

**Type:** Deterministic signal tool (not an agent). Invoked by Claude, no autonomous decisions.

**Purpose:** Fetches today's intraday price bars for a given stock symbol — a
same-day attribution drill-down, not a second anomaly-detection signal. Lets Claude
line up a specific timestamped headline (from `get_news_volume`'s `articles`, each
of which carries a "time" field) against a specific intraday price move, when a
ticker's daily price move or news volume is already classified elevated/unusual.

**Trigger:** Invoked by Claude via MCP, typically after `get_price_move` or
`get_news_volume` flags a ticker as elevated/unusual and same-day attribution is
needed (e.g. more than one plausible news-driven move that day).

**Scope:** Read-only against the yfinance API. No write access to any system.

**Constraints:**
- Deterministic only. Fetches today's intraday bars at a configurable interval
  (`[price] intraday_interval` in `config.toml`, default 5-minute) and returns each
  bar's exchange-local time, price, and cumulative return since the previous close.
- No model inference occurs inside this tool. All LLM reasoning begins after the tool
  returns its output.
- Requires no API key (yfinance is unauthenticated).
- Not called for the whole watchlist — `monitor.py` never calls it, and
  `briefing.py` only builds its cache for tickers already classified
  elevated/unusual, to keep yfinance call volume bounded.

### `health_check` (MCP Tool)

**Type:** Deterministic health check tool (not an agent).

**Purpose:** Verifies that upstream dependencies (Finnhub API) are healthy and accessible.

**Trigger:** Invoked by Claude via MCP or manually for diagnostics.

**Scope:** Read-only health checks against Finnhub API.

**Constraints:**
- Returns a simple pass/fail status with timestamp
- No inference or caching of results
- Does not modify any state

---

## How to add a new agent or tool

1. Determine whether this is an **agent** (autonomous decisions) or a **tool** (pure function).
   See the Terminology section at the top for the distinction.

2. Add an entry to the appropriate section (Autonomous Agents or Non-Autonomous Tools):
   - **Agents:** Include purpose, trigger, scope, constraints, and any approval gates.
   - **Tools:** Include purpose, trigger, scope, and implementation constraints.

3. Update `docs/SKILL.md` if the agent needs additional behavioral rules or cross-cutting
   constraints not already covered.

4. Add tests or CI steps if the agent/tool changes repository behavior.

5. If this is an agent with autonomous decision-making, ensure it appears in the
   constraints under "SKILL.md" regarding human approval gates.
