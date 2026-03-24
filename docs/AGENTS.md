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

### Diagnostic

**Purpose:** Investigates anomalous behaviour — unexpected LLM output, bad Finnhub data,
or errors surfaced in the error log — and proposes fixes where the cause is identified.

**Triggers:**
1. **Scheduled (automatic):** Runs daily. Reads the error log as configured in
   `config.toml` (default: `logs/financial_news.error.log`). If any error lines are
   present for the current day, begins investigation automatically.
2. **Ad hoc (human):** Triggered manually for urgent or complex incidents where a
   scheduled run would be too slow.

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
  deviation, and z-score using NumPy, classifies the result against fixed thresholds,
  and returns a structured string.
- No model inference occurs inside this tool. All LLM reasoning begins after the tool
  returns its output.
- Requires `FINNHUB_API_KEY` in the environment.
- Classification thresholds (normal / elevated / unusual) must not be changed without
  human review.

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
