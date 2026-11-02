# SKILL: Agent Guidance

This file defines how agents should behave when working in this repository.

Purpose: behavioral rules, constraints, and workflow expectations for automated agents.
The agent catalog (roles, triggers, scope) lives in `docs/AGENTS.md`.

Audience: Agents / automation + Developers

---

## Scope boundaries

- Primary application code lives in `financial_news/server.py`,
  `financial_news/config.py`, and `financial_news/monitor.py`.
- Agents may update docs, tests, and focused implementation details within the scope
  defined for their role in `docs/AGENTS.md`.
- Agents must not operate on other repositories. Work is scoped to this repository only
  unless a human explicitly instructs otherwise for a specific task.

## Files requiring explicit human review

No agent may modify the following without explicit human instruction. The canonical list
is `.github/protected-files.txt`:

- `docs/AGENTS.md` — agent governance catalog
- `docs/SKILL.md` — this file
- `.github/` — workflows, shared actions, `CODEOWNERS`, and the protected-files list itself
- `.claude/settings.json` — agent permission rules
- `uv.lock` — locked dependency versions
- `financial_news/sandbox.py` — network sandbox and host allowlist
- `financial_news/approved_models.py` — approved, pinned sentiment models
- `financial_news/eval_gate.py` — evaluation gate: what triggers a re-run and the pass mark
- `tests/test_boundary.py`, `tests/test_sandbox_integration.py` — boundary tests
- Secrets, API keys, or credentials of any kind

## Human review gate (enforced by CI and GitHub)

All agent-authored changes — regardless of role — require explicit human approval before
landing on `main`. No agent may self-approve or merge its own changes. This applies to
code, tests, documentation, and configuration.

**Enforcement:**
- GitHub branch protection on `main` requires a PR and passing CI, with no bypass. The
  maintainer reviews and merges each PR. (With a single maintainer GitHub can't require
  an approving review, because no one can approve their own PR.)
- CI check (`protected-files`) fails any PR touching a protected file until the maintainer
  adds the `protected-change-approved` label. It reads the list from `main`, so a PR
  can't shorten its own check.
- `.claude/settings.json` denies agent pushes to `main` and force pushes.
- CODEOWNERS requests the maintainer's review on PRs modifying critical files.

## Deterministic/LLM boundary

The boundary between the deterministic signal layer (`get_news_volume`, `monitor.py`, and
their helpers) and the LLM reasoning layer (Claude, via MCP) is a deliberate architectural
property of this system. Agents must not:

- Move business logic (z-score computation, classification thresholds, Finnhub fetch
  behaviour) into prompts, LLM calls, or agent reasoning steps.
- Move LLM interpretation or contextual reasoning into the deterministic layer.
- Modify classification thresholds (`threshold_elevated`, `threshold_unusual` in `config.toml` or their defaults in `AnalysisConfig`) without explicit human review and a documented rationale.
- Widen the sandbox allowlist, add or re-pin a sentiment model, or weaken a boundary test
  without explicit human instruction. These are protected files.
- Run the MCP server or monitor outside the network sandbox. Launch them with
  `python -m financial_news.sandbox {server,monitor}`.

The boundary is enforced at runtime by the network sandbox (`financial_news/sandbox.py`)
and statically by `tests/test_boundary.py`. See "How the boundary is enforced" in
`README.md`, including what it doesn't cover.

Any change that touches this boundary must be flagged explicitly in the PR or change
summary, and reviewed by a human before merge.

## Required references

All agents making changes should consult:

- `CONTRIBUTING.md` — canonical validation workflow, including lint, format, and coverage.
- `docs/engineering-standards.md` — code quality expectations for human and agent-authored
  code alike.
- `docs/developer-infra.md` — packaging, install, and repository workflow context.

## Validation workflow

Before proposing any change for merge, run:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest --cov=financial_news --cov-report=term-missing -q
```

All checks must pass. Do not propose changes that fail lint or drop coverage below the
threshold configured in `pyproject.toml` under `[tool.coverage.report]`.

**Enforcement:** Agents must include the output of the validation workflow commands in their change proposal to demonstrate compliance. If any check fails, the proposal must explain the failure and provide a plan to resolve it before proceeding.

## Commit guidance

- Use concise commit messages: `<scope>: <short description>`.
- Examples: `fix: guard z-score division when std is zero`, `test: add config edge cases`
- Do not amend published commits. Create a new commit if a fix is needed after push.

## Diagnostic agent: investigation protocol

When the Diagnostic agent runs — whether on schedule or ad hoc — it must follow this
sequence:

1. Read the error log for the relevant time window.
2. Produce a written summary: what errors occurred, when, and how many times.
3. Trace the error to a specific location in the codebase.
4. Only then propose a fix, referencing both the log evidence and the code location.
5. If the cause cannot be determined, state that explicitly. Do not speculate.

## Code review checklist: Deterministic/LLM boundary

When reviewing changes to `financial_news/server.py`, `financial_news/monitor.py`, or
`financial_news/price.py`, verify:

- [ ] Does `get_news_volume()`, `get_price_move()`, `get_price_intraday()`, `run()`, or their helpers call any LLM API? (Must be NO)
- [ ] Are classification thresholds only modified via `AnalysisConfig`/`PriceConfig` in `config.py` or `config.toml`?
      (Not inlined in code, not in prompts)
- [ ] Are all data transformations in the deterministic layer testable and reproducible?
- [ ] If thresholds changed: Is there a clear justification in the commit message or PR?
- [ ] Does the docstring in `get_news_volume()`/`get_price_move()`/`get_price_intraday()` still accurately describe the boundary?
- [ ] Does `get_price_intraday` (and `briefing.py`'s intraday cache) stay an on-demand, elevated/unusual-only drill-down — not fetched eagerly for the whole watchlist in `monitor.py`?
- [ ] Is `GAUGE_SPECS` in `analysis.py` / `PRICE_GAUGE_SPECS` in `price.py` the single source of truth for metric registration?
      (Gauge keys must not be hardcoded in `monitor.py`)

## Adding new agents

When a new agent role is introduced:

1. Add an entry to `docs/AGENTS.md` with purpose, trigger, scope, and constraints.
2. Distinguish between autonomous agents (require human approval) and tools (pure functions).
3. Update this file if the new role requires cross-cutting behavioral rules not already
covered.
4. Update CI or add tests if the agent affects repository behaviour.
