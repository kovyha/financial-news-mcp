# SKILL: Agent Guidance

This file defines how agents should behave when working in this repository.

Purpose: behavioral rules, constraints, and workflow expectations for automated agents.
The agent catalog (roles, triggers, scope) lives in `docs/AGENTS.md`.

Audience: Agents / automation + Developers

---

## Scope boundaries

- Primary application code lives in `financial_news/server.py` and
  `financial_news/config.py`.
- Agents may update docs, tests, and focused implementation details within the scope
  defined for their role in `docs/AGENTS.md`.
- Agents must not operate on other repositories. Work is scoped to this repository only
  unless a human explicitly instructs otherwise for a specific task.

## Files requiring explicit human review

No agent may modify the following without explicit human instruction:

- `.github/workflows/ci.yaml` — CI configuration
- `uv.lock` — locked dependency versions
- `docs/AGENTS.md` — agent governance catalog
- `docs/SKILL.md` — this file
- Secrets, API keys, or credentials of any kind

## Human review gate (enforced by CI and GitHub)

All agent-authored changes — regardless of role — require explicit human approval before
landing on `main`. No agent may self-approve or merge its own changes. This applies to
code, tests, documentation, and configuration.

**Enforcement:**
- GitHub branch protection on `main` requires 1+ approval from `@{GITHUB_ORG}/maintainers`
- CI check (`protected-files`) blocks changes to governance files without explicit flags
- CODEOWNERS automatically assigns reviewers to PRs modifying critical files

## Deterministic/LLM boundary

The boundary between the deterministic signal layer (`get_news_volume` and its helpers)
and the LLM reasoning layer (Claude, via MCP) is a deliberate architectural property of
this system. Agents must not:

- Move business logic (z-score computation, classification thresholds, Finnhub fetch
  behaviour) into prompts, LLM calls, or agent reasoning steps.
- Move LLM interpretation or contextual reasoning into the deterministic layer.
- Modify classification thresholds (`z < 2`, `z < 3`) without explicit human review and
  a documented rationale.

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

When reviewing changes to `financial_news/server.py`, verify:

- [ ] Does `get_news_volume()` or its helpers call any LLM API? (Must be NO)
- [ ] Are classification thresholds (z < 2, z < 3) only modified via explicit constant?
      (Not inlined, not in prompts)
- [ ] Are all data transformations in the deterministic layer testable and reproducible?
- [ ] If thresholds changed: Is there a clear justification in the commit message or PR?
- [ ] Does the docstring in `get_news_volume()` still accurately describe the boundary?

## Adding new agents

When a new agent role is introduced:

1. Add an entry to `docs/AGENTS.md` with purpose, trigger, scope, and constraints.
2. Distinguish between autonomous agents (require human approval) and tools (pure functions).
3. Update this file if the new role requires cross-cutting behavioral rules not already
   covered.
4. Update CI or add tests if the agent affects repository behaviour.
