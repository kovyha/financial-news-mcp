# Contributing

Thanks for contributing.

## Development workflow

Install the project and all dependency groups (including `diagnostic`, which provides `anthropic` for the LLM-powered agent tests):

```bash
uv sync --all-groups
```

The GitHub Actions CI workflow at `.github/workflows/ci.yaml` runs on pull requests and on
pushes to `main`. It has three jobs:

- `lint-and-test` — `uv run ruff format --check .`, `uv run ruff check .`, and
  `uv run pytest --cov=financial_news --cov-report=term-missing -q`.
- `sandbox` — runs the real network sandbox on Linux and checks every escape route in
  `tests/test_sandbox_integration.py` is blocked. Locally: `RUN_SANDBOX_TESTS=1 uv run
  pytest tests/test_sandbox_integration.py --no-cov` (needs Node.js).
- `protected-files` — PRs only; see below.

Changes reach `main` only through a PR. Branch protection requires all three jobs to pass,
with no bypass.

Run the validation flow before submitting changes:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest --cov=financial_news --cov-report=term-missing -q
```

The coverage threshold is configured in `pyproject.toml` under `[tool.coverage.report]`
and enforced automatically by `pytest-cov`. To change the threshold, update that value
and commit the change.

## Expectations

- Keep changes focused.
- Add or update tests for behavior changes.
- Follow the repo's existing style and structure.
- Keep coverage at or above the threshold in `pyproject.toml` unless there is an explicit
  reason to change it.

## Review checklist for maintainers

Before approving a PR, verify:

### Boundary changes
- [ ] Does this PR touch the deterministic/LLM boundary in `financial_news/server.py`?
      - If yes: verify the change has clear justification and documentation
      - See `docs/SKILL.md` for the full boundary checklist

### Protected files
- [ ] Does this PR modify a file in `.github/protected-files.txt` (governance docs,
      workflows, `uv.lock`, the sandbox and its allowlist, approved models, or the
      boundary tests)?
      - If yes: the `protected-files` check fails until you approve it. Verify the change
        is intentional and necessary, then add the `protected-change-approved` label.
      - If the label is already on and new commits arrive, re-review them: the label
        stays on until removed.

### Agent roles
- [ ] Does this PR introduce or modify any agent roles?
      - If yes: ensure entries in `docs/AGENTS.md` are updated
      - Verify the role has appropriate constraints in `docs/SKILL.md`

### Governance compliance
- [ ] Does this PR add new agent capabilities?
      - If yes: ensure human approval gate is documented in AGENTS.md and SKILL.md

### Standards
- [ ] Does this PR pass all automated checks (lint, format, coverage)?
      - Run: `uv run ruff check . && uv run ruff format . && uv run pytest --cov=financial_news`
- [ ] Does this PR follow `docs/engineering-standards.md`?

## Related docs

- Developer workflow: [docs/developer-infra.md](docs/developer-infra.md)
- Engineering standards: [docs/engineering-standards.md](docs/engineering-standards.md)
- Change history: [CHANGELOG.md](CHANGELOG.md)