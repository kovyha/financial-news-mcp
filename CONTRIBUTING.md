# Contributing

Thanks for contributing.

## Development workflow

Install the project in editable mode:

```bash
uv pip install -e .
```

The repository includes a minimal GitHub Actions CI workflow at `.github/workflows/ci.yaml`.
It runs `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest --cov=financial_news --cov-report=term-missing -q` on pushes and pull requests.

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
- [ ] Does this PR modify `.github/workflows/ci.yaml`, `uv.lock`, `docs/AGENTS.md`, or
      `docs/SKILL.md`?
      - If yes: verify it's intentional and necessary

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