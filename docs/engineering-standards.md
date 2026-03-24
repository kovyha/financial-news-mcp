# Engineering Standards

Audience: Developers & Automation

This file defines the coding standards for this repository. It is intended to improve the quality of both human-written and agent-written changes by making expectations explicit.

## Purpose

Passing tests is necessary but not sufficient. Code in this repository should also be readable, maintainable, appropriately efficient, and easy to review.

Use this document as the default standard for new code, refactors, and test additions.

## ALWAYS

- Keep changes focused on the task being solved.
- Add or update tests when behavior changes.
- Prefer deterministic tests over tests that depend on live APIs or timing.
- Preserve public behavior unless the change explicitly intends to alter it.
- Update documentation when the user-facing behavior or developer workflow changes.
- Follow `CONTRIBUTING.md` for the canonical validation workflow.

## PREFER

- Prefer small, pure helper functions for isolated logic such as calculations and classification rules.
- Prefer explicit, descriptive names over clever or abbreviated names.
- Prefer parameterized tests when the same logic is being exercised across multiple scenarios.
- Prefer one clear pass over data instead of repeated scans when the logic is performance-sensitive.
- Prefer local repository patterns over generic framework patterns when both are viable.
- Prefer small refactors that improve structure without changing behavior.

### Better instruction examples

- Prefer extracting pure functions when branching logic becomes hard to test in-place.
- Prefer adding a targeted unit test before adding a larger integration-style test.
- Prefer keeping MCP tool functions thin and moving calculations into helpers.
- Prefer documenting a non-obvious abstraction with a short comment or docstring.

## AVOID

- Avoid duplicating business logic across multiple files.
- Avoid hidden network calls in unit tests.
- Avoid broad rewrites when a narrow fix or refactor is enough.
- Avoid adding new dependencies unless they provide clear project-level value.
- Avoid changing unrelated formatting in files you touch.
- Avoid silent changes to error handling, classification thresholds, or numeric assumptions.

## REQUIRE

- Require tests for behavior changes.
- Require coverage checks through the canonical workflow in `CONTRIBUTING.md`, with the repository threshold maintained unless intentionally changed.
- Require explicit review before changing lockfiles, CI behavior, or secrets-related configuration.
- Require clear reasoning for new abstractions, especially if they add indirection.
- Require documentation updates for changes that affect setup, packaging, or usage.
- Require that unit-testable logic be extracted when it is otherwise buried in I/O-heavy code.

## Testing standards

- Unit tests should focus on pure logic and edge cases.
- Integration-style tests should verify behavior at the boundary of a tool or user-facing function.
- Use monkeypatching or mocking to avoid live external calls in tests.
- Test names should describe behavior, not implementation details.

For this repo specifically:
- Put z-score and classification math in direct unit tests.
- Keep news-fetching tests isolated from the real Finnhub API.
- Treat `financial_news/server.py` as an orchestration layer and extract logic when it becomes too dense.

## Performance standards

- Optimize for clarity first when data sizes are small.
- If a path becomes performance-sensitive, prefer simple algorithmic improvements before adding complexity.
- Do not introduce performance-oriented abstractions without an identified bottleneck.

For this repo specifically:
- NumPy is appropriate for small numeric workflows here because it improves clarity and reliability.
- Do not optimize prematurely beyond the current scale of news-volume analysis.

## Dependency policy

- New dependencies should have a clear reason.
- Prefer mature, widely used libraries.
- Prefer standard library solutions when they are sufficient and keep the code simple.
- If a dependency is added, update `pyproject.toml` and ensure the development workflow still works with `uv`.

## Review checklist

Before considering a change complete, ask:

- Is this the simplest correct design?
- Is any logic duplicated unnecessarily?
- Can the most important behavior be tested in isolation?
- Is the code more understandable after the change?
- Did the change preserve the project’s packaging, test, and development workflow?
