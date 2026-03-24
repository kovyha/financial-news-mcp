# Z-score Iteration — Summary

Audience: Developers

## Overview
This iteration implements z-score based normalization and related changes to the data pipeline. It introduces NumPy as a dependency and updates the project to compute sample and population standard deviations where appropriate.

## What changed
- Added NumPy as a project dependency and installed it into the project environment using `uv`.
- Updated code to compute means and standard deviations using `numpy` (uses `ddof=0` for population, `ddof=1` for sample where specified).
- Implemented the z-score computation and integrated it into the processing flow (committed in the latest commit).

## Files touched (examples)
- `financial_news/server.py` — introduced `calculate_z_score()` and integrated z-score classification into the MCP flow.
- `pyproject.toml` — dependency entry for `numpy` (and possibly dev-group entries).
- `uv.lock` — lockfile updated by `uv` to record installed versions.
- `tests/test_calculate_z_score.py` — unit coverage for z-score guard logic.
- `tests/test_get_news_volume.py` — classification-oriented tests for tool output.

## Verification

This iteration was validated using the standard contributor workflow in `CONTRIBUTING.md`.

## Rationale
- `numpy` was chosen because the data size and logic are small but numeric operations benefit from NumPy's API and clarity.
- `ddof` handling was made explicit to distinguish sample vs population std where statistical correctness matters.

## Next steps / TODOs
- Add usage docs showing where in the pipeline z-score normalization is applied.
- Review whether `tests/conftest.py` can be simplified now that editable installs are in place.
