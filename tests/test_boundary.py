"""
Tests to enforce the deterministic/LLM boundary.

These tests verify that the deterministic signal layer contains no model inference,
LLM API calls, or reasoning logic. All inference is reserved for the LLM reasoning
layer (Claude via MCP).
"""

import inspect
import tomllib
from pathlib import Path

import pytest

from financial_news import analysis, monitor, server


class TestBoundaryEnforcement:
    """Verify that the deterministic layer has no LLM inference."""

    @pytest.mark.parametrize(
        "mod", [analysis, monitor, server], ids=["analysis", "monitor", "server"]
    )
    def test_deterministic_modules_contain_no_llm_imports(self, mod):
        """Verify no deterministic module imports an LLM client library."""
        source = inspect.getsource(mod)
        forbidden = [
            "from anthropic",
            "import anthropic",
            "from openai",
            "import openai",
        ]
        for term in forbidden:
            assert term not in source.lower(), (
                f"Found forbidden import '{term}' in {mod.__name__}"
            )

    @pytest.mark.parametrize(
        "mod", [analysis, monitor, server], ids=["analysis", "monitor", "server"]
    )
    def test_deterministic_modules_contain_no_model_calls(self, mod):
        """Verify no deterministic module calls LLM inference APIs."""
        source = inspect.getsource(mod)
        for pattern in [".predict(", ".generate(", ".complete(", ".chat("]:
            assert pattern not in source, (
                f"Found forbidden pattern '{pattern}' in {mod.__name__}"
            )

    def test_threshold_constants_exist(self):
        """Verify classification thresholds are exported as named constants."""
        assert hasattr(analysis, "THRESHOLD_ELEVATED"), "THRESHOLD_ELEVATED not defined"
        assert hasattr(analysis, "THRESHOLD_UNUSUAL"), "THRESHOLD_UNUSUAL not defined"
        assert isinstance(analysis.THRESHOLD_ELEVATED, (int, float))
        assert isinstance(analysis.THRESHOLD_UNUSUAL, (int, float))
        assert analysis.THRESHOLD_ELEVATED < analysis.THRESHOLD_UNUSUAL, (
            "Thresholds out of order"
        )

    def test_thresholds_not_inlined_in_classification(self):
        """Verify classification logic uses named constants, not hardcoded literals."""
        source = inspect.getsource(analysis.compute_volume_stats)
        assert "THRESHOLD_ELEVATED" in source, (
            "compute_volume_stats must reference THRESHOLD_ELEVATED"
        )
        assert "THRESHOLD_UNUSUAL" in source, (
            "compute_volume_stats must reference THRESHOLD_UNUSUAL"
        )
        assert "z_score < 2" not in source, (
            "Hardcoded literal 2 found in classification; use THRESHOLD_ELEVATED"
        )
        assert "z_score < 3" not in source, (
            "Hardcoded literal 3 found in classification; use THRESHOLD_UNUSUAL"
        )

    def test_no_llm_packages_in_dependencies(self):
        """Verify no LLM client libraries are listed as project dependencies."""
        data = tomllib.loads(
            (Path(__file__).parent.parent / "pyproject.toml").read_text()
        )
        prod_deps = data.get("project", {}).get("dependencies", [])
        dev_deps = data.get("dependency-groups", {}).get("dev", [])
        all_deps = [str(d) for d in prod_deps + dev_deps]
        forbidden = ["anthropic", "openai", "cohere", "mistralai"]
        for dep in all_deps:
            for pkg in forbidden:
                assert pkg not in dep.lower(), (
                    f"LLM package '{pkg}' found in dependencies: {dep}"
                )
