"""
Tests to enforce the deterministic/LLM boundary.

These tests verify that the deterministic signal layer contains no model inference,
LLM API calls, or reasoning logic. All inference is reserved for the LLM reasoning
layer (Claude via MCP).
"""

import inspect
from pathlib import Path

from financial_news import analysis, server


class TestBoundaryEnforcement:
    """Verify that the deterministic layer has no LLM inference."""

    def test_server_module_contains_no_llm_imports(self):
        """Verify the server module has no LLM/model API imports."""
        source = inspect.getsource(server)
        forbidden = [
            "from anthropic",
            "import anthropic",
            "from openai",
            "import openai",
        ]
        for term in forbidden:
            assert term not in source.lower(), (
                f"Found forbidden import '{term}' in server module"
            )

    def test_server_module_contains_no_model_calls(self):
        """Verify the server module makes no calls to LLM APIs."""
        source = inspect.getsource(server)
        forbidden_patterns = [".predict(", ".generate(", ".complete(", ".chat("]
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"Found forbidden pattern '{pattern}' in server module"
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
        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        forbidden = ["anthropic", "openai", "cohere", "mistralai"]
        for package in forbidden:
            assert package not in pyproject, (
                f"LLM package '{package}' found in pyproject.toml dependencies"
            )
