"""
Tests to enforce the deterministic/LLM boundary.

These tests verify that the deterministic signal layer contains no LLM inference,
LLM API calls, or reasoning logic. All LLM inference is reserved for the LLM reasoning
layer (Claude via MCP). finBERT runs locally in `sentiment` as a preprocessing step;
it is not an LLM, so these tests permit it.

They are the static half of the boundary. The runtime half is the network sandbox
(financial_news/sandbox.py, tests/test_sandbox_integration.py), which blocks hosted
LLM calls however they are written. These tests catch what the sandbox can't see,
such as a model loaded in-process or an LLM request sent back through MCP, and they
fail before the code ever runs.
"""

import ast
import inspect
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

import financial_news
from financial_news import analysis, price, sandbox
from financial_news.approved_models import APPROVED_SENTIMENT_MODELS

PACKAGE = "financial_news"
PACKAGE_DIR = Path(financial_news.__file__).parent
REPO_ROOT = PACKAGE_DIR.parent

# Every runnable module must be listed in exactly one of these.
DETERMINISTIC_ENTRY_POINTS = {f"{PACKAGE}.server", f"{PACKAGE}.monitor"}
LLM_ENTRY_POINTS = {f"{PACKAGE}.briefing", f"{PACKAGE}.diagnostic"}
TOOLING_ENTRY_POINTS = {
    f"{PACKAGE}.evaluation",
    f"{PACKAGE}.eval_gate",
    f"{PACKAGE}.sandbox",
}

HOSTED_LLM_PACKAGES = {"anthropic", "cohere", "google", "mistralai", "openai"}
# Libraries whose job is running a language model in-process or on localhost.
LOCAL_LLM_PACKAGES = {
    "ctransformers",
    "exllamav2",
    "gpt4all",
    "llama-cpp-python",
    "llama_cpp",
    "lmdeploy",
    "mlx-lm",
    "mlx_lm",
    "ollama",
    "sglang",
    "vllm",
}

FORBIDDEN_PATTERNS = [
    # Generic LLM call shapes.
    ".predict(",
    ".generate(",
    ".complete(",
    ".chat(",
    # Asking the MCP client to run an LLM (sampling) needs no network call.
    "create_message",
    "sampling",
    # Generative pipelines and model classes.
    "text-generation",
    "text2text-generation",
    "ForCausalLM",
    "ForSeq2SeqLM",
    # Dynamic imports would hide a package from the import walk.
    "import_module(",
    "__import__(",
    # Raw HTTP to a provider.
    "api.anthropic.com",
    "api.openai.com",
]

MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}


def _module_path(module: str) -> Path:
    return PACKAGE_DIR / f"{module.removeprefix(f'{PACKAGE}.')}.py"


def _imported_names(module: str) -> set[str]:
    """Return every dotted name a module imports, including inside functions."""
    tree = ast.parse(_module_path(module).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # The package is flat, so a relative import always resolves inside it.
            base = PACKAGE if node.level else ""
            if node.module:
                base = f"{base}.{node.module}" if base else node.module
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _reachable(entry_points: set[str]) -> tuple[set[str], set[str]]:
    """Walk imports from entry_points; return (internal modules, external roots)."""
    modules: set[str] = set()
    external: set[str] = set()
    pending = list(entry_points)
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        modules.add(module)
        for name in _imported_names(module):
            if name.split(".")[0] != PACKAGE:
                external.add(name.split(".")[0])
            elif _module_path(name).exists():
                pending.append(name)
    return modules, external


def _main_block(module: str) -> ast.If | None:
    tree = ast.parse(_module_path(module).read_text())
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return node
    return None


DETERMINISTIC_MODULES, DETERMINISTIC_EXTERNAL = _reachable(DETERMINISTIC_ENTRY_POINTS)


class TestBoundaryEnforcement:
    """Verify that the deterministic layer has no LLM inference."""

    def test_no_llm_package_reachable_from_deterministic_entry_points(self):
        """Verify no import chain from server or monitor reaches an LLM library."""
        found = DETERMINISTIC_EXTERNAL & (HOSTED_LLM_PACKAGES | LOCAL_LLM_PACKAGES)
        assert not found, f"LLM packages reachable from deterministic code: {found}"

    def test_no_direct_curl_cffi_in_project_code(self):
        """curl_cffi bypasses Python's networking; only yfinance may use it."""
        assert "curl_cffi" not in DETERMINISTIC_EXTERNAL

    def test_import_walk_detects_llm_packages(self):
        """Verify the walk finds the LLM client the LLM layer really imports."""
        _, external = _reachable(LLM_ENTRY_POINTS)
        assert "anthropic" in external

    def test_import_walk_follows_the_tool_path(self):
        """Verify the walk follows server -> enrichment -> sentiment (lazy imports)."""
        assert {f"{PACKAGE}.enrichment", f"{PACKAGE}.sentiment"} <= (
            DETERMINISTIC_MODULES
        )
        assert "transformers" in DETERMINISTIC_EXTERNAL

    @pytest.mark.parametrize("module", sorted(DETERMINISTIC_MODULES))
    def test_deterministic_modules_contain_no_model_calls(self, module):
        """Verify no deterministic module calls LLM inference APIs."""
        source = _module_path(module).read_text()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in source, (
                f"Found forbidden pattern '{pattern}' in {module}"
            )

    def test_every_entry_point_is_classified(self):
        """A new runnable module must be declared deterministic, LLM or tooling."""
        runnable = {
            f"{PACKAGE}.{path.stem}"
            for path in PACKAGE_DIR.glob("*.py")
            if _main_block(f"{PACKAGE}.{path.stem}") is not None
        }
        declared = DETERMINISTIC_ENTRY_POINTS | LLM_ENTRY_POINTS | TOOLING_ENTRY_POINTS
        assert runnable == declared

    @pytest.mark.parametrize("module", sorted(DETERMINISTIC_ENTRY_POINTS))
    def test_deterministic_entry_points_require_sandbox(self, module):
        """Each deterministic entry point refuses to start outside the sandbox."""
        block = _main_block(module)
        first = ast.unparse(block.body[0])
        assert first == "require_sandbox()", f"{module} must call it first"
        assert module.removeprefix(f"{PACKAGE}.") in sandbox.ALLOWED_HOSTS

    def test_no_local_llm_runtime_in_lockfile(self):
        """A local LLM runtime can't arrive as a dependency, even transitively."""
        lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
        names = {pkg["name"] for pkg in lock.get("package", [])}
        assert not names & LOCAL_LLM_PACKAGES

    def test_no_model_weights_committed(self):
        """Weights can't arrive by being committed to the repo."""
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.split()
        weights = [f for f in tracked if Path(f).suffix in MODEL_WEIGHT_SUFFIXES]
        assert not weights

    def test_approved_models_are_pinned_to_a_commit(self):
        for model_name, revision in APPROVED_SENTIMENT_MODELS.items():
            assert re.fullmatch(r"[0-9a-f]{40}", revision), model_name

    def test_sandbox_runtime_version_is_pinned(self):
        assert re.fullmatch(
            r"@anthropic-ai/sandbox-runtime@\d+\.\d+\.\d+", sandbox.SRT_PACKAGE
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

    def test_price_threshold_constants_exist(self):
        """Verify price classification thresholds are exported as named constants."""
        assert hasattr(price, "PRICE_THRESHOLD_ELEVATED"), (
            "PRICE_THRESHOLD_ELEVATED not defined"
        )
        assert hasattr(price, "PRICE_THRESHOLD_UNUSUAL"), (
            "PRICE_THRESHOLD_UNUSUAL not defined"
        )
        assert isinstance(price.PRICE_THRESHOLD_ELEVATED, (int, float))
        assert isinstance(price.PRICE_THRESHOLD_UNUSUAL, (int, float))
        assert price.PRICE_THRESHOLD_ELEVATED < price.PRICE_THRESHOLD_UNUSUAL, (
            "Price thresholds out of order"
        )

    def test_price_thresholds_not_inlined_in_compute_price_stats(self):
        """Verify price classification logic uses named constants, not literals."""
        source = inspect.getsource(price.compute_price_stats)
        assert "PRICE_THRESHOLD_ELEVATED" in source, (
            "compute_price_stats must reference PRICE_THRESHOLD_ELEVATED"
        )
        assert "PRICE_THRESHOLD_UNUSUAL" in source, (
            "compute_price_stats must reference PRICE_THRESHOLD_UNUSUAL"
        )
        assert "z_score) < 2" not in source, (
            "Hardcoded literal 2 found in classification; use PRICE_THRESHOLD_ELEVATED"
        )
        assert "z_score) < 3" not in source, (
            "Hardcoded literal 3 found in classification; use PRICE_THRESHOLD_UNUSUAL"
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
