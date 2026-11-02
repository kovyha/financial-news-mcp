"""Evaluation gate: re-run the full-system judge whenever anything its verdicts
depend on changes, and fail if the verdicts got worse.

The fingerprint (eval/fingerprint.json) records what the judge's verdicts depend
on: its prompts, the code on the verdict path, the Claude model and token limit,
the locked versions of the libraries that produce verdicts, the pinned finBERT
revision, the article-selection settings, and the labels and candidates it's
scored on. CI compares a PR's fingerprint with the base branch's.
If they differ, CI re-runs the judge (after a maintainer approves the run) and
fails if more labeled rows went from right to wrong than the pass mark allows.

Commands:
    # CI, every PR, no API calls: is the committed fingerprint current, and does
    # it differ from the base branch's?
    uv run python -m financial_news.eval_gate fingerprint-check \\
        --base FILE --base-verdicts FILE

    # Locally, after changing anything fingerprinted: re-run the judge and write
    # the new verdicts and fingerprint for the PR to commit.
    ANTHROPIC_API_KEY=<key> uv run python -m financial_news.eval_gate update

    # CI, only when the fingerprint changed: re-run the judge and compare.
    ANTHROPIC_API_KEY=<key> uv run python -m financial_news.eval_gate check \\
        --base-verdicts FILE

All commands use the committed defaults (Config()), never a local config.toml,
so a run is reproducible from the repo alone. See docs/evaluation.md.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import sys
import textwrap
import tomllib
from pathlib import Path

from financial_news import evaluation
from financial_news.approved_models import APPROVED_SENTIMENT_MODELS
from financial_news.config import Config

FINGERPRINT_PATH = Path("eval/fingerprint.json")
VERDICTS_PATH = evaluation._FULL_SYSTEM_VERDICTS_PATH

# Where the code that turns a candidate into a verdict starts. Every project
# function these call, directly or indirectly, is hashed too (see
# _verdict_path_functions), so a new helper can't slip past the fingerprint.
_VERDICT_PATH_ROOTS = (
    evaluation._judge_system_prompt,
    evaluation.run_full_system,
)
_PACKAGE = "financial_news"


# Libraries whose upgrade can change verdicts: the Claude client, and finBERT's
# runtime (tokenisation and inference).
_VERDICT_PATH_PACKAGES = ("anthropic", "torch", "transformers")
_LOCKFILE_PATH = Path("uv.lock")


def _locked_versions() -> dict[str, str]:
    """Versions of the verdict-path libraries pinned in uv.lock. A package locked
    at several versions (one per platform) lists them all, sorted.
    """
    versions: dict[str, set[str]] = {}
    for package in tomllib.loads(_LOCKFILE_PATH.read_text())["package"]:
        if package["name"] in _VERDICT_PATH_PACKAGES:
            versions.setdefault(package["name"], set()).add(package["version"])
    return {
        name: ", ".join(sorted(versions.get(name, {"not locked"})))
        for name in _VERDICT_PATH_PACKAGES
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_source(fn) -> str:
    """Source without comments, docstrings or formatting, so only edits that can
    change behaviour change the fingerprint.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _resolve(node: ast.expr, namespace: dict):
    """The object a Name or module.attr expression refers to, if it's simple."""
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        owner = namespace.get(node.value.id)
        if inspect.ismodule(owner):
            return getattr(owner, node.attr, None)
    return None


def _in_package(obj) -> bool:
    return (getattr(obj, "__module__", None) or "").startswith(_PACKAGE)


def _verdict_path_functions() -> tuple[list, dict[str, str]]:
    """Every project function reachable by calls from the roots, and the
    module-level constants they read, as {"module.NAME": repr}.
    """
    seen: dict[str, object] = {}
    constants: dict[str, str] = {}
    pending = list(_VERDICT_PATH_ROOTS)
    while pending:
        fn = pending.pop()
        name = f"{fn.__module__}.{fn.__qualname__}"
        if name in seen:
            continue
        seen[name] = fn
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if not isinstance(node, (ast.Name, ast.Attribute)):
                continue
            target = _resolve(node, fn.__globals__)
            if inspect.isfunction(target) and _in_package(target):
                pending.append(target)
            elif isinstance(target, (str, int, float, tuple, frozenset)):
                key = node.id if isinstance(node, ast.Name) else node.attr
                if key.isupper() or key.lstrip("_").isupper():
                    constants[f"{fn.__module__}.{key}"] = repr(target)
    return [seen[k] for k in sorted(seen)], dict(sorted(constants.items()))


def code_fingerprint(cfg: Config) -> dict:
    """Fingerprint components computable from the repo alone (no private data)."""
    enrich = evaluation._enrich_config(cfg)
    prompts = "\n---\n".join(
        evaluation._judge_system_prompt(include_news_zscore=flag)
        for flag in (True, False)
    )
    functions, constants = _verdict_path_functions()
    source = "\n".join(_normalized_source(fn) for fn in functions)
    source += "\n" + json.dumps(constants, sort_keys=True)
    return {
        "claude_model": evaluation._MODEL,
        "claude_max_tokens": evaluation._MAX_TOKENS,
        "sentiment_model": (
            f"{enrich.model_name}@{APPROVED_SENTIMENT_MODELS[enrich.model_name]}"
        ),
        "article_selection": {
            "confidence_threshold": enrich.confidence_threshold,
            "min_articles": enrich.min_articles,
            "max_articles": enrich.max_articles,
            "labels": sorted(enrich.valid_labels),
        },
        "library_versions": _locked_versions(),
        "judge_system_prompt_sha256": _sha256(prompts.encode()),
        "verdict_code_sha256": _sha256(source.encode()),
        "labels_sha256": _sha256(evaluation._LABELS_PATH.read_bytes()),
    }


def full_fingerprint(cfg: Config) -> dict:
    """Code fingerprint plus the checksum of the private candidate pool."""
    return {
        **code_fingerprint(cfg),
        "candidates_sha256": _sha256(evaluation._CANDIDATES_PATH.read_bytes()),
    }


def allowed_net_worse(n: int) -> int:
    """Pass mark: how many more rows may go right→wrong than wrong→right."""
    return max(2, math.ceil(0.02 * n))


def compare_verdicts(
    truth: dict[str, bool], base: dict[str, bool], new: dict[str, bool]
) -> dict:
    """Paired comparison of two verdict sets on the labeled rows both cover."""
    keys = sorted(truth.keys() & base.keys() & new.keys())
    newly_wrong = [k for k in keys if base[k] == truth[k] and new[k] != truth[k]]
    newly_right = [k for k in keys if base[k] != truth[k] and new[k] == truth[k]]
    return {
        "n": len(keys),
        "newly_wrong": newly_wrong,
        "newly_right": newly_right,
        "net_worse": len(newly_wrong) - len(newly_right),
        "allowed_net_worse": allowed_net_worse(len(keys)),
        "not_compared": sorted(truth.keys() - keys),
    }


def run_judge(candidates: list[dict], labels: dict, cfg: Config) -> list[dict]:
    """Full-system verdict for every confirmed label, in verdicts-file format."""
    return evaluation.run_full_system(
        evaluation._labeled_rows(candidates, labels), evaluation._enrich_config(cfg)
    )


def _as_map(rows: list[dict], field: str) -> dict[str, bool]:
    return {
        evaluation._label_key(r["ticker"], r["date"]): r[field] == "news_driven"
        for r in rows
    }


def _print_comparison(title: str, result: dict) -> bool:
    passed = result["net_worse"] <= result["allowed_net_worse"]
    print(f"\n{title}: {'PASS' if passed else 'FAIL'}")
    print(
        f"  rows compared: {result['n']}  net worse: {result['net_worse']}"
        f" (allowed: {result['allowed_net_worse']})"
    )
    print(f"  right → wrong: {result['newly_wrong'] or 'none'}")
    print(f"  wrong → right: {result['newly_right'] or 'none'}")
    if result["not_compared"]:
        print(f"  not in both sets, so not compared: {result['not_compared']}")
    return passed


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def fingerprint_check(
    cfg: Config, base_path: Path | None, base_verdicts_path: Path | None = None
) -> tuple[int, bool]:
    """Return (exit code, whether the judge must be re-run in CI).

    A re-run is needed when the fingerprint differs from the base branch's, or
    when the stored verdicts do: they become the next baseline, so a PR can't
    change them without the judge reproducing them.
    """
    committed = _read_json(FINGERPRINT_PATH)
    current = code_fingerprint(cfg)
    stale = sorted(k for k, v in current.items() if committed.get(k) != v)
    if stale:
        print(
            f"❌ {FINGERPRINT_PATH} is out of date for: {', '.join(stale)}.\n"
            "These change the judge's verdicts, so re-run it and commit the result:\n"
            "    ANTHROPIC_API_KEY=<key> uv run python -m financial_news.eval_gate"
            " update"
        )
        return 1, False
    if base_path is None or not base_path.exists():
        print("✅ Fingerprint is current. The base branch has none yet; no re-run.")
        return 0, False
    base = _read_json(base_path)
    changed = sorted(
        k for k in committed.keys() | base.keys() if committed.get(k) != base.get(k)
    )
    if base_verdicts_path is not None and (
        not base_verdicts_path.exists()
        or _read_json(base_verdicts_path) != _read_json(VERDICTS_PATH)
    ):
        changed.append("stored verdicts")
    if not changed:
        print("✅ Fingerprint matches the base branch; no re-run needed.")
        return 0, False
    print(f"Differs from the base branch in: {', '.join(changed)}.")
    print("The eval-run job will re-run the judge once a maintainer approves it.")
    return 0, True


def update(cfg: Config) -> int:
    """Re-run the judge, report against the stored verdicts, write the new ones."""
    candidates = evaluation.load_candidates(evaluation._CANDIDATES_PATH)
    if not candidates:
        print(f"❌ {evaluation._CANDIDATES_PATH} is missing; see docs/evaluation.md.")
        return 1
    labels = evaluation.load_labels(evaluation._LABELS_PATH)
    new_rows = run_judge(candidates, labels, cfg)
    truth = _as_map(new_rows, "true_label")
    if VERDICTS_PATH.exists():
        old = _as_map(_read_json(VERDICTS_PATH), "predicted")
        result = compare_verdicts(truth, old, _as_map(new_rows, "predicted"))
        _print_comparison("Against the stored verdicts", result)
    _write_json(VERDICTS_PATH, new_rows)
    _write_json(FINGERPRINT_PATH, full_fingerprint(cfg))
    print(f"\nWrote {VERDICTS_PATH} and {FINGERPRINT_PATH}. Commit both.")
    return 0


def check(cfg: Config, base_verdicts_path: Path) -> int:
    """CI: re-run the judge and compare with the base branch's verdicts, and
    check the verdicts this PR commits are ones the judge actually produces.
    """
    committed = _read_json(FINGERPRINT_PATH)
    actual = _sha256(evaluation._CANDIDATES_PATH.read_bytes())
    if committed.get("candidates_sha256") != actual:
        print(
            "❌ The candidate pool doesn't match candidates_sha256 in"
            f" {FINGERPRINT_PATH}. Update the private data repo or the fingerprint."
        )
        return 1
    new_rows = run_judge(
        evaluation.load_candidates(evaluation._CANDIDATES_PATH),
        evaluation.load_labels(evaluation._LABELS_PATH),
        cfg,
    )
    truth = _as_map(new_rows, "true_label")
    new = _as_map(new_rows, "predicted")
    base_ok = _print_comparison(
        "Against the base branch's verdicts",
        compare_verdicts(
            truth, _as_map(_read_json(base_verdicts_path), "predicted"), new
        ),
    )
    # Same tolerance, applied to the verdicts this PR commits as the new baseline:
    # they must be ones the judge reproduces, not hand-edited.
    committed_rows = _as_map(_read_json(VERDICTS_PATH), "predicted")
    keys = sorted(truth.keys() & committed_rows.keys())
    disagree = [k for k in keys if committed_rows[k] != new[k]]
    missing = sorted(truth.keys() - committed_rows.keys())
    allowed = allowed_net_worse(len(truth))
    committed_ok = len(disagree) <= allowed and not missing
    print(
        f"\nCommitted verdicts reproduce: {'PASS' if committed_ok else 'FAIL'}\n"
        f"  disagree with this run: {disagree or 'none'} (allowed: {allowed})"
    )
    if missing:
        print(f"  labeled rows missing from {VERDICTS_PATH}: {missing}")
    return 0 if base_ok and committed_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m financial_news.eval_gate")
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("fingerprint-check")
    fp.add_argument("--base", type=Path)
    fp.add_argument("--base-verdicts", type=Path)
    sub.add_parser("update")
    ck = sub.add_parser("check")
    ck.add_argument("--base-verdicts", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = Config()
    if args.command == "fingerprint-check":
        code, needs_eval = fingerprint_check(cfg, args.base, args.base_verdicts)
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as f:
                f.write(f"needs_eval={str(needs_eval).lower()}\n")
        return code
    if args.command == "update":
        return update(cfg)
    return check(cfg, args.base_verdicts)


if __name__ == "__main__":
    sys.exit(main())
