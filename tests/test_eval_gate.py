"""Tests for financial_news.eval_gate."""

import json

import pytest

from financial_news import eval_gate, evaluation
from financial_news.config import Config

# ── fixtures ──────────────────────────────────────────────────────────────────


def _candidate(ticker: str, date: str) -> dict:
    return {"ticker": ticker, "date": date}


def _verdict(ticker: str, date: str, predicted: str, true_label: str = "noise"):
    return {
        "ticker": ticker,
        "date": date,
        "true_label": true_label,
        "predicted": predicted,
        "reasoning": f"VERDICT: {predicted}",
    }


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    """Point every eval path at tmp_path, with two labeled candidates."""
    candidates = tmp_path / "candidates.json"
    labels = tmp_path / "labels.json"
    candidates.write_text(
        json.dumps([_candidate("AAA", "2026-01-02"), _candidate("BBB", "2026-01-02")])
    )
    evaluation.save_labels(
        {
            "AAA:2026-01-02": evaluation.Label("AAA", "2026-01-02", "news_driven"),
            "BBB:2026-01-02": evaluation.Label("BBB", "2026-01-02", "noise"),
        },
        labels,
    )
    monkeypatch.setattr(evaluation, "_CANDIDATES_PATH", candidates)
    monkeypatch.setattr(evaluation, "_LABELS_PATH", labels)
    monkeypatch.setattr(eval_gate, "FINGERPRINT_PATH", tmp_path / "fingerprint.json")
    monkeypatch.setattr(eval_gate, "VERDICTS_PATH", tmp_path / "verdicts.json")
    return tmp_path


def _judge(verdicts: dict[str, bool]):
    """A predict_full_system stand-in returning fixed verdicts by ticker."""

    def predict(candidate, enrich_cfg):
        is_news = verdicts[candidate["ticker"]]
        return is_news, f"VERDICT: {'news_driven' if is_news else 'noise'}"

    return predict


# ── fingerprint ───────────────────────────────────────────────────────────────


def test_committed_fingerprint_is_current():
    """eval/fingerprint.json must match the code. If this fails, re-run the judge:
    ANTHROPIC_API_KEY=<key> uv run python -m financial_news.eval_gate update
    """
    committed = json.loads(eval_gate.FINGERPRINT_PATH.read_text())
    current = eval_gate.code_fingerprint(Config())
    assert {k: committed.get(k) for k in current} == current


def test_normalized_source_ignores_comments_docstrings_and_formatting():
    def a(x):
        """Docstring."""
        # comment
        return x + 1

    def b(x):
        return x+1  # fmt: skip

    def c(x):
        return x + 2

    norm = eval_gate._normalized_source
    assert norm(a).replace("def a", "def f") == norm(b).replace("def b", "def f")
    assert norm(b).replace("def b", "def f") != norm(c).replace("def c", "def f")


def test_verdict_path_follows_calls_into_helpers():
    """Helpers the judge reaches indirectly are hashed without being listed."""
    functions, constants = eval_gate._verdict_path_functions()
    names = {f"{fn.__module__}.{fn.__qualname__}" for fn in functions}
    assert {
        "financial_news.evaluation.predict_full_system",
        "financial_news.enrichment.select_articles",
        "financial_news.sentiment.score_headlines",
        "financial_news.sentiment._compose_text",
    } <= names
    assert all(name.startswith("financial_news.") for name in names)
    assert constants["financial_news.evaluation._MODEL"] == repr(evaluation._MODEL)


def test_resolve_ignores_non_module_attributes():
    import ast

    node = ast.parse("x.y", mode="eval").body
    assert eval_gate._resolve(node, {"x": object()}) is None
    assert eval_gate._resolve(ast.parse("f()", mode="eval").body, {}) is None


def test_normalized_source_keeps_a_body_when_only_a_docstring():
    def only_doc():
        """Nothing else."""

    assert "pass" in eval_gate._normalized_source(only_doc)


@pytest.mark.parametrize(
    "attr, value",
    [("_MODEL", "claude-other"), ("_MAX_TOKENS", 999)],
)
def test_code_fingerprint_changes_with_model_settings(monkeypatch, attr, value):
    before = eval_gate.code_fingerprint(Config())
    monkeypatch.setattr(evaluation, attr, value)
    assert eval_gate.code_fingerprint(Config()) != before


def test_code_fingerprint_changes_with_judge_prompt(monkeypatch):
    before = eval_gate.code_fingerprint(Config())
    monkeypatch.setattr(evaluation, "_NEWS_ZSCORE_CLAUSE", " something else,")
    after = eval_gate.code_fingerprint(Config())
    assert after["judge_system_prompt_sha256"] != before["judge_system_prompt_sha256"]


def test_code_fingerprint_changes_with_selection_settings():
    cfg = Config()
    before = eval_gate.code_fingerprint(cfg)
    cfg.briefing.confidence_threshold = 0.5
    assert (
        eval_gate.code_fingerprint(cfg)["article_selection"]
        != (before["article_selection"])
    )


def test_locked_versions_reads_uv_lock(tmp_path, monkeypatch):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "version = 1\n"
        '[[package]]\nname = "anthropic"\nversion = "1.0"\n'
        '[[package]]\nname = "torch"\nversion = "2.1"\n'
        '[[package]]\nname = "torch"\nversion = "2.0"\n'
        '[[package]]\nname = "numpy"\nversion = "9.9"\n'
    )
    monkeypatch.setattr(eval_gate, "_LOCKFILE_PATH", lock)
    assert eval_gate._locked_versions() == {
        "anthropic": "1.0",
        "torch": "2.0, 2.1",
        "transformers": "not locked",
    }


def test_code_fingerprint_changes_with_library_upgrade(monkeypatch):
    before = eval_gate.code_fingerprint(Config())
    monkeypatch.setattr(
        eval_gate,
        "_locked_versions",
        lambda: {**before["library_versions"], "anthropic": "999.0"},
    )
    assert eval_gate.code_fingerprint(Config()) != before


def test_fingerprint_changes_with_labels_and_candidates(eval_dir):
    before = eval_gate.full_fingerprint(Config())
    (eval_dir / "labels.json").write_text("[]\n")
    (eval_dir / "candidates.json").write_text("[]\n")
    after = eval_gate.full_fingerprint(Config())
    assert after["labels_sha256"] != before["labels_sha256"]
    assert after["candidates_sha256"] != before["candidates_sha256"]


# ── pass mark and comparison ──────────────────────────────────────────────────


@pytest.mark.parametrize("n, allowed", [(0, 2), (43, 2), (100, 2), (101, 3), (500, 10)])
def test_allowed_net_worse_is_max_of_two_and_two_percent(n, allowed):
    assert eval_gate.allowed_net_worse(n) == allowed


def test_compare_verdicts_counts_paired_changes():
    truth = {"a": True, "b": False, "c": True, "d": False, "e": True}
    base = {"a": True, "b": False, "c": False, "d": False}
    new = {"a": False, "b": True, "c": True, "d": False, "e": True}
    result = eval_gate.compare_verdicts(truth, base, new)
    assert result["n"] == 4
    assert result["newly_wrong"] == ["a", "b"]
    assert result["newly_right"] == ["c"]
    assert result["net_worse"] == 1
    assert result["allowed_net_worse"] == 2
    assert result["not_compared"] == ["e"]


def test_run_judge_scores_only_confirmed_labels(eval_dir, monkeypatch):
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": True, "BBB": True})
    )
    labels = evaluation.load_labels(evaluation._LABELS_PATH)
    labels["CCC:2026-01-02"] = evaluation.Label("CCC", "2026-01-02", "ambiguous")
    candidates = evaluation.load_candidates(evaluation._CANDIDATES_PATH)
    candidates.append(_candidate("CCC", "2026-01-02"))
    rows = eval_gate.run_judge(candidates, labels, Config())
    assert [(r["ticker"], r["true_label"], r["predicted"]) for r in rows] == [
        ("AAA", "news_driven", "news_driven"),
        ("BBB", "noise", "news_driven"),
    ]


# ── fingerprint-check ─────────────────────────────────────────────────────────


def _write_fingerprint(path, fp):
    path.write_text(json.dumps(fp))


def test_fingerprint_check_fails_when_committed_is_stale(eval_dir, capsys):
    current = eval_gate.code_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, {**current, "claude_model": "old"})
    assert eval_gate.fingerprint_check(Config(), None) == (1, False)
    assert "out of date for: claude_model" in capsys.readouterr().out


def test_fingerprint_check_without_base_needs_no_run(eval_dir):
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, eval_gate.full_fingerprint(Config()))
    assert eval_gate.fingerprint_check(Config(), None) == (0, False)
    assert eval_gate.fingerprint_check(Config(), eval_dir / "absent.json") == (
        0,
        False,
    )


def test_fingerprint_check_matching_base_needs_no_run(eval_dir):
    fp = eval_gate.full_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, fp)
    _write_fingerprint(eval_dir / "base.json", fp)
    assert eval_gate.fingerprint_check(Config(), eval_dir / "base.json") == (0, False)


def test_fingerprint_check_changed_base_needs_run(eval_dir, capsys):
    fp = eval_gate.full_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, fp)
    _write_fingerprint(eval_dir / "base.json", {**fp, "candidates_sha256": "old"})
    assert eval_gate.fingerprint_check(Config(), eval_dir / "base.json") == (0, True)
    assert "candidates_sha256" in capsys.readouterr().out


def test_fingerprint_check_changed_verdicts_need_run(eval_dir, capsys):
    fp = eval_gate.full_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, fp)
    _write_fingerprint(eval_dir / "base.json", fp)
    eval_gate.VERDICTS_PATH.write_text(json.dumps(_RIGHT))
    (eval_dir / "base_verdicts.json").write_text(json.dumps(_RIGHT[:1]))
    result = eval_gate.fingerprint_check(
        Config(), eval_dir / "base.json", eval_dir / "base_verdicts.json"
    )
    assert result == (0, True)
    assert "stored verdicts" in capsys.readouterr().out


def test_fingerprint_check_unchanged_verdicts_need_no_run(eval_dir):
    fp = eval_gate.full_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, fp)
    _write_fingerprint(eval_dir / "base.json", fp)
    eval_gate.VERDICTS_PATH.write_text(json.dumps(_RIGHT))
    (eval_dir / "base_verdicts.json").write_text(json.dumps(_RIGHT, indent=4))
    result = eval_gate.fingerprint_check(
        Config(), eval_dir / "base.json", eval_dir / "base_verdicts.json"
    )
    assert result == (0, False)


# ── update ────────────────────────────────────────────────────────────────────


def test_update_without_candidates_fails(eval_dir, capsys):
    (eval_dir / "candidates.json").unlink()
    assert eval_gate.update(Config()) == 1
    assert "missing" in capsys.readouterr().out


def test_update_writes_verdicts_and_fingerprint(eval_dir, monkeypatch, capsys):
    eval_gate.VERDICTS_PATH.write_text(
        json.dumps(
            [
                _verdict("AAA", "2026-01-02", "news_driven", "news_driven"),
                _verdict("BBB", "2026-01-02", "noise"),
            ]
        )
    )
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": False, "BBB": False})
    )
    assert eval_gate.update(Config()) == 0
    rows = json.loads(eval_gate.VERDICTS_PATH.read_text())
    assert [r["predicted"] for r in rows] == ["noise", "noise"]
    fp = json.loads(eval_gate.FINGERPRINT_PATH.read_text())
    assert fp == eval_gate.full_fingerprint(Config())
    out = capsys.readouterr().out
    assert "right → wrong: ['AAA:2026-01-02']" in out


def test_update_without_stored_verdicts_skips_comparison(eval_dir, monkeypatch, capsys):
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": True, "BBB": False})
    )
    assert eval_gate.update(Config()) == 0
    assert "Against the stored verdicts" not in capsys.readouterr().out


# ── check ─────────────────────────────────────────────────────────────────────


def _prepare_check(eval_dir, base_rows, committed_rows):
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, eval_gate.full_fingerprint(Config()))
    (eval_dir / "base.json").write_text(json.dumps(base_rows))
    eval_gate.VERDICTS_PATH.write_text(json.dumps(committed_rows))
    return eval_dir / "base.json"


_RIGHT = [
    _verdict("AAA", "2026-01-02", "news_driven", "news_driven"),
    _verdict("BBB", "2026-01-02", "noise"),
]


def test_check_passes_when_verdicts_hold(eval_dir, monkeypatch):
    base = _prepare_check(eval_dir, _RIGHT, _RIGHT)
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": True, "BBB": False})
    )
    assert eval_gate.check(Config(), base) == 0


def test_check_passes_within_the_pass_mark(eval_dir, monkeypatch):
    """Two rows newly wrong is within max(2, 2%) at any small N."""
    committed = [
        _verdict("AAA", "2026-01-02", "noise", "news_driven"),
        _verdict("BBB", "2026-01-02", "news_driven"),
    ]
    base = _prepare_check(eval_dir, _RIGHT, committed)
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": False, "BBB": True})
    )
    assert eval_gate.check(Config(), base) == 0


def test_check_fails_beyond_the_pass_mark(eval_dir, monkeypatch, capsys):
    monkeypatch.setattr(eval_gate, "allowed_net_worse", lambda n: 1)
    committed = [
        _verdict("AAA", "2026-01-02", "noise", "news_driven"),
        _verdict("BBB", "2026-01-02", "news_driven"),
    ]
    base = _prepare_check(eval_dir, _RIGHT, committed)
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": False, "BBB": True})
    )
    assert eval_gate.check(Config(), base) == 1
    assert "Against the base branch's verdicts: FAIL" in capsys.readouterr().out


def test_check_fails_when_committed_verdicts_dont_reproduce(
    eval_dir, monkeypatch, capsys
):
    monkeypatch.setattr(eval_gate, "allowed_net_worse", lambda n: 1)
    hand_edited = [
        _verdict("AAA", "2026-01-02", "noise", "news_driven"),
        _verdict("BBB", "2026-01-02", "news_driven"),
    ]
    base = _prepare_check(eval_dir, _RIGHT, hand_edited)
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": True, "BBB": False})
    )
    assert eval_gate.check(Config(), base) == 1
    assert "Committed verdicts reproduce: FAIL" in capsys.readouterr().out


def test_check_fails_when_committed_verdicts_miss_rows(eval_dir, monkeypatch, capsys):
    base = _prepare_check(eval_dir, _RIGHT, _RIGHT[:1])
    monkeypatch.setattr(
        evaluation, "predict_full_system", _judge({"AAA": True, "BBB": False})
    )
    assert eval_gate.check(Config(), base) == 1
    assert "missing from" in capsys.readouterr().out


def test_check_fails_when_candidates_dont_match_checksum(eval_dir, monkeypatch):
    base = _prepare_check(eval_dir, _RIGHT, _RIGHT)
    (eval_dir / "candidates.json").write_text("[]\n")
    monkeypatch.setattr(evaluation, "predict_full_system", _judge({}))
    assert eval_gate.check(Config(), base) == 1


# ── main ──────────────────────────────────────────────────────────────────────


def test_main_fingerprint_check_writes_github_output(eval_dir, monkeypatch):
    fp = eval_gate.full_fingerprint(Config())
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, fp)
    _write_fingerprint(eval_dir / "base.json", {**fp, "claude_model": "old"})
    output = eval_dir / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    code = eval_gate.main(["fingerprint-check", "--base", str(eval_dir / "base.json")])
    assert code == 0
    assert output.read_text() == "needs_eval=true\n"


def test_main_fingerprint_check_without_github_output(eval_dir, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    _write_fingerprint(eval_gate.FINGERPRINT_PATH, eval_gate.full_fingerprint(Config()))
    assert eval_gate.main(["fingerprint-check"]) == 0


@pytest.mark.parametrize(
    "argv, target",
    [(["update"], "update"), (["check", "--base-verdicts", "b.json"], "check")],
)
def test_main_dispatches(monkeypatch, argv, target):
    called = {}
    monkeypatch.setattr(
        eval_gate, target, lambda *a: called.setdefault(target, a) and 0
    )
    eval_gate.main(argv)
    assert target in called
