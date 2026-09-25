"""scripts.public_guard: catches forbidden strings before a publish, rather
than transforming them at publish time (第15計画, see public_guard.py's own
docstring for why -- 第14計画's regex-substitution approach repeatedly
produced unnatural prose)."""
from pathlib import Path

from scripts.public_guard import scan


def _write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_clean_tree_reports_no_hits(tmp_path):
    _write(tmp_path, "README.md", "# ClearCut\n\nA background removal tool.\n")
    _write(tmp_path, "src/main.py", "def run():\n    return 1\n")
    assert scan(tmp_path) == []


def test_client_name_is_caught(tmp_path):
    _write(tmp_path, "notes.md", "measured on the nitto clip\n")
    hits = scan(tmp_path)
    assert len(hits) == 1
    assert hits[0][0].name == "notes.md"
    assert hits[0][1] == 1


def test_personal_path_is_caught(tmp_path):
    _write(tmp_path, "script.sh", "cd /home/kohei-shintani/project\n")
    hits = scan(tmp_path)
    assert len(hits) == 1


def test_match_is_case_insensitive(tmp_path):
    _write(tmp_path, "a.py", "# SAKAI fixture\n")
    hits = scan(tmp_path)
    assert len(hits) == 1


def test_multiple_hits_across_files_are_all_reported(tmp_path):
    _write(tmp_path, "a.md", "nitto\n")
    _write(tmp_path, "b.py", "# RTX 5090\n")
    _write(tmp_path, "c/d.json", '{"note": "kohei-shintani"}\n')
    hits = scan(tmp_path)
    assert len(hits) == 3


def test_binary_and_ignored_extensions_are_skipped(tmp_path):
    _write(tmp_path, "asset.gif", "nitto")  # not scanned: wrong suffix
    assert scan(tmp_path) == []


def test_git_and_node_modules_directories_are_skipped(tmp_path):
    _write(tmp_path, ".git/COMMIT_EDITMSG", "nitto\n")
    _write(tmp_path, "web/node_modules/pkg/index.js", "// nitto\n")
    assert scan(tmp_path) == []


def test_word_boundary_does_not_flag_unrelated_words(tmp_path):
    """'stretch' was deliberately dropped from FORBIDDEN_PATTERNS -- it's an
    ordinary English word (e.g. "a stretch of frames") that would false-
    positive constantly; the actual clip identifier was already generalised
    to 'widepose' throughout the dev tree (第15計画 Step3), so there's
    nothing left for a 'stretch' pattern to usefully catch."""
    _write(tmp_path, "a.py", "# this code needs a stretch goal\n")
    assert scan(tmp_path) == []
