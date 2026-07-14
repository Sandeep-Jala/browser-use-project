"""subtask_store tests: context normalization, identity, archive/meta/manifest round-trips.
Library paths are module constants, so tests monkeypatch them onto tmp_path (the same
pattern as test_task_store)."""
import json

import pytest

from automation.pipeline import subtask_store as ss


@pytest.fixture
def library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    monkeypatch.setattr(ss, "LIBRARY_DIR", lib)
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", lib / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    lib.mkdir()
    return lib


# ------------------------------- normalize_context -------------------------------


@pytest.mark.parametrize("url,expected", [
    # Origin + query stripped; path lowercased.
    ("https://app.example.com/Bookkeeping/Inputs?tab=2", "/bookkeeping/inputs"),
    # Volatile segments starred: digit runs, GUIDs, long hex.
    ("https://app/x/bookkeeping/12345/inputs/sales", "/x/bookkeeping/*/inputs/sales"),
    ("https://app/r/6f9619ff-8b86-d011-b42d-00c04fc964ff/edit", "/r/*/edit"),
    ("https://app/r/0123456789abcdef0123/edit", "/r/*/edit"),
    # SPA fragment kept and normalized like a path.
    ("https://app/x/bookkeeping/12345/inputs?q=1#Invoices/99", "/x/bookkeeping/*/inputs#invoices/*"),
    # Degenerate inputs stay stable.
    ("https://app", "/"),
    ("", "/"),
])
def test_normalize_context(url, expected):
    assert ss.normalize_context(url) == expected


def test_subtask_id_stable_across_whitespace_and_case():
    a = ss.subtask_id("Select  {{business}} Business", "/bookkeeping")
    b = ss.subtask_id("select {{business}} business", "/bookkeeping")
    assert a == b
    assert len(a) == 16


def test_subtask_id_context_sensitive():
    """The same subtask wording starting on a different page is a DIFFERENT library entry."""
    prompt = "go to inputs section, select sales"
    assert ss.subtask_id(prompt, "/bookkeeping/*/dashboard") != \
        ss.subtask_id(prompt, "/bookkeeping/*/inputs")


# ------------------------------- manifest + meta -------------------------------


def test_update_and_load_manifest_roundtrip(library):
    ss.update_manifest("sid1", "select {{business}} business",
                       params={"business": "290 CREW LIMITED"}, context="/bookkeeping")
    entry = ss.load_manifest()["sid1"]
    assert entry["template_prompt"] == "select {{business}} business"
    assert entry["params"] == {"business": "290 CREW LIMITED"}
    assert entry["context"] == "/bookkeeping"
    assert entry["created"] and entry["updated"]


def test_bump_meta_counts_and_success_resets_failures(library):
    ss.bump_meta("sid1", fail_count=1)
    ss.bump_meta("sid1", fail_count=1)
    assert ss.load_meta("sid1")["fail_count"] == 2
    # A successful use resets the CONSECUTIVE failure counter.
    meta = ss.bump_meta("sid1", uses=1)
    assert meta["uses"] == 1
    assert meta["fail_count"] == 0
    assert meta["last_used"]


def test_archive_if_failing_thresholds(library):
    ss.steps_path("sid1").write_text("[]")
    ss.update_manifest("sid1", "prompt")
    ss.bump_meta("sid1", fail_count=1)
    assert ss.archive_if_failing("sid1", threshold=2) is False
    assert ss.has_script("sid1")

    ss.bump_meta("sid1", fail_count=1)
    assert ss.archive_if_failing("sid1", threshold=2) is True
    assert not ss.has_script("sid1")
    assert "sid1" not in ss.load_manifest()


def test_archive_entry_moves_files_and_drops_registry(library):
    sid = "sid2"
    ss.steps_path(sid).write_text('[{"action": "click"}]')
    ss.template_path(sid).write_text('{"params": {}}')
    ss.recording_path(sid).write_text('{"history": []}')
    ss.update_manifest(sid, "prompt")
    ss.bump_meta(sid, fail_count=3)

    moved = ss.archive_entry(sid)

    assert len(moved) == 3
    assert all(p.parent == library / "archive" for p in moved)
    assert {p.name.split(".")[1] for p in moved} == {"steps", "template", "recording"}
    assert json.loads(next(p for p in moved if ".steps." in p.name).read_text()) == \
        [{"action": "click"}]
    assert not ss.meta_path(sid).exists()
    assert sid not in ss.load_manifest()


def test_archive_entry_missing_is_noop(library):
    assert ss.archive_entry("ghost") == []
    assert not (library / "archive").exists()


# ------------------------------- decomposition cache -------------------------------


def test_decomposition_roundtrip(library):
    data = {"parent_prompt": "do the thing", "source": "llm",
            "subtasks": [{"template_prompt": "do {{what}}", "values": {"what": "the thing"},
                          "marker": None, "postcondition": None}]}
    ss.save_decomposition("tid1", data)
    assert ss.load_decomposition("tid1") == data
    assert ss.load_decomposition("missing") is None
    assert list(ss.all_decompositions()) == ["tid1"]


def test_corrupt_decomposition_is_none(library):
    ss.DECOMPOSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    ss.decomposition_path("bad").write_text("{not json")
    assert ss.load_decomposition("bad") is None
    assert ss.all_decompositions() == {}
