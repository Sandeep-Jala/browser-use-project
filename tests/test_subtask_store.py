"""subtask_store tests: context normalization, identity, archive/meta/manifest round-trips.
Library paths are module constants, so tests monkeypatch them onto tmp_path."""
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


def test_subtask_id_ignores_token_names_and_trailing_punctuation():
    """Two decompositions of different parent tasks must converge on ONE library entry
    even when the LLM named the token differently or cut the span before/after a '.'."""
    a = ss.subtask_id("go to Bookkeeping module, search and select {{business}} "
                      "business name", "/admin")
    b = ss.subtask_id("go to Bookkeeping module, search and select {{business_name}} "
                      "business name.", "/admin")
    assert a == b
    # But different wording (a different navigation target) stays a different entry.
    assert a != ss.subtask_id("go to Bookkeeping module and select {{business}}", "/admin")


def test_subtask_id_context_sensitive():
    """The same subtask wording starting on a different page is a DIFFERENT library entry."""
    prompt = "go to inputs section, select sales"
    assert ss.subtask_id(prompt, "/bookkeeping/*/dashboard") != \
        ss.subtask_id(prompt, "/bookkeeping/*/inputs")


# ------------------------------- aux-tab context + tab_url validation -------------------------------


@pytest.mark.parametrize("url,expected", [
    ("https://www.google.com/search?q=x", "www.google.com/search"),
    ("https://DuckDuckGo.com", "duckduckgo.com/"),
    ("https://app.example.com/r/12345/edit", "app.example.com/r/*/edit"),  # still starred
    ("", "/"),
])
def test_normalize_aux_context(url, expected):
    assert ss.normalize_aux_context(url) == expected


def test_aux_and_main_contexts_never_collide():
    # Aux contexts start with a hostname, main contexts with "/" — the self-describing
    # prefix evaluate_gate picks its normalizer by, and a sid discriminator for free.
    assert ss.normalize_aux_context("https://google.com/x").startswith("google.com")
    assert ss.normalize_context("https://google.com/x").startswith("/")
    assert ss.subtask_id("search {{q}}", ss.normalize_aux_context("https://google.com")) \
        != ss.subtask_id("search {{q}}", ss.normalize_context("https://google.com"))


@pytest.mark.parametrize("url,ok", [
    ("https://duckduckgo.com", True),
    ("http://sub.example.co.uk/path?q=1", True),
    ("duckduckgo.com", False),           # no scheme
    ("https://localhost", False),        # dotless host
    ("ftp://files.example.com", False),  # wrong scheme
    ("https://", False),
    (None, False),
    (42, False),
])
def test_is_absolute_http_url(url, ok):
    assert ss.is_absolute_http_url(url) is ok


# ------------------------------- manifest + meta -------------------------------


def test_update_and_load_manifest_roundtrip(library):
    ss.update_manifest("sid1", "select {{business}} business", create=True,
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
    ss.update_manifest("sid1", "prompt", create=True)
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
    ss.update_manifest(sid, "prompt", create=True)
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


# ------- a body on disk is not a cached entry; a learner may not invent one -------
# Run 20260828_124929: a run died between save_steps and update_manifest, leaving a steps
# file with no manifest entry. has_script said True, so the un-committed script replayed —
# ticking a previous run's employee and reporting ok=True. The end-title learner then
# conjured a manifest entry out of its single field, whose `end_title` was a leftover OTP
# portal tab's, making a wrong page the expected end state of an unrelated subtask.


def test_an_unregistered_body_is_not_replayable(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "manifest.json")
    ss.steps_path("orphan").write_text("[]")

    assert ss.has_script("orphan") is False, "an orphan body must never replay"

    ss.update_manifest("orphan", "do the thing", create=True, context="/x", steps=1)
    assert ss.has_script("orphan") is True


def test_a_learner_cannot_register_an_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "manifest.json")

    # exactly the end-title learner's call: one field, no create
    ss.update_manifest("ghost", "do the thing", end_title="Employee Approval Request")
    assert "ghost" not in ss.load_manifest()

    # ...but it may refine an entry the commit path registered
    ss.update_manifest("ghost", "do the thing", create=True, context="/x", steps=2)
    ss.update_manifest("ghost", "do the thing", end_title="Invoices - App")
    entry = ss.load_manifest()["ghost"]
    assert entry["end_title"] == "Invoices - App"
    assert entry["context"] == "/x"       # the registered fields survive the refinement

