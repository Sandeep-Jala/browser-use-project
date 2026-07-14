"""archive_script + manifest tests (task_store paths are module constants, so tests
monkeypatch them onto tmp_path)."""
import json

import pytest

from automation.pipeline import task_store as ts


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    rec = tmp_path / "recordings"
    monkeypatch.setattr(ts, "RECORDINGS_DIR", rec)
    monkeypatch.setattr(ts, "MANIFEST_PATH", rec / "manifest.json")
    rec.mkdir()
    return rec


def test_archive_script_moves_steps_and_template(recordings):
    tid = "abc123"
    ts.steps_path(tid).write_text('[{"action": "goto", "url": "x"}]')
    ts.template_path(tid).write_text('{"params": {}}')

    moved = ts.archive_script(tid)

    assert not ts.steps_path(tid).exists()
    assert not ts.template_path(tid).exists()
    assert len(moved) == 2
    assert all(p.parent == recordings / "archive" for p in moved)
    assert {p.name.split(".")[1] for p in moved} == {"steps", "template"}
    # Content survives the move.
    steps_archived = next(p for p in moved if ".steps." in p.name)
    assert json.loads(steps_archived.read_text())[0]["action"] == "goto"


def test_archive_script_missing_files_is_a_noop(recordings):
    assert ts.archive_script("nothing-here") == []
    assert not (recordings / "archive").exists()


def test_archive_script_steps_only(recordings):
    tid = "def456"
    ts.steps_path(tid).write_text("[]")
    moved = ts.archive_script(tid)
    assert len(moved) == 1
    assert ".steps." in moved[0].name


def test_update_manifest_reauthor_count_increments(recordings):
    tid = "abc123"
    ts.update_manifest(tid, "some task",
                       reauthor_count=ts.load_manifest().get(tid, {}).get("reauthor_count", 0) + 1)
    ts.update_manifest(tid, "some task",
                       reauthor_count=ts.load_manifest().get(tid, {}).get("reauthor_count", 0) + 1)
    assert ts.load_manifest()[tid]["reauthor_count"] == 2
