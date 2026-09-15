"""Auto Agent: the files a prompt names.

A prompt references an upload by BARE BASENAME and the framework resolves it against
`automation/uploads/` before login, refusing the whole run if it cannot (`__main__.py:107`). So the
upload page's job is not just to store a file — it is to guarantee the prompt side will be able to
see it.

The check that does the real work is `referenced_files(name) == [name]`: the prompt-side scanner
has a lookbehind (`(?<![\\w/\\\\.:\\-])`) that makes a bare name invisible in URL-ish contexts, and
a name containing a space only resolves when QUOTED. A file can therefore sit in the directory,
correctly named, and still be unreachable from any prompt. Better to find that out at upload time
and hand the user the exact snippet to paste.

Two more rules inherited rather than invented:

* **A 0-byte file counts as missing** — `find_file` tests `st_size > 0` deliberately, because this
  repo lives on an iCloud-synced Desktop where an undownloaded placeholder is a real, empty file.
* **Deletes trash rather than unlink**, and refuse outright while a prompt still references the
  file, because that reference is what a run checks before it even logs in.
"""
from __future__ import annotations

import io

import pytest

from automation import tasks as tasks_mod
from automation.pipeline import files as pfiles
from automation.ui import store, uploads
from automation.ui.model import PromptModel, StepModel


@pytest.fixture()
def uploads_dir(tmp_path, monkeypatch):
    d = tmp_path / "uploads"
    d.mkdir()
    # One setattr: uploads._dir() resolves through files.UPLOADS_DIR, the same global that
    # find_file and resolve_prompt_files read.
    monkeypatch.setattr(pfiles, "UPLOADS_DIR", d)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    monkeypatch.setattr(tasks_mod, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(tasks_mod, "TASKS_FILE", tmp_path / "tasks.yaml")
    return d


def _save(name: str, data: bytes = b"a,b\n1,2\n"):
    return uploads.save_upload(name, io.BytesIO(data))


# ── accepting a file ──────────────────────────────────────────────────────────────────────

def test_a_csv_is_saved_and_reported(uploads_dir):
    saved = _save("Employees.csv")
    assert (uploads_dir / "Employees.csv").is_file()
    assert saved.name == "Employees.csv" and saved.size > 0
    assert saved.visible_to_parser is True


def test_the_snippet_is_what_a_prompt_should_actually_say(uploads_dir):
    """Plain name for a plain name; QUOTED when it has a space, because the bare-name scanner
    cannot see a name with a space in it at all."""
    assert _save("Employees.csv").snippet == "Employees.csv"
    assert _save("Staff List 2026.csv").snippet == "'Staff List 2026.csv'"


def test_a_name_the_prompt_scanner_cannot_see_is_flagged(uploads_dir):
    """The file saves — refusing it would be worse, since it may be referenced by some other
    means — but the UI must not pretend a prompt can name it."""
    saved = _save("-weird.csv")
    assert saved.visible_to_parser is False
    assert "cannot" in (saved.warning or "").lower() or "not" in (saved.warning or "").lower()


def test_an_upload_is_atomic(uploads_dir):
    """Streamed to a `.part` and then replaced: a half-written file with a non-zero size is
    exactly what `find_file` would accept, and a run would then hand it to the browser."""
    _save("Employees.csv", b"x" * 4096)
    assert sorted(f.name for f in uploads_dir.iterdir()) == ["Employees.csv"]


# ── refusing a file ───────────────────────────────────────────────────────────────────────

def test_an_unknown_extension_is_refused(uploads_dir):
    with pytest.raises(ValueError, match="extension"):
        _save("notes.exe")
    assert list(uploads_dir.iterdir()) == []


def test_a_path_in_the_name_is_refused_with_the_frameworks_own_wording(uploads_dir):
    for bad in ("../escape.csv", "sub/dir.csv", "/abs.csv"):
        with pytest.raises(ValueError, match="bare file name"):
            _save(bad)
    assert list(uploads_dir.iterdir()) == []


def test_an_empty_file_is_refused(uploads_dir):
    """`find_file` requires st_size > 0, so an empty file would be reported as MISSING at run
    time — a confusing way to discover it."""
    with pytest.raises(ValueError, match="empty"):
        _save("Employees.csv", b"")
    assert list(uploads_dir.iterdir()) == []


def test_an_oversized_file_is_refused(uploads_dir, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 16)
    with pytest.raises(ValueError, match="too large"):
        _save("Employees.csv", b"x" * 32)
    assert list(uploads_dir.iterdir()) == []


def test_an_existing_name_is_refused_unless_overwrite_is_asked_for(uploads_dir):
    _save("Employees.csv")
    with pytest.raises(FileExistsError):
        _save("Employees.csv")
    uploads.save_upload("Employees.csv", io.BytesIO(b"new,data\n3,4\n"), overwrite=True)
    assert "new,data" in (uploads_dir / "Employees.csv").read_text()


# ── listing ───────────────────────────────────────────────────────────────────────────────

def test_listing_reports_a_zero_byte_placeholder(uploads_dir):
    """An iCloud file that was never downloaded. A run WILL refuse it, so say so here."""
    (uploads_dir / "Ghost.csv").write_bytes(b"")
    (row,) = uploads.list_uploads()
    assert row.name == "Ghost.csv" and row.empty is True
    assert "placeholder" in (row.warning or "").lower()


def test_listing_skips_the_trash_and_part_files(uploads_dir):
    _save("Employees.csv")
    (uploads_dir / ".trash").mkdir()
    (uploads_dir / ".trash" / "old.csv").write_text("x")
    (uploads_dir / ".Employees.csv.part").write_text("x")
    assert [r.name for r in uploads.list_uploads()] == ["Employees.csv"]


# ── cross-reference ───────────────────────────────────────────────────────────────────────

def test_listing_names_the_prompts_that_reference_a_file(uploads_dir):
    _save("Employees.csv")
    store.write_prompt(PromptModel(key="uses_it", steps=[
        StepModel(prompt="Upload Employees.csv and click Submit.")]))
    store.write_prompt(PromptModel(key="does_not", steps=[
        StepModel(prompt="Just click Save.")]))

    by_name = {r.name: r for r in uploads.list_uploads()}
    assert by_name["Employees.csv"].referenced_by == ("uses_it",)


def test_a_quoted_reference_with_spaces_is_found_too(uploads_dir):
    _save("Staff List.csv")
    store.write_prompt(PromptModel(key="uses_it", steps=[
        StepModel(prompt="Upload 'Staff List.csv' and click Submit.")]))
    (row,) = uploads.list_uploads()
    assert row.referenced_by == ("uses_it",)


# ── deleting ──────────────────────────────────────────────────────────────────────────────

def test_delete_trashes_an_unreferenced_file(uploads_dir):
    _save("Employees.csv")
    trashed = uploads.delete_upload("Employees.csv")
    assert not (uploads_dir / "Employees.csv").exists()
    assert trashed.is_file() and trashed.parent.name == ".trash"


def test_delete_refuses_a_referenced_file_unless_forced(uploads_dir):
    _save("Employees.csv")
    store.write_prompt(PromptModel(key="uses_it", steps=[
        StepModel(prompt="Upload Employees.csv and click Submit.")]))

    with pytest.raises(uploads.FileInUse) as exc:
        uploads.delete_upload("Employees.csv")
    assert "uses_it" in str(exc.value)

    uploads.delete_upload("Employees.csv", force=True)
    assert not (uploads_dir / "Employees.csv").exists()


def test_delete_rejects_a_name_that_is_not_a_bare_file(uploads_dir):
    for bad in ("../x.csv", "sub/x.csv"):
        with pytest.raises(ValueError):
            uploads.delete_upload(bad)
