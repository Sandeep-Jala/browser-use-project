"""Prompt-named upload files: extraction + the automation/uploads/ convention.

The live failure this guards (2026-07-22): with no file machinery, the agent passed a
bare RELATIVE name to upload_file; the CDP-attached session is classified remote so
browser-use waved it through, Chrome attached a nonexistent path, and clicking Save made
the page READ the ungranted file — the renderer was killed (RESULT_CODE_KILLED_BAD_MESSAGE,
the "Aw, Snap!" tab). Startup resolution against ONE folder, with loud misses, is the fix.
"""
import pytest

from automation.pipeline import files as pfiles


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(pfiles, "UPLOADS_DIR", d)
    return d


def test_referenced_files_extraction_ignores_urls_and_hosts():
    prompt = ("go to https://www.fakenamegenerator.com/gen-random.csv.php , upload the "
              "file New_Employees_List_-_WI_LTD.csv and 'Staff List 2026.xlsx', then "
              "visit fakenamegenerator.com and save report.data.csv")
    assert pfiles.referenced_files(prompt) == [
        "Staff List 2026.xlsx",                # quoted (spaces) come first
        "New_Employees_List_-_WI_LTD.csv",
        "report.data.csv",                     # dotted names match from their start
    ]
    # A URL path segment must never leak a phantom file reference.
    assert pfiles.referenced_files("open https://host/data.csv now") == []
    assert pfiles.referenced_files("") == []


def test_resolve_prompt_files_hit_miss_and_empty(uploads):
    (uploads / "a.csv").write_text("x,y\n1,2\n")
    (uploads / "empty.csv").write_text("")     # iCloud-placeholder shape: 0 bytes
    ok, problems = pfiles.resolve_prompt_files(
        "upload a.csv then empty.csv then ghost.csv")
    assert ok == [str((uploads / "a.csv").resolve())]
    assert len(problems) == 2
    assert any("empty.csv" in p for p in problems)
    assert any("ghost.csv" in p for p in problems)
    assert all(str(pfiles.UPLOADS_DIR) in p for p in problems)


def test_resolve_rejects_quoted_paths_with_guidance(uploads):
    ok, problems = pfiles.resolve_prompt_files('upload "/tmp/x.csv" and save')
    assert ok == []
    assert len(problems) == 1 and "bare file name" in problems[0]


def test_prompt_without_file_references_resolves_empty(uploads):
    ok, problems = pfiles.resolve_prompt_files(
        "go to Payroll module, search and select WI LTD business name")
    assert ok == [] and problems == []


def test_find_file_uses_basename_only(uploads):
    (uploads / "list.csv").write_text("a,b\n")
    hit = pfiles.find_file("anywhere/else/list.csv")
    assert hit == uploads / "list.csv"
    assert pfiles.find_file("nope.csv") is None


# --------------------- workspace seeding (the cloud-workspaces analog) ---------------------


async def test_seed_workspace_files_registers_text_files_by_basename(tmp_path):
    """The OSS analog of cloud workspaces.upload: the run's upload files are written into
    the agent's FileSystem so upload_file resolves the BASENAME natively (local
    sessions). Binary formats are skipped — they stay on the allowlist path."""
    from types import SimpleNamespace

    from browser_use.filesystem.file_system import FileSystem

    from automation.pipeline.runner import _seed_workspace_files, _workspace_files_note

    src = tmp_path / "uploads"
    src.mkdir()
    (src / "people.csv").write_text("name,age\nAlice,30\n")
    (src / "report.xlsx").write_bytes(b"\x50\x4b\x03\x04binary")
    files = [str(src / "people.csv"), str(src / "report.xlsx")]

    agent = SimpleNamespace(file_system=FileSystem(tmp_path / "fs"))
    seeded = await _seed_workspace_files(agent, files)
    assert seeded == ["people.csv"]
    assert agent.file_system.get_file("people.csv") is not None   # upload_file can resolve it
    assert agent.file_system.get_file("report.xlsx") is None      # binary: allowlist-only

    note = _workspace_files_note(files)
    assert "pass just the file NAME: people.csv; report.xlsx" in note
    assert str(src / "people.csv") in note                        # absolute-path fallback

    # No file_system on the agent (defensive) -> no-op.
    assert await _seed_workspace_files(SimpleNamespace(), files) == []
