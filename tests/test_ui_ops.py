"""Auto Agent's logic layer — the half of the old route table that was never about HTTP.

`app.py` died with the Starlette UI, and with it would have gone five rules that are LOGIC. They
live in `ops.py` now, and this file is what stops them being deleted twice. Each one exists
because of a concrete failure it prevents, and the docstrings say which.

Nothing here starts a Gradio server. `ops.py` imports no gradio at all, which is the whole point of
the seam: the behaviour is testable as plain functions, and `gradio_app.py` above it is layout that
a construction smoke test is enough to cover.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from automation import tasks as tasks_mod
from automation.pipeline import files as pfiles
from automation.ui import ops, store
from automation.ui.model import PromptModel, StepModel


@pytest.fixture()
def prompts_dir(tmp_path, monkeypatch):
    d = tmp_path / "prompts"
    d.mkdir()
    monkeypatch.setattr(tasks_mod, "PROMPTS_DIR", d)
    return d


@pytest.fixture()
def tasks_file(tmp_path, monkeypatch):
    f = tmp_path / "tasks.yaml"
    f.write_text(yaml.safe_dump({"invoice": {"prompt": "make an invoice", "marker": "Invoices"}}))
    monkeypatch.setattr(tasks_mod, "TASKS_FILE", f)
    return f


@pytest.fixture()
def artifacts(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


def _run_dir(artifacts, run_id="20260101_010101_000001", **report):
    d = artifacts / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.html").write_text("<html>the report</html>")
    (d / "report.json").write_text(json.dumps({
        "task": "t", "is_successful": True, "is_done": True, "assertions_passed": True,
        **report}))
    return d


# ── the import rule ───────────────────────────────────────────────────────────────────────

def test_the_ui_package_never_imports_browser_use():
    """browser_use installs a StreamHandler on the ROOT logger at import time
    (`browser_use/__init__.py` calls `setup_logging()`), which would hijack the server's own
    logging, and it costs seconds to load. The UI needs none of it.

    Run in a FRESH interpreter, because this is a property of the import graph, not of this
    process: any earlier test in the suite has already imported browser_use into `sys.modules`,
    so asserting against the live `sys.modules` passes in isolation and means nothing in a full
    run. (It did exactly that before this was fixed.)

    `gradio_app` is in the list deliberately — gradio drags in fastapi, pandas and
    huggingface_hub, and this pins that none of that chain reaches browser_use either.
    """
    script = (
        "import os, sys\n"
        "os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')\n"
        "for m in ('automation.ui.ops', 'automation.ui.store', 'automation.ui.runs',\n"
        "          'automation.ui.supervisor', 'automation.ui.uploads',\n"
        "          'automation.ui.model', 'automation.ui.paths',\n"
        "          'automation.ui.gradio_app', 'automation.ui.__main__'):\n"
        "    __import__(m)\n"
        "print('LEAKED' if 'browser_use' in sys.modules else 'CLEAN')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=str(__import__("automation.ui.paths", fromlist=["x"]).REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "CLEAN", (
        "something in automation.ui pulled in browser_use — look for a pipeline.hybrid or "
        "pipeline.runner import")


# ── rule 1: a prompt may not take a tasks.yaml key ────────────────────────────────────────

def test_a_prompt_cannot_take_a_curated_task_key(prompts_dir, tasks_file):
    """`_load_prompt_files` resolves a collision in tasks.yaml's favour and only LOGS it, so a
    prompt file named after a built-in task would be written, listed, and then silently never
    run. Refusing at the point of creation is the only place the user finds out."""
    refusal = ops.refuse_key_collision("invoice")
    assert "tasks.yaml wins" in refusal

    ok, msg, _ = ops.save_prompt({"key": "invoice", "steps": [{"prompt": "do a thing"}]},
                                 is_new=True)
    assert not ok and "tasks.yaml wins" in msg
    assert not (prompts_dir / "invoice.yaml").exists(), "refused, so nothing should be written"


def test_a_free_key_is_not_refused(prompts_dir, tasks_file):
    assert ops.refuse_key_collision("something_else") == ""


# ── rule 2: a broken tasks.yaml must not take the UI down ─────────────────────────────────

def test_a_broken_tasks_yaml_does_not_take_the_ui_down(prompts_dir, tasks_file):
    """One malformed built-in registry should cost the user a table, not a server that will not
    start. Everything reading the registry goes through `curated()` for exactly this reason."""
    tasks_file.write_text("this: is: not: valid: yaml: [[[")
    assert ops.curated() == {}
    assert ops.task_rows() == []

    store.write_prompt(PromptModel(key="mine", steps=[StepModel(prompt="still works")]))
    rows, broken = ops.prompt_rows()
    assert [r[0] for r in rows] == ["mine"], "a broken tasks.yaml must not hide the user's prompts"
    assert broken == ""


# ── rule 3: the .strip() lives at the edge, not in the model ──────────────────────────────

def test_textarea_whitespace_is_trimmed_at_the_edge():
    """A browser textarea adds trailing whitespace. It has to be removed BEFORE the text becomes
    a step, because `subtask_id` is a sha256 of the normalized prompt and a stray newline would
    fork a step's identity away from the recording it should have replayed."""
    model = ops.model_from_form({
        "key": " k ", "marker": "  Invoices ", "tags": " a , b ,, ",
        "steps": [{"prompt": "  do the thing  \n", "marker": " M ",
                   "verify": [{"kind": "text_visible", "text": "  Saved \n"}]}]})
    assert model.key == "k"
    assert model.marker == "Invoices"
    assert model.tags == ("a", "b")
    assert model.steps[0].prompt == "do the thing"
    assert model.steps[0].marker == "M"
    assert model.steps[0].verify[0].text == "Saved"


def test_the_model_itself_still_does_not_strip():
    """The trimming belongs to the edge. `StepModel` stays a faithful mapping — if it started
    rewriting text, `from_entry` would silently alter prompts read off disk."""
    assert StepModel(prompt="  padded  ").prompt == "  padded  "


# ── rule 4: a step after a failure was never reached ──────────────────────────────────────

def test_steps_after_a_failure_are_not_reached_not_waiting():
    """A failed segment BREAKS the loop in `run_hybrid_task`, so the planned steps after it never
    ran. Calling them "waiting" makes a finished, failed run look like it is still going — which
    is exactly the misreading that teaches an operator to distrust the status."""
    progress = {
        "segments": [{"index": 0, "prompt": "one", "ok": True, "mode": "replay"},
                     {"index": 1, "prompt": "two", "ok": False, "mode": "authored",
                      "error": "boom"}],
        "planned": [{"prompt": "one"}, {"prompt": "two"}, {"prompt": "three"}],
    }
    rows = ops.segment_rows(progress, finished=False)
    assert [r[3] for r in rows] == ["ok", "failed", "not reached"]
    assert "boom" in rows[1][6]


def test_steps_are_waiting_while_a_run_is_still_healthy():
    progress = {"segments": [{"index": 0, "prompt": "one", "ok": True, "mode": "replay"}],
                "planned": [{"prompt": "one"}, {"prompt": "two"}]}
    assert [r[3] for r in ops.segment_rows(progress, finished=False)] == ["ok", "waiting"]


def test_a_finished_run_never_shows_waiting():
    """Even with every segment green, a finished run has nothing left to wait for."""
    progress = {"segments": [{"index": 0, "prompt": "one", "ok": True}],
                "planned": [{"prompt": "one"}, {"prompt": "two"}]}
    assert [r[3] for r in ops.segment_rows(progress, finished=True)] == ["ok", "not reached"]


def test_no_progress_is_an_empty_table_not_a_crash():
    assert ops.segment_rows(None, finished=False) == []
    assert ops.segment_rows({}, finished=True) == []


# ── rule 5: the run-artifact path guards ──────────────────────────────────────────────────

def test_a_run_file_is_served(artifacts):
    d = _run_dir(artifacts)
    assert ops.report_path(artifacts, d.name, "report.html").read_text() == "<html>the report</html>"


@pytest.mark.parametrize("run_id", ["..", "../../etc", "not_a_run", "20260101_010101",
                                    "20260101_010101_000001x", ""])
def test_a_bad_run_id_cannot_reach_a_file(artifacts, run_id):
    _run_dir(artifacts)
    with pytest.raises((ValueError, FileNotFoundError)):
        ops.report_path(artifacts, run_id, "report.html")


@pytest.mark.parametrize("name", ["../report.json", "a/b.html", "..", "", "re port.html"])
def test_a_filename_outside_the_run_dir_is_refused(artifacts, name):
    d = _run_dir(artifacts)
    with pytest.raises((ValueError, FileNotFoundError)):
        ops.report_path(artifacts, d.name, name)


def test_a_symlink_out_of_the_run_dir_is_refused(artifacts, tmp_path):
    """The resolve() has to come BEFORE the containment test. A name that passes `_FILENAME_RE`
    can still be a symlink pointing anywhere on the disk."""
    secret = tmp_path / "secret.txt"
    secret.write_text("password")
    d = _run_dir(artifacts)
    (d / "escape.txt").symlink_to(secret)
    with pytest.raises((ValueError, FileNotFoundError)):
        ops.report_path(artifacts, d.name, "escape.txt")


# ── the Host-header guard ─────────────────────────────────────────────────────────────────

def test_a_foreign_host_header_is_refused():
    """DNS rebinding: a page on the internet can resolve its own hostname to 127.0.0.1 and then
    talk to whatever is listening. Binding to loopback does not stop that; checking Host does.
    Gradio has no equivalent, which is why this server is mounted on our own app.

    Tested on a bare FastAPI with one dummy route — no Gradio mount — so the guard is pinned
    independently of whatever Gradio does with routing.
    """
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from automation.ui.__main__ import _LocalOnly

    api = FastAPI()

    @api.get("/")
    def home():
        return {"ok": True}

    client = TestClient(_LocalOnly(api))
    assert client.get("/", headers={"Host": "evil.com"}).status_code == 403
    for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:8765"):
        assert client.get("/", headers={"Host": host}).status_code == 200, host


# ── prompt actions ────────────────────────────────────────────────────────────────────────

def test_a_prompt_can_be_created_read_updated_and_deleted(prompts_dir, tasks_file):
    ok, msg, key = ops.save_prompt({"key": "My Prompt", "steps": [{"prompt": "go to the module"}]},
                                   is_new=True)
    assert ok, msg
    assert key == "my_prompt", "the name is slugified into the filename, which IS the key"
    assert (prompts_dir / "my_prompt.yaml").exists()

    form = ops.model_to_form(store.read_prompt("my_prompt"))
    form["steps"][0]["prompt"] = "go to Bookkeeping"
    ok, msg, _ = ops.save_prompt(form, is_new=False)
    assert ok, msg
    assert store.read_prompt("my_prompt").steps[0].prompt == "go to Bookkeeping"

    ok, msg = ops.delete_prompt("my_prompt")
    assert ok and not (prompts_dir / "my_prompt.yaml").exists()
    assert (prompts_dir / ".trash").is_dir(), "deletes are trashed, never unlinked"


def test_two_prompts_cannot_share_a_key(prompts_dir, tasks_file):
    ops.save_prompt({"key": "dup", "steps": [{"prompt": "one"}]}, is_new=True)
    ok, msg, _ = ops.save_prompt({"key": "dup", "steps": [{"prompt": "two"}]}, is_new=True)
    assert not ok and "already exists" in msg


def test_an_invalid_prompt_is_refused_and_says_which_step(prompts_dir, tasks_file):
    ok, msg, _ = ops.save_prompt(
        {"key": "bad", "steps": [{"prompt": "fine"}, {"prompt": "   "}]}, is_new=True)
    assert not ok
    assert "step 2" in msg, f"the message must point at the offending step, got {msg!r}"
    assert not (prompts_dir / "bad.yaml").exists()


def test_an_uppercase_token_is_refused(prompts_dir, tasks_file):
    """`tasks._DECL_TOKEN` substitutes `{{Business}}` happily, but `subtask_store.TOKEN_RE` only
    normalizes lowercase ones — so an uppercase token survives normalization literally and forks
    the step's identity. No loader catches it; the editor has to."""
    ok, msg, _ = ops.save_prompt(
        {"key": "upper", "steps": [{"prompt": "open {{Business}} now"}]}, is_new=True)
    assert not ok and "step 1" in msg


def test_a_curated_task_can_be_copied_into_an_editable_prompt(prompts_dir, tasks_file):
    ok, msg, key = ops.import_task("invoice", "my copy")
    assert ok, msg
    assert key == "my_copy"
    assert store.read_prompt("my_copy").prompt == "make an invoice"


def test_validation_writes_nothing(prompts_dir, tasks_file):
    model = ops.model_from_form({"key": "probe", "steps": [{"prompt": "do a thing"}]})
    banner, per_step, derived = ops.validation_report(model)
    assert derived == "do a thing"
    assert per_step == {}
    assert "Ready" in banner
    assert list(prompts_dir.glob("*.yaml")) == []


def test_validation_reports_the_files_a_prompt_names(prompts_dir, tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    (uploads_dir / "staff.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(pfiles, "UPLOADS_DIR", uploads_dir)

    model = ops.model_from_form({"key": "f", "steps": [{"prompt": "upload staff.csv and save"}]})
    banner, _, _ = ops.validation_report(model)
    assert "staff.csv" in banner

    model = ops.model_from_form({"key": "f", "steps": [{"prompt": "upload missing.csv and save"}]})
    banner, _, _ = ops.validation_report(model)
    assert "missing.csv" in banner and "⚠️" in banner


# ── tables ────────────────────────────────────────────────────────────────────────────────

def test_a_broken_prompt_file_is_reported_not_skipped(prompts_dir, tasks_file):
    """A file the loader refuses is the one the user most needs to find."""
    (prompts_dir / "wrecked.yaml").write_text("wrecked:\n  subtasks: 'not a list'\n")
    store.write_prompt(PromptModel(key="fine", steps=[StepModel(prompt="ok")]))
    rows, broken = ops.prompt_rows()
    assert [r[0] for r in rows] == ["fine"]
    assert "wrecked" in broken


def test_runs_are_listed_newest_first(artifacts):
    _run_dir(artifacts, "20260101_010101_000001")
    _run_dir(artifacts, "20260202_020202_000002")
    assert [r[1] for r in ops.run_rows(artifacts)] == ["20260202_020202_000002",
                                                       "20260101_010101_000001"]


def test_deleting_a_run_is_permanent_and_guarded(artifacts):
    d = _run_dir(artifacts)
    ok, _ = ops.delete_runs(artifacts, [d.name])
    assert ok and not d.exists()

    ok, msg = ops.delete_runs(artifacts, ["../../etc"])
    assert not ok and "../../etc" in msg
