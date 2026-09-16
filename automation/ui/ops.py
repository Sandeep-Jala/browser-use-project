"""Everything the UI *does*, with no `gradio` import anywhere in the file.

This module is the boundary that keeps the UI testable. `gradio_app.py` above it is layout and
event wiring; the modules beside it (`store`, `runs`, `uploads`, `supervisor`) are the framework.
Here is the glue that used to live inline in `app.py`'s request handlers — table shaping, the
save/delete/import actions, and five rules that are LOGIC rather than HTTP and would otherwise have
been deleted along with the route table:

1. `refuse_key_collision` — a `prompts/<key>.yaml` whose key matches a `tasks.yaml` key is silently
   ignored by the loader (tasks.yaml wins), so creating one must be refused, not written.
2. `curated` — a broken `tasks.yaml` must not take the UI down.
3. `model_from_form` — the `.strip()` on textarea text happens HERE, at the edge. Not in
   `model.py`, which is a faithful mapping: rewording a step re-hashes its identity
   (`subtask_store.subtask_id` is a sha256 of the normalised text) and orphans the `library/`
   recording it would otherwise have replayed for free.
4. `segment_rows` — a failed segment BREAKS the loop in `run_hybrid_task`, so planned steps after
   it never ran. They are "not reached", never "waiting", or a finished failed run looks like it
   is still going. This used to be written twice, once in `render.py` and once in `app.js`.
5. `report_path` — `RUN_ID_RE` + `_FILENAME_RE` + `.resolve()` containment, in that order, with the
   resolve BEFORE the containment test so a symlink out of the run directory is caught.

Actions return `(ok, message, ...)` rather than raising, so the wiring layer stays trivial and the
tests can assert on the message a user would actually read.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from automation.pipeline import files as pfiles
from automation.tasks import load_tasks
from automation.ui import store, uploads
from automation.ui.model import (CheckModel, PromptModel, StepModel, derived_prompt)
from automation.ui.runs import RUN_ID_RE, RunSummary, scan_runs

log = logging.getLogger("framework.ui")

# Recovered verbatim from app.py, itself recovered from the removed dashboard: no separators, no
# dots-only names, nothing that could walk out of a run directory even before the resolve() check.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")

STEP_TYPE_LABELS = {"action": "Action", "check": "Check", "conditional": "Conditional"}
LABEL_TO_STEP_TYPE = {v: k for k, v in STEP_TYPE_LABELS.items()}


# ── the registry ──────────────────────────────────────────────────────────────────────────

def curated() -> dict[str, Any]:
    """The read-only `tasks.yaml` half of the registry, keyed. Loaded with an explicit path so the
    UI's own prompts are not mixed in, and behind a bare except so a broken `tasks.yaml` costs the
    user a Prompts tab with one table instead of a server that will not start."""
    from automation import tasks as tasks_mod
    try:
        return load_tasks(tasks_mod.TASKS_FILE)
    except Exception:  # noqa: BLE001 - a broken tasks.yaml must not take the UI down
        log.exception("could not read %s", tasks_mod.TASKS_FILE)
        return {}


def refuse_key_collision(key: str) -> str:
    """"" if the key is free, else the sentence explaining why it cannot be used."""
    if key in curated():
        return (f"{key} is already a built-in task in tasks.yaml. A prompt file with that name "
                f"would be ignored — tasks.yaml wins — so pick a different name.")
    return ""


def prompt_keys() -> list[str]:
    rows, _ = store.list_prompts()
    return [r.key for r in rows]


def run_options() -> list[str]:
    """Everything runnable, prompts first. The Run tab's dropdown."""
    return prompt_keys() + sorted(curated())


# ── the editor's form dict ────────────────────────────────────────────────────────────────
#
# `@gr.render` needs a plain, JSON-ish structure it can diff, and every step and check needs a
# STABLE id so a reorder moves a component's key with it rather than by position. `PromptModel` has
# neither, so the form dict is a third representation sitting between the widgets and `to_entry`.
# `test_ui_editor.py` pins the round-trip through it, because a single character altered here
# orphans a library recording exactly the way model.py's docstring warns about.

_uid_counter = 0


def _uid(prefix: str) -> str:
    global _uid_counter
    _uid_counter += 1
    return f"{prefix}{_uid_counter}"


def blank_check() -> dict[str, Any]:
    return {"uid": _uid("c"), "kind": "text_visible", "text": "", "timeout_s": None}


def blank_step() -> dict[str, Any]:
    return {
        "uid": _uid("s"), "prompt": "", "step_type": "action",
        "probe_kind": "text_visible", "probe_text": "", "probe_timeout_s": None,
        "verify": [], "marker": "", "tab_url": "", "allow_write_refusal": False,
        "values": {}, "declared_kind": None, "extra": {}, "advanced_combination": False,
    }


def model_to_form(model: PromptModel) -> dict[str, Any]:
    """`PromptModel` → the form dict, minting a uid per step and per check."""
    return {
        "key": model.key, "marker": model.marker, "tags": ", ".join(model.tags),
        "prompt": model.prompt,
        "steps": [{
            "uid": _uid("s"), "prompt": s.prompt, "step_type": s.step_type,
            "probe_kind": s.probe_kind, "probe_text": s.probe_text,
            "probe_timeout_s": s.probe_timeout_s,
            "verify": [{"uid": _uid("c"), "kind": c.kind, "text": c.text,
                        "timeout_s": c.timeout_s} for c in s.verify],
            "marker": s.marker, "tab_url": s.tab_url,
            "allow_write_refusal": s.allow_write_refusal,
            "values": dict(s.values), "declared_kind": s.declared_kind, "extra": dict(s.extra),
            "advanced_combination": s.advanced_combination,
        } for s in model.steps],
        "extra": dict(model.extra),
    }


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def model_from_form(form: dict[str, Any]) -> PromptModel:
    """The form dict → `PromptModel`.

    Trimming happens HERE, at the edge, rather than in the model: a browser textarea adds trailing
    whitespace, and `PromptModel` is a faithful mapping that must not rewrite text.
    """
    form = form if isinstance(form, dict) else {}
    tags = form.get("tags") or ""
    if isinstance(tags, str):
        tags = tuple(t.strip() for t in tags.split(",") if t.strip())
    else:
        tags = tuple(str(t).strip() for t in tags if str(t).strip())

    steps: list[StepModel] = []
    for s in form.get("steps") or []:
        s = s if isinstance(s, dict) else {}
        steps.append(StepModel(
            prompt=str(s.get("prompt") or "").strip(),
            step_type=str(s.get("step_type") or "action"),
            probe_kind=str(s.get("probe_kind") or "text_visible"),
            probe_text=str(s.get("probe_text") or "").strip(),
            probe_timeout_s=_num(s.get("probe_timeout_s")),
            verify=[CheckModel(kind=str(c.get("kind") or "text_visible"),
                               text=str(c.get("text") or "").strip(),
                               timeout_s=_num(c.get("timeout_s")))
                    for c in (s.get("verify") or []) if isinstance(c, dict)],
            marker=str(s.get("marker") or "").strip(),
            tab_url=str(s.get("tab_url") or "").strip(),
            allow_write_refusal=bool(s.get("allow_write_refusal")),
            values={str(k).strip(): str(v).strip()
                    for k, v in (s.get("values") or {}).items() if str(k).strip()},
            declared_kind=s.get("declared_kind"),
            extra=dict(s.get("extra") or {}),
            advanced_combination=bool(s.get("advanced_combination")),
        ))

    return PromptModel(
        key=str(form.get("key") or "").strip(),
        marker=str(form.get("marker") or "").strip(),
        tags=tags,
        prompt=str(form.get("prompt") or "").strip(),
        steps=steps,
        extra=dict(form.get("extra") or {}),
    )


# ── step-list mutators (pure; the render calls these) ──────────────────────────────────────

def add_step(form: dict[str, Any]) -> dict[str, Any]:
    form["steps"] = list(form.get("steps") or []) + [blank_step()]
    return form


def remove_step(form: dict[str, Any], i: int) -> dict[str, Any]:
    steps = list(form.get("steps") or [])
    if 0 <= i < len(steps):
        steps.pop(i)
    form["steps"] = steps
    return form


def move_step(form: dict[str, Any], i: int, delta: int) -> dict[str, Any]:
    steps = list(form.get("steps") or [])
    j = i + delta
    if 0 <= i < len(steps) and 0 <= j < len(steps):
        steps[i], steps[j] = steps[j], steps[i]
    form["steps"] = steps
    return form


def add_check(form: dict[str, Any], i: int) -> dict[str, Any]:
    steps = list(form.get("steps") or [])
    if 0 <= i < len(steps):
        steps[i]["verify"] = list(steps[i].get("verify") or []) + [blank_check()]
    form["steps"] = steps
    return form


def remove_check(form: dict[str, Any], i: int, j: int) -> dict[str, Any]:
    steps = list(form.get("steps") or [])
    if 0 <= i < len(steps):
        checks = list(steps[i].get("verify") or [])
        if 0 <= j < len(checks):
            checks.pop(j)
        steps[i]["verify"] = checks
    form["steps"] = steps
    return form


# ── validation ────────────────────────────────────────────────────────────────────────────

def validation_report(model: PromptModel) -> tuple[str, dict[int, str], str]:
    """`(banner_markdown, {step_index: markdown}, derived_prompt)`.

    Runs the framework's real loaders, so the editor cannot save a prompt the CLI would reject.
    Writes nothing.
    """
    errors = store.validate_model(model)
    per_step: dict[int, str] = {}
    globals_: list[str] = []
    for e in errors:
        if e.step is None:
            globals_.append(e.message)
        else:
            per_step[e.step] = f"⚠️ {e.message}" if e.step not in per_step else (
                f"{per_step[e.step]}<br>⚠️ {e.message}")

    derived = ""
    try:
        derived = derived_prompt(model)
    except Exception as exc:  # noqa: BLE001 - a half-typed prompt is not an error yet
        globals_.append(str(exc))

    if derived:
        resolved, problems = pfiles.resolve_prompt_files(derived)
        globals_.extend(problems)
        if resolved and not problems:
            names = ", ".join(Path(p).name for p in resolved)
            globals_.append(f"ℹ️ files this prompt names: {names}")

    banner = ""
    if globals_:
        banner = "\n\n".join(f"⚠️ {g}" if not g.startswith("ℹ️") else g for g in globals_)
    elif not per_step and derived:
        banner = f"✅ Ready. Reads as: *{derived}*"
    return banner, per_step, derived


# ── prompt actions ────────────────────────────────────────────────────────────────────────

def save_prompt(form: dict[str, Any], *, is_new: bool) -> tuple[bool, str, str]:
    model = model_from_form(form)
    if not model.key:
        return False, "a prompt needs a name", ""
    try:
        key = store.slugify(model.key) if is_new else model.key
    except ValueError as exc:
        return False, str(exc), ""
    model.key = key

    if is_new:
        if (refusal := refuse_key_collision(key)):
            return False, refusal, ""
        if store.path_for(key).exists():
            return False, f"a prompt named {key} already exists", ""

    if (errors := store.validate_model(model)):
        first = errors[0]
        where = f"step {first.step + 1}: " if first.step is not None else ""
        return False, f"{where}{first.message}", ""

    path = store.write_prompt(model)
    return True, f"saved {path.name}", key


def import_task(task_key: str, new_name: str) -> tuple[bool, str, str]:
    if not task_key:
        return False, "pick a built-in task to copy", ""
    try:
        key = store.slugify(new_name or task_key)
    except ValueError as exc:
        return False, str(exc), ""
    if (refusal := refuse_key_collision(key)):
        return False, refusal, ""
    if store.path_for(key).exists():
        return False, f"a prompt named {key} already exists", ""
    try:
        store.import_task(task_key, key)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), ""
    return True, f"copied {task_key} to {key}", key


def delete_prompt(key: str) -> tuple[bool, str]:
    if not key:
        return False, "pick a prompt first"
    try:
        path = store.delete_prompt(key)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, f"moved to {path.parent.name}/{path.name}"


# ── tables ────────────────────────────────────────────────────────────────────────────────

def _when(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def _shape(mode: str, steps: int) -> str:
    return f"{steps} steps" if mode == "steps" else "one instruction"


def prompt_rows() -> tuple[list[list[str]], str]:
    """`(rows, broken_markdown)`. A broken file is REPORTED, not skipped — it is the one the user
    most needs to find."""
    rows, broken = store.list_prompts()
    table = [[r.key, _shape(r.mode, r.steps), r.marker or "— off —",
              ", ".join(r.tags), _when(r.modified)] for r in rows]
    note = "\n\n".join(f"⚠️ **{b.key}** will not load: {b.error}" for b in broken)
    return table, note


def task_rows() -> list[list[str]]:
    out = []
    for key, spec in sorted(curated().items()):
        n = len(spec.subtasks or ())
        out.append([key, _shape("steps" if n else "one", n), spec.marker or "— off —",
                    ", ".join(spec.tags)])
    return out


def file_rows() -> list[list[str]]:
    return [[r.name, r.snippet, _bytes(r.size), ", ".join(r.referenced_by) or "—",
             r.warning or ""] for r in uploads.list_uploads()]


def _bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def _secs(s: float | None) -> str:
    if not s:
        return "—"
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f}m"


def run_rows(artifacts_dir: Path, active_run_id: str | None = None) -> list[list[str]]:
    out = []
    for r in scan_runs(artifacts_dir, active_run_id):
        replayed = ("—" if r.subtasks_total is None
                    else f"{r.subtasks_replayed}/{r.subtasks_total}")
        out.append([r.status, r.run_id, _when(r.started_at), (r.task or "")[:80],
                    _secs(r.duration_seconds), str(r.n_steps if r.n_steps is not None else "—"),
                    replayed, f"{r.total_tokens:,}" if r.total_tokens else "—",
                    _bytes(r.size_bytes), r.note or ""])
    return out


def run_summary(artifacts_dir: Path, run_id: str, *, active: bool = False) -> RunSummary | None:
    if not RUN_ID_RE.fullmatch(run_id or ""):
        return None
    d = artifacts_dir / run_id
    if not d.is_dir():
        return None
    from automation.ui.runs import load_run_summary
    return load_run_summary(d, active=active)


def delete_runs(artifacts_dir: Path, ids: list[str]) -> tuple[bool, str]:
    from automation.ui.runs import delete_run
    ok, errs = 0, []
    for run_id in (ids or [])[:200]:
        try:
            delete_run(artifacts_dir, run_id)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{run_id}: {exc}")
    msg = f"deleted {ok} run(s)"
    if errs:
        msg += " — " + "; ".join(errs[:3])
    return not errs, msg


# ── the step table (rule 4) ───────────────────────────────────────────────────────────────

SEGMENT_HEADERS = ["#", "step", "mode", "status", "actions", "secs", "note"]


def segment_rows(progress: dict[str, Any] | None, *, finished: bool) -> list[list[str]]:
    """One row per step, live or finished.

    The rule that matters: a failed segment BREAKS the loop in `run_hybrid_task`, so the planned
    steps after it never ran. They render as "not reached", never "waiting" — otherwise a finished,
    failed run looks like it is still going.
    """
    progress = progress or {}
    segments = progress.get("segments") or []
    planned = progress.get("planned") or []
    rows: list[list[str]] = []

    failed = any(not s.get("ok") for s in segments)
    for s in segments:
        note = ""
        for field, label in (("error", "Error"), ("replay_error", "Replay failed"),
                             ("replay_progress", "Got as far as")):
            if s.get(field):
                note = f"{label}: {s[field]}"
                break
        if not note and s.get("skip_reason"):
            note = f"skipped: {s['skip_reason']}"
        kind = " · judged live" if s.get("kind") == "judge" else ""
        rows.append([
            str(s.get("index", "")), str(s.get("prompt") or "")[:110],
            f"{s.get('mode') or '?'}{kind}",
            "ok" if s.get("ok") else "failed",
            str(s.get("steps_executed", 0)),
            _secs(s.get("duration_seconds")),
            note[:120],
        ])

    label = "not reached" if (failed or finished) else "waiting"
    for i in range(len(segments), len(planned)):
        p = planned[i] or {}
        rows.append([str(i), str(p.get("prompt") or "")[:110], p.get("kind") or "—",
                     label, "—", "—", ""])
    return rows


# ── run artifacts (rule 5) ────────────────────────────────────────────────────────────────

def report_path(artifacts_dir: Path, run_id: str, name: str) -> Path:
    """The file a `/report/<run_id>/<name>` request may have, or a raise.

    Three guards, in order, and the `resolve()` comes BEFORE the containment test so a symlink
    planted inside the run directory cannot point out of it.
    """
    if not RUN_ID_RE.fullmatch(run_id or ""):
        raise ValueError("no such run")
    if not _FILENAME_RE.fullmatch(name or ""):
        raise ValueError("no such file")
    run_dir = (artifacts_dir / run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError("no such run")
    target = (run_dir / name).resolve()
    if not target.is_relative_to(run_dir.resolve()) or not target.is_file():
        raise FileNotFoundError("no such file")
    return target


# ── uploads ───────────────────────────────────────────────────────────────────────────────

def save_upload_path(tmp_path: str, *, overwrite: bool = False) -> tuple[bool, str]:
    """Gradio hands a path into its own cache where `uploads.save_upload` wants an `IO[bytes]`.
    Three lines here rather than a second signature on the module that already works."""
    if not tmp_path:
        return False, "pick a file first"
    p = Path(tmp_path)
    try:
        with p.open("rb") as fh:
            row = uploads.save_upload(p.name, fh, overwrite=overwrite)
    except FileExistsError:
        return False, f"{p.name} already exists — tick 'replace' to overwrite it"
    except (ValueError, OSError) as exc:
        return False, str(exc)
    msg = f"saved {row.name} ({_bytes(row.size)}) — write it in a prompt as {row.snippet}"
    return True, (f"{msg}\n\n⚠️ {row.warning}" if row.warning else msg)


def delete_upload(name: str, *, force: bool = False) -> tuple[bool, str]:
    if not name:
        return False, "pick a file first"
    try:
        path = uploads.delete_upload(name, force=force)
    except uploads.FileInUse as exc:
        return False, str(exc)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return False, str(exc)
    return True, f"moved to {path.parent.name}/{path.name}"
