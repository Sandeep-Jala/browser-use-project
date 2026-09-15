"""The `prompts/` data layer: list, read, write, delete, duplicate, import, validate.

One file per prompt, `prompts/<key>.yaml`, holding a single entry in the exact `tasks.yaml` schema.
The filename IS the key — that is what lets the UI address a prompt for edit and delete — and
`tasks._load_prompt_files` enforces the match on the way back in.

`tasks.yaml` is never written. Its ~1576 lines carry hand-written comments recording why each
marker and slice is worded as it is, and no YAML writer preserves them.

Validation runs the REAL loaders rather than reimplementing them (`_spec_from_entry`, which pulls
in `checks.parse_verify`/`parse_probe`, plus `decompose._validate` for the Tier-1 token rules), so
the editor cannot produce a prompt the CLI would then reject. The two rules added on top are ones
no loader can catch: an uppercase `{{Token}}` (which substitutes fine and then silently forks the
step's identity) and an empty step.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from automation import tasks as tasks_mod
from automation.pipeline.subtask_store import TOKEN_RE, task_id
from automation.tasks import _spec_from_entry
from automation.ui.model import PromptModel, derived_prompt, from_entry, to_entry

# Any {{name}} the LOADER would substitute. Deliberately wider than TOKEN_RE (which is
# lowercase-only) — the gap between the two is the bug this module refuses.
_ANY_TOKEN = re.compile(r"\{\{(\w+)\}\}")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ValidationError:
    message: str
    step: int | None = None      # 0-based step index, or None for a prompt-level problem


@dataclass(frozen=True)
class PromptRow:
    """One row of the prompts list."""

    key: str
    path: Path
    mode: str                    # "steps" | "one"
    steps: int
    marker: str | None
    tags: tuple[str, ...]
    modified: datetime
    task_id: str


@dataclass(frozen=True)
class BrokenPrompt:
    """A prompt file the loader refuses. Listed, not hidden — it is still editable."""

    key: str
    path: Path
    error: str


def _dir() -> Path:
    """Resolved through `tasks.PROMPTS_DIR` every call, not captured at import.

    One configuration point, deliberately: the loader and the store MUST agree on where prompts
    live, and a second copy here would be a second thing to keep in step.
    """
    return tasks_mod.PROMPTS_DIR


def path_for(key: str) -> Path:
    return _dir() / f"{key}.yaml"


def slugify(name: str) -> str:
    """A display name → a task key. Keys are lowercased by the loader, so the slug is too."""
    slug = _SLUG_STRIP.sub("_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"{name!r} has no letters or digits to make a key from")
    return slug


# ── reading ───────────────────────────────────────────────────────────────────────────────

def list_prompts() -> tuple[list[PromptRow], list[BrokenPrompt]]:
    """Every prompt file, and every one that does not load. A broken file is REPORTED, not
    skipped silently — it is the one the user most needs to find."""
    rows: list[PromptRow] = []
    broken: list[BrokenPrompt] = []
    d = _dir()
    if not d.exists():
        return rows, broken
    for f in sorted(d.glob("*.yaml")):
        if f.name.startswith("."):
            continue
        key = f.stem.strip().lower()
        try:
            entry = _read_entry(f, key)
            spec = _spec_from_entry(key, entry, source=str(f))
        except Exception as exc:  # noqa: BLE001 - a broken prompt is a row, not a crash
            broken.append(BrokenPrompt(key=key, path=f, error=str(exc)))
            continue
        rows.append(PromptRow(
            key=key, path=f,
            mode="steps" if spec.subtasks else "one",
            steps=len(spec.subtasks or ()),
            marker=spec.marker, tags=spec.tags,
            modified=datetime.fromtimestamp(f.stat().st_mtime),
            task_id=task_id(spec.prompt),
        ))
    return rows, broken


def read_prompt(key: str) -> PromptModel:
    """The editor's model for a stored prompt. Opens even a file the loader would reject —
    that is the only way to fix one."""
    f = path_for(key)
    return from_entry(key, _read_entry(f, key, lenient=True))


def read_raw(key: str) -> str:
    return path_for(key).read_text(encoding="utf-8")


def _read_entry(f: Path, key: str, *, lenient: bool = False) -> dict[str, Any]:
    data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or len(data) != 1:
        if lenient:
            return data if isinstance(data, dict) else {}
        raise ValueError(f"{f} must hold exactly one task entry")
    raw_key, entry = next(iter(data.items()))
    if not lenient and str(raw_key).strip().lower() != key:
        raise ValueError(f"{f} holds the key {raw_key!r}, which does not match its filename")
    return entry if isinstance(entry, dict) else {}


# ── writing ───────────────────────────────────────────────────────────────────────────────

def write_prompt(model: PromptModel) -> Path:
    """Write `prompts/<key>.yaml` atomically (temp + replace), so `load_tasks` — which may be
    reading at any moment from the run subprocess — never sees a half file."""
    if not model.key:
        raise ValueError("a prompt needs a key")
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    entry = {model.key: to_entry(model)}
    # default_flow_style=False and allow_unicode keep the file readable by hand; sort_keys=False
    # preserves the order to_entry chose (prompt first on every step).
    text = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=96,
                          default_flow_style=False)
    target = path_for(model.key)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def delete_prompt(key: str) -> Path:
    """Move the file to `prompts/.trash/<key>.<timestamp>.yaml`.

    Trashed, not unlinked: a prompt is hand-authored work and the UI's delete button is one
    click. `.trash/` is gitignored and `load_tasks` skips it.
    """
    src = path_for(key)
    if not src.is_file():
        raise FileNotFoundError(f"no such prompt: {key}")
    trash = _dir() / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{key}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    shutil.move(str(src), str(dest))
    return dest


def duplicate_prompt(key: str, new_key: str) -> Path:
    return write_prompt(_rekeyed(read_prompt(key), new_key))


def import_task(task_key: str, new_key: str) -> Path:
    """Copy a curated `tasks.yaml` entry into an editable prompt, VERBATIM.

    Verbatim matters: the copy must hash to the same task id and the same subtask ids as the
    original, or importing a working task would orphan every recording it already has.
    """
    data = yaml.safe_load(tasks_mod.TASKS_FILE.read_text(encoding="utf-8")) or {}
    entry = next((v for k, v in data.items() if str(k).strip().lower() == task_key), None)
    if entry is None:
        raise KeyError(f"no such task in {tasks_mod.TASKS_FILE}: {task_key}")
    return write_prompt(_rekeyed(from_entry(task_key, entry), new_key))


def _rekeyed(model: PromptModel, new_key: str) -> PromptModel:
    """A copy under a new key. The key is NOT part of any hash (identity is the prompt text), so
    renaming is identity-free — worth stating, since nothing else about this codebase is."""
    model.key = new_key
    return model


# ── validation ────────────────────────────────────────────────────────────────────────────

def validate_model(model: PromptModel) -> list[ValidationError]:
    """Every reason the engine, or identity, would reject this prompt. Writes nothing."""
    errors: list[ValidationError] = []

    for i, step in enumerate(model.steps):
        if not step.prompt.strip():
            errors.append(ValidationError("this step has no instruction text", step=i))
        # The asymmetry no loader catches: tasks._DECL_TOKEN substitutes any \w+ name, but
        # subtask_store.TOKEN_RE only erases lowercase ones from normalize_template — so an
        # uppercase token stays in the text that gets hashed and forks this step away from the
        # identical lowercase version, for good.
        for name in _ANY_TOKEN.findall(step.prompt):
            if not TOKEN_RE.fullmatch("{{" + name + "}}"):
                errors.append(ValidationError(
                    f"the value name {{{{{name}}}}} must be lowercase "
                    f"({name.lower()!r}) — an uppercase name is substituted correctly but is "
                    f"then baked into this step's identity, so it can never share a recording "
                    f"with the same step written in lowercase", step=i))

    if not model.steps and not model.prompt.strip():
        errors.append(ValidationError("a prompt needs either an instruction or at least one step"))

    if errors:
        return errors   # the loaders would only repeat these in less useful language

    # Now the real thing: exactly what the CLI does at load time.
    try:
        _spec_from_entry(model.key or "untitled", to_entry(model),
                         source=str(path_for(model.key or "untitled")))
    except Exception as exc:  # noqa: BLE001 - any loader refusal is an editor error
        errors.append(_attributed(str(exc)))
        return errors

    errors.extend(_tier1_errors(model))
    return errors


def _tier1_errors(model: PromptModel) -> list[ValidationError]:
    """The declared-split rules `decompose._validate` applies to authored subtasks (token closure,
    every value present in the parent prompt). Checked here so a prompt cannot pass the editor and
    then silently degrade to a whole-prompt fallback at run time."""
    if not model.steps:
        return []
    from automation.pipeline import decompose
    raw = [{"template_prompt": s.prompt, "values": dict(s.values), "marker": s.marker or None,
            "tab_url": s.tab_url or None}
           for s in model.steps]
    try:
        # NOTE the polarity: _validate returns None when the split is SOUND and the reason
        # string when it is not. Its own message names the subtask and the exact problem
        # ("subtask 2: tokens [...] do not close over values [...]"), which is better than
        # anything paraphrased here — so pass it through verbatim.
        reason = decompose._validate(raw, derived_prompt(model), authored=True)
    except Exception:  # noqa: BLE001 - a validator crash must not block saving
        return []
    return [] if reason is None else [_attributed(reason)]


_SUBTASK_IN_MESSAGE = re.compile(r"subtask (\d+)")


def _attributed(message: str) -> ValidationError:
    """Loader messages embed `subtask {i}`, so the editor can show them on the step that raised
    instead of as an anonymous banner."""
    m = _SUBTASK_IN_MESSAGE.search(message)
    return ValidationError(message, step=int(m.group(1)) if m else None)
