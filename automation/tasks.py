"""Declarative task registry, loaded from tasks.yaml at the repo root.

Adding a task is ONE entry in tasks.yaml — no Python change. Per entry: `prompt` (required,
written like a user would type it, WITHOUT login steps — the framework logs in itself);
`marker`, the network ground-truth URL fragment (a successful POST/PUT/PATCH to a URL
containing it proves the record saved) — OMITTED for tasks with no known create-write
(verification / read-only flows), which then run with the gate disabled instead of being
force-failed; `tags` as free grouping metadata; `assertions` for per-task assertion
overrides (see pipeline/assertions.py); and `subtasks` as a rarely-needed escape hatch when
the LLM decomposer keeps splitting a specific task wrongly.

CRITICAL: prompts are identity. subtask_store.task_id hashes the prompt to key the task's
cached subtask decomposition, so editing a prompt's text (whitespace is normalized, words are
not) orphans that cache and forces a fresh LLM decomposition — which may cut the task into
different subtasks and so miss the library entries the old split used.
tests/test_tasks.py pins every prompt's task id for exactly this reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from automation.pipeline.subtask_store import is_absolute_http_url


@dataclass(frozen=True)
class SubtaskDecl:
    """One declared subtask of a task (hybrid subtask engine, see pipeline/hybrid.py).

    `prompt` may carry {{tokens}} whose concrete values live in `values` — the tokenized
    prompt is the subtask's LIBRARY identity, so two tasks that differ only in values share
    one library recording. `marker` marks the save-owning subtask (the parent's create-write
    fires here); `postcondition` is an optional cheap success check for subtasks with no
    write: {"url_contains": "..."} or {"visible": "<selector>"}. `kind` overrides the node
    classification ("action" = replayable, "judge" = cognitive verification, "loop" =
    repeat-until action; judge and loop always run LLM-live and are never cached) —
    normally left None so decompose.node_kind decides. `tab_url` runs the
    subtask in a separate helper tab opened at that URL (same browser context) — the tab is
    closed when the subtask ends and the main app page is never navigated.
    """
    prompt: str
    values: dict[str, str] | None = None
    marker: str | None = None
    postcondition: dict[str, Any] | None = None
    kind: str | None = None
    tab_url: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    key: str
    prompt: str
    marker: str | None = None          # ground-truth URL fragment; None = gate disabled
    assertions: dict[str, Any] | None = None  # per-task assertion overrides; None = defaults
    tags: tuple[str, ...] = ()
    # Escape-hatch override for the hybrid engine; normally None — subtask decomposition
    # is the LLM decomposer's job (computed once per prompt, cached under
    # decompositions/<tid>.json, regenerable with --redecompose). Declare subtasks here
    # only when the decomposer keeps splitting a specific task wrongly.
    subtasks: tuple[SubtaskDecl, ...] | None = None


TASKS_FILE = Path("tasks.yaml")


def _spec_from_entry(key: str, entry: Any) -> TaskSpec:
    """Materialize one tasks.yaml entry into a TaskSpec. Bad entries fail loud — a broken
    registry must be caught at load, not as a silent no-marker/no-prompt run."""
    if not isinstance(entry, dict) or not str(entry.get("prompt") or "").strip():
        raise ValueError(f"tasks.yaml entry {key!r} must be a mapping with a non-empty 'prompt'")
    subtasks = None
    if entry.get("subtasks"):
        subtasks = tuple(
            SubtaskDecl(prompt=str(d["prompt"]), values=d.get("values"),
                        marker=d.get("marker"), postcondition=d.get("postcondition"),
                        kind=d.get("kind"), tab_url=d.get("tab_url"))
            for d in entry["subtasks"]
        )
        for i, s in enumerate(subtasks):
            if s.tab_url is not None and not is_absolute_http_url(s.tab_url):
                raise ValueError(
                    f"tasks.yaml entry {key!r} subtask {i}: tab_url must be an absolute "
                    f"http(s) URL, got {s.tab_url!r}")
    return TaskSpec(
        key=key,
        # Collapse the YAML block-scalar line wrapping. task_id normalizes whitespace the
        # same way, so re-wrapping a prompt in the file can never change its identity.
        prompt=" ".join(str(entry["prompt"]).split()),
        marker=entry.get("marker") or None,
        assertions=entry.get("assertions"),
        tags=tuple(str(t) for t in (entry.get("tags") or ())),
        subtasks=subtasks,
    )


def load_tasks(path: str | Path | None = None) -> dict[str, TaskSpec]:
    """Read the YAML registry -> {key: TaskSpec}, keys lowercased.

    A missing file is an EMPTY registry, not an error — free-text prompts still run. A
    malformed file or entry raises ValueError so the CLI fails before logging in.
    """
    p = Path(path) if path is not None else TASKS_FILE
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a YAML mapping of task-key -> fields")
    return {str(key).strip().lower(): _spec_from_entry(str(key).strip().lower(), entry)
            for key, entry in data.items()}


def resolve_task(raw: str) -> TaskSpec:
    """Resolve --task input to a TaskSpec: a key from tasks.yaml, or a free-text prompt
    (must contain a space so a typo'd key is not silently run as a one-word prompt).

    A free-text prompt gets NO marker: a new task may legitimately fire no create-write
    (verification / read-only flows), and a guessed marker would force-fail an honest
    success. Pass --marker <fragment> to enable the network gate for an ad-hoc write task.
    Raises ValueError for an unknown key so the CLI can print the known keys.
    """
    tasks = load_tasks()
    key = raw.strip().lower()
    if key in tasks:
        return tasks[key]
    if " " in raw:
        return TaskSpec(key="adhoc", prompt=raw.strip(), marker=None)
    known = ", ".join(sorted(tasks)) or f"(none — is {TASKS_FILE} missing?)"
    raise ValueError(
        f"Unknown TASK key {raw!r}. Known keys: {known}.\n"
        'Or pass a full prompt: --task "go to Bookkeeping module, ..."'
    )


