"""Declarative task registry, loaded from tasks.yaml at the repo root.

Adding a task is ONE entry in tasks.yaml — no Python change. Per entry: `prompt` (written
like a user would type it, WITHOUT login steps — the framework logs in itself); `marker`,
the network ground-truth URL fragment (a successful POST/PUT/PATCH to a URL containing it
proves the record saved) — OMITTED for tasks with no known create-write (verification /
read-only flows), which then run with the gate disabled instead of being force-failed;
`tags` as free grouping metadata; `assertions` for per-task assertion overrides (see
pipeline/assertions.py); and `subtasks` as an escape hatch when the LLM decomposer keeps
splitting a specific task wrongly.

A task that declares `subtasks:` may OMIT `prompt:` entirely — the prompt is then DERIVED
by joining the instantiated slices, making the subtask blocks the single edit surface (no
prompt/slice lockstep to maintain). Keeping both is allowed only while they agree
verbatim; a mismatch fails the load loudly, because silent drift between the two texts
would fork the task's identity.

CRITICAL: prompts are identity. subtask_store.task_id hashes the prompt to key the task's
cached subtask decomposition, so editing a prompt's text (whitespace is normalized, words are
not) orphans that cache and forces a fresh LLM decomposition — which may cut the task into
different subtasks and so miss the library entries the old split used.
tests/test_tasks.py pins every prompt's task id for exactly this reason.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from automation.pipeline.checks import Check, parse_probe, parse_verify
from automation.pipeline.subtask_store import is_absolute_http_url

logger = logging.getLogger("framework.tasks")


@dataclass(frozen=True)
class SubtaskDecl:
    """One declared subtask of a task (hybrid subtask engine, see pipeline/hybrid.py).

    `prompt` may carry {{tokens}} whose concrete values live in `values` — the tokenized
    prompt is the subtask's LIBRARY identity, so two tasks that differ only in values share
    one library recording. `marker` marks the save-owning subtask (the parent's create-write
    fires here); `postcondition` is an optional cheap success check for subtasks with no
    write: {"url_contains": "..."} or {"visible": "<selector>"}. `kind` declares the node
    classification — "action" (replayable, the default) or "judge" (a verification, which
    always runs LLM-live and is never cached). It is the ONLY way to mark a verification:
    nothing infers kind from the prompt's wording, so an undeclared slice is an action and
    is recorded. Validated on load; "loop" was removed on 2026-08-28 (a repeat is stated
    with the repeat_click tool). `tab_url` runs the
    subtask in a separate helper tab opened at that URL (same browser context) — the tab is
    closed when the subtask ends and the main app page is never navigated. `verify` is the
    slice's declared deterministic checks (pipeline/checks.py), parsed and token-substituted
    at load time; they gate the segment on top of its base gate and never touch identity.
    `probe` (leading-"If" conditional slices only) is ONE declared check that stands in
    for the agent's live presence judgment: absent resolves the segment as a zero-LLM
    no-op, present runs the branch like a normal action (replayable/committable) —
    recorded TRUE-branch steps only ever run behind a TRUE probe. Like verify, it is
    parsed and token-substituted at load, never touches identity, and is ignored on
    non-conditional slices. `allow_write_refusal` exempts the slice from the window write
    rule (checks.window_write_rollup): a slice whose own wording declares an error branch
    ("click Submit. if it shows an error, click cancel") ends legitimately on a REFUSED
    write, and the rule cannot know that — it judges observed traffic and never reads
    prose, which is exactly what makes it hold for every undeclared task. The refusal is
    still reported under the segment's write_rollup; it just stops failing the segment.
    """
    prompt: str
    values: dict[str, str] | None = None
    marker: str | None = None
    postcondition: dict[str, Any] | None = None
    kind: str | None = None
    tab_url: str | None = None
    verify: tuple[Check, ...] | None = None
    probe: Check | None = None
    allow_write_refusal: bool = False


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

# The UI-owned half of the registry: one file per prompt, `prompts/<key>.yaml`, in the SAME schema
# as a tasks.yaml entry. Separate because tasks.yaml is a document — its comments record why each
# marker and slice is worded as it is, and a YAML writer round-tripping it destroys them. One file
# per prompt so the UI can create, edit and delete one without rewriting any other.
#
# A module global (not a constant inlined at the call site) so tests can point it somewhere else,
# the same reason TASKS_FILE is one.
PROMPTS_DIR = Path("prompts")


_DECL_TOKEN = re.compile(r"\{\{(\w+)\}\}")


def _instantiated(decl: SubtaskDecl) -> str:
    """The slice's concrete text: {{tokens}} replaced from its values (unknown tokens are
    left verbatim — _validate rejects them later with the closure error)."""
    values = decl.values or {}
    return _DECL_TOKEN.sub(lambda m: str(values.get(m.group(1), m.group(0))), decl.prompt)


def _parsed_verify(key: str, i: int, d: dict[str, Any], *,
                   source: str = "tasks.yaml") -> tuple[Check, ...] | None:
    """One slice's `verify:` block → validated Check tuple, {{tokens}} substituted from
    the slice's values. Substitution happens HERE (load time) because the downstream
    token grammar (sstore.TOKEN_RE) is lowercase-only and would silently skip uppercase
    names; a token that stays unresolved fails loud — a verify arg has no later closure
    validation, and probing for literal braces would be a silent always-fail."""
    raw = d.get("verify")
    if raw is None:
        return None
    values = d.get("values") or {}

    def _sub(text: str) -> str:
        return _DECL_TOKEN.sub(lambda m: str(values.get(m.group(1), m.group(0))), text)

    if isinstance(raw, (list, tuple)):
        raw = [
            {k: (_sub(v) if isinstance(v, str) else v) for k, v in item.items()}
            if isinstance(item, dict) else item
            for item in raw
        ]
    checks = parse_verify(raw, where=f"{source} entry {key!r} subtask {i} verify")
    for c in checks:
        if _DECL_TOKEN.search(c.arg):
            raise ValueError(
                f"{source} entry {key!r} subtask {i} verify: unresolved token in "
                f"{c.kind} {c.arg!r} — add it to the subtask's values")
    return checks


def _parsed_probe(key: str, i: int, d: dict[str, Any], *,
                  source: str = "tasks.yaml") -> Check | None:
    """One slice's `probe:` mapping → validated Check, {{tokens}} substituted from the
    slice's values — same load-time substitution contract (and rationale) as
    _parsed_verify."""
    raw = d.get("probe")
    if raw is None:
        return None
    values = d.get("values") or {}
    if isinstance(raw, dict):
        raw = {k: (_DECL_TOKEN.sub(lambda m: str(values.get(m.group(1), m.group(0))), v)
                   if isinstance(v, str) else v)
               for k, v in raw.items()}
    check = parse_probe(raw, where=f"{source} entry {key!r} subtask {i} probe")
    if check is not None and _DECL_TOKEN.search(check.arg):
        raise ValueError(
            f"{source} entry {key!r} subtask {i} probe: unresolved token in "
            f"{check.kind} {check.arg!r} — add it to the subtask's values")
    return check


def _parsed_write_waiver(key: str, i: int, d: dict[str, Any], *,
                         source: str = "tasks.yaml") -> bool:
    """One slice's `allow_write_refusal:` → bool. A non-bool fails loud for the same
    reason a bad `kind` does: silently ignoring the typo would re-fail the run for the
    exact reason the declaration exists to prevent, and it would look like it worked."""
    raw = d.get("allow_write_refusal", False)
    if not isinstance(raw, bool):
        raise ValueError(
            f"{source} entry {key!r} subtask {i}: allow_write_refusal must be true or "
            f"false, got {raw!r}")
    return raw


def _spec_from_entry(key: str, entry: Any, *, source: str = "tasks.yaml") -> TaskSpec:
    """Materialize one tasks.yaml entry into a TaskSpec. Bad entries fail loud — a broken
    registry must be caught at load, not as a silent no-marker/no-prompt run."""
    if not isinstance(entry, dict):
        raise ValueError(f"{source} entry {key!r} must be a mapping")
    subtasks = None
    if entry.get("subtasks"):
        subtasks = tuple(
            SubtaskDecl(prompt=str(d["prompt"]), values=d.get("values"),
                        marker=d.get("marker"), postcondition=d.get("postcondition"),
                        kind=d.get("kind"), tab_url=d.get("tab_url"),
                        verify=_parsed_verify(key, i, d, source=source),
                        probe=_parsed_probe(key, i, d, source=source),
                        allow_write_refusal=_parsed_write_waiver(key, i, d, source=source))
            for i, d in enumerate(entry["subtasks"])
        )
        for i, s in enumerate(subtasks):
            # `kind` is the only mechanism now — decompose.node_kind reads the declaration
            # and nothing else — so an unrecognised value must be LOUD. Before 2026-08-28 a
            # typo fell through to wording inference and looked like it worked; today it
            # would silently become a recorded action, which for a verification slice is
            # exactly the bug the declaration exists to prevent.
            if s.kind is not None and s.kind not in ("action", "judge"):
                raise ValueError(
                    f"{source} entry {key!r} subtask {i}: kind must be 'action' or "
                    f"'judge', got {s.kind!r}"
                    + (" — the 'loop' kind was removed on 2026-08-28; state a repeat with "
                       "the repeat_click tool instead" if str(s.kind).lower() == "loop"
                       else ""))
            if s.tab_url is not None and not is_absolute_http_url(s.tab_url):
                raise ValueError(
                    f"{source} entry {key!r} subtask {i}: tab_url must be an absolute "
                    f"http(s) URL, got {s.tab_url!r}")
    # Collapse the YAML block-scalar line wrapping. task_id normalizes whitespace the
    # same way, so re-wrapping a prompt in the file can never change its identity.
    prompt = " ".join(str(entry.get("prompt") or "").split())
    if subtasks:
        # Declared slices are the single source of truth: the whole-task prompt is their
        # join, so rewording a slice IS rewording the prompt (same identity semantics).
        derived = " ".join(" ".join(_instantiated(s).split()) for s in subtasks)
        if prompt and prompt != derived:
            diverge = next((i for i, (a, b) in enumerate(
                zip(prompt.split(), derived.split())) if a != b),
                min(len(prompt.split()), len(derived.split())))
            context_p = " ".join(prompt.split()[max(0, diverge - 3):diverge + 5])
            context_d = " ".join(derived.split()[max(0, diverge - 3):diverge + 5])
            raise ValueError(
                f"{source} entry {key!r}: prompt and subtasks disagree around word "
                f"{diverge}: prompt says '…{context_p}…' but the slices join to "
                f"'…{context_d}…'. Drop the 'prompt:' key (it is derived from the "
                f"slices) or fix the diverging slice.")
        prompt = derived
    if not prompt:
        raise ValueError(
            f"{source} entry {key!r} must have a non-empty 'prompt' or 'subtasks'")
    return TaskSpec(
        key=key,
        prompt=prompt,
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
    curated: dict[str, TaskSpec] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p} must be a YAML mapping of task-key -> fields")
        curated = {str(key).strip().lower():
                   _spec_from_entry(str(key).strip().lower(), entry, source=str(p))
                   for key, entry in data.items()}
    if path is not None:
        # An explicit path means "exactly this file" — the contract every other caller and
        # every test relies on. Merging a real directory into it would make those callers
        # depend on whatever happens to be in the working tree.
        return curated
    return {**curated, **_load_prompt_files(curated)}


def _load_prompt_files(curated: dict[str, TaskSpec]) -> dict[str, TaskSpec]:
    """The UI-owned prompts in `PROMPTS_DIR`, one task entry per `<key>.yaml`.

    A bad file is SKIPPED AND LOGGED, not raised. This registry feeds the CLI and the whole
    test suite, so a hard error would let one file in a directory the UI writes to break
    `automation --task invoice` and collection for 1100 tests. Skipping keeps the damage local
    and legible: `resolve_task` then raises "Unknown TASK key" for that one key.

    The one hard error is two PROMPT files claiming the same key — both are UI-owned, there is
    no correct winner to fall back to, and the UI can fix it.
    """
    if not PROMPTS_DIR.exists():
        return {}
    found: dict[str, TaskSpec] = {}
    owners: dict[str, Path] = {}
    for f in sorted(PROMPTS_DIR.glob("*.yaml")):
        if f.name.startswith("."):
            continue
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict) or len(data) != 1:
                raise ValueError(
                    f"{f} must hold exactly ONE task entry (the UI addresses prompts as "
                    f"{PROMPTS_DIR}/<key>.yaml), found {len(data) if isinstance(data, dict) else 0}")
            raw_key, entry = next(iter(data.items()))
            key = str(raw_key).strip().lower()
            if key != f.stem.strip().lower():
                raise ValueError(
                    f"{f} holds the key {raw_key!r}, which does not match its filename — the UI "
                    f"edits and deletes a prompt by its filename, so a mismatched key is "
                    f"unreachable. Rename the file to {key}.yaml or rename the key to {f.stem!r}.")
            spec = _spec_from_entry(key, entry, source=str(f))
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not break the registry
            logger.error("skipping prompt file %s: %s", f, exc)
            continue
        if key in owners:
            raise ValueError(
                f"two prompt files claim the task key {key!r}: {owners[key]} and {f} "
                f"(keys are compared lowercased). Rename one of them.")
        if key in curated:
            # Never let a UI file shadow a curated task: that would silently redirect
            # `automation --task <key>` to whatever someone saved in the browser.
            logger.error(
                "prompt file %s is ignored: the task key %r already belongs to %s, which wins. "
                "Rename the prompt.", f, key, TASKS_FILE)
            continue
        owners[key] = f
        found[key] = spec
    return found


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


