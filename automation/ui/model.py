"""The editor's wire model, and the ONLY place it becomes YAML.

Everything the browser sends and receives is a `PromptModel`; everything on disk is a `tasks.yaml`
entry. `from_entry` / `to_entry` are the single crossing, which is what makes the round-trip
testable — and it has to be tested, because a round-trip that alters one character of wording
silently re-hashes a step's identity (`subtask_store.subtask_id` is a sha256 of the normalized
text) and orphans the `library/` recording that step would otherwise have replayed for free.

Three design points worth knowing before editing this file:

**"Step type" collapses two orthogonal schema axes.** The engine reads them independently
(`hybrid` computes `is_judge` from `kind` and `is_conditional` from `probe is not None`):

* `kind` ∈ `action | judge` — is this WORK, or a verification the LLM must judge live?
* `probe` present/absent — does this step RUN AT ALL?

A three-way picker (Action / Check / Conditional) covers every combination anyone authors: the
fourth (judge + probe) is meaningless in practice, since a judge never caches anyway. An imported
entry that has both is PRESERVED and flagged, never silently normalised.

**Unknown keys are preserved.** `tasks._spec_from_entry` reads only the keys it knows and ignores
the rest without complaint, so nothing downstream protects `assertions:` or `postcondition:`.
`extra` carries them through untouched. Neither gets an editor: `postcondition` overlaps `verify`,
and offering two check mechanisms is exactly the confusion this UI exists to remove.

**Absent means absent.** A declaration the user did not make is omitted, never written as a null or
an empty mapping — `kind: null` would fail the loader's own validation and `probe: {}` would fail
the check parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automation.pipeline.checks import _CHECK_KINDS, _CHECK_TIMEOUT_S, _PROBE_TIMEOUT_S

# The five check kinds, in plain English. Labels only ever appear in the UI; the keys are the
# engine's own identifiers from checks._CHECK_KINDS, so this mapping cannot drift from it
# (CHECK_LABELS is asserted complete at import).
CHECK_LABELS: dict[str, str] = {
    "text_visible": "Page shows the text…",
    "text_absent": "Page does NOT show the text…",
    "control_exists": "A control exists…",
    "url_contains": "The URL contains…",
    "write_accepted": "A save was accepted (network write)",
}
assert set(CHECK_LABELS) == set(_CHECK_KINDS), "CHECK_LABELS is out of step with checks.py"

STEP_TYPES = ("action", "check", "conditional")

# Surfaced to the editor so the help text cannot hardcode a number the engine might change.
VERIFY_DEFAULT_TIMEOUT_S = _CHECK_TIMEOUT_S
PROBE_DEFAULT_TIMEOUT_S = _PROBE_TIMEOUT_S


@dataclass
class CheckModel:
    """One `verify:` row, or the single `probe:` check."""

    kind: str = "text_visible"
    text: str = ""
    timeout_s: float | None = None      # None = use the engine's default

    def to_mapping(self, *, default_timeout: float) -> dict[str, Any]:
        out: dict[str, Any] = {self.kind: self.text}
        if self.timeout_s is not None and float(self.timeout_s) != float(default_timeout):
            out["timeout_s"] = self.timeout_s
        return out

    @classmethod
    def from_mapping(cls, raw: Any) -> "CheckModel":
        if not isinstance(raw, dict):
            return cls()
        kind = next((k for k in raw if k in _CHECK_KINDS), "text_visible")
        return cls(kind=kind, text=str(raw.get(kind) or ""),
                   timeout_s=raw.get("timeout_s"))


@dataclass
class StepModel:
    """One step card. `step_type` is the UI's collapse of `kind` + `probe` (see module docstring)."""

    prompt: str = ""
    step_type: str = "action"                  # action | check | conditional
    probe_kind: str = "text_visible"
    probe_text: str = ""
    probe_timeout_s: float | None = None
    verify: list[CheckModel] = field(default_factory=list)
    marker: str = ""
    tab_url: str = ""
    allow_write_refusal: bool = False
    values: dict[str, str] = field(default_factory=dict)
    # Exactly what `kind:` said in the file, or None if it was absent. `step_type` is what the UI
    # shows; this is what gets written back. Keeping them apart matters for `kind: action`, which
    # node_kind treats the same as an absent kind but which an author may have declared
    # deliberately — tasks.yaml has a slice whose comment records exactly that reasoning.
    declared_kind: str | None = None
    # Keys this editor does not own (postcondition, anything added later). Round-tripped verbatim.
    extra: dict[str, Any] = field(default_factory=dict)
    # True when an imported entry declares BOTH kind: judge and a probe — preserved, not normalised.
    advanced_combination: bool = False


@dataclass
class PromptModel:
    """One prompt. Exactly one of `prompt` (one-instruction mode) or `steps` is meaningful."""

    key: str = ""
    marker: str = ""
    tags: tuple[str, ...] = ()
    prompt: str = ""                            # one-instruction mode
    steps: list[StepModel] = field(default_factory=list)   # step-by-step mode
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "steps" if self.steps else "one"


# Keys this module owns at each level; anything else rides `extra`.
_TASK_KEYS = {"prompt", "subtasks", "marker", "tags"}
_STEP_KEYS = {"prompt", "kind", "probe", "verify", "marker", "tab_url",
              "allow_write_refusal", "values"}


def from_entry(key: str, entry: dict[str, Any]) -> PromptModel:
    """A `tasks.yaml` entry → the editor's model. Never raises on odd input: the editor must be
    able to OPEN a file the loader would reject, which is the only way to fix it."""
    entry = entry if isinstance(entry, dict) else {}
    steps: list[StepModel] = []
    for raw in entry.get("subtasks") or []:
        raw = raw if isinstance(raw, dict) else {}
        kind = raw.get("kind")
        probe = raw.get("probe")
        if probe is not None:
            step_type = "conditional"
        elif kind == "judge":
            step_type = "check"
        else:
            step_type = "action"
        probe_model = CheckModel.from_mapping(probe) if probe is not None else CheckModel()
        steps.append(StepModel(
            prompt=str(raw.get("prompt") or ""),
            step_type=step_type,
            probe_kind=probe_model.kind,
            probe_text=probe_model.text,
            probe_timeout_s=probe_model.timeout_s,
            verify=[CheckModel.from_mapping(v) for v in (raw.get("verify") or [])],
            marker=str(raw.get("marker") or ""),
            tab_url=str(raw.get("tab_url") or ""),
            allow_write_refusal=bool(raw.get("allow_write_refusal") or False),
            values={str(k): str(v) for k, v in (raw.get("values") or {}).items()},
            declared_kind=kind if isinstance(kind, str) else None,
            extra={k: v for k, v in raw.items() if k not in _STEP_KEYS},
            advanced_combination=probe is not None and kind == "judge",
        ))
    return PromptModel(
        key=key,
        marker=str(entry.get("marker") or ""),
        tags=tuple(str(t) for t in (entry.get("tags") or ())),
        # A prompt is only the model's own when there are no slices: with slices it is DERIVED,
        # and carrying it would recreate the divergence error the editor exists to avoid.
        prompt="" if steps else str(entry.get("prompt") or ""),
        steps=steps,
        extra={k: v for k, v in entry.items() if k not in _TASK_KEYS},
    )


def to_entry(model: PromptModel) -> dict[str, Any]:
    """The editor's model → a `tasks.yaml` entry.

    Never emits `prompt:` alongside `subtasks:`. That pairing is what triggers
    `_spec_from_entry`'s word-index divergence error, and the fix is not to keep the two in sync
    but to make the combination unreachable.
    """
    out: dict[str, Any] = {}
    if model.marker.strip():
        out["marker"] = model.marker.strip()
    if model.tags:
        out["tags"] = [str(t) for t in model.tags]
    out.update(model.extra)

    if model.steps:
        out["subtasks"] = [_step_to_entry(s) for s in model.steps]
    else:
        out["prompt"] = model.prompt
    return out


def _step_to_entry(step: StepModel) -> dict[str, Any]:
    raw: dict[str, Any] = {"prompt": step.prompt}

    # `kind` and `probe` are independent; step_type picks which of them is written.
    if step.step_type == "check" or step.advanced_combination:
        raw["kind"] = "judge"
    elif step.declared_kind == "action":
        # Preserved, not dropped: node_kind treats an absent kind as "action" so this changes no
        # behaviour, but it is a declaration the author made on purpose.
        raw["kind"] = "action"
    if step.step_type == "conditional" or step.advanced_combination:
        raw["probe"] = CheckModel(
            kind=step.probe_kind, text=step.probe_text, timeout_s=step.probe_timeout_s,
        ).to_mapping(default_timeout=PROBE_DEFAULT_TIMEOUT_S)

    if step.verify:
        raw["verify"] = [c.to_mapping(default_timeout=VERIFY_DEFAULT_TIMEOUT_S)
                         for c in step.verify]
    if step.marker.strip():
        raw["marker"] = step.marker.strip()
    if step.tab_url.strip():
        raw["tab_url"] = step.tab_url.strip()
    if step.allow_write_refusal:
        raw["allow_write_refusal"] = True
    if step.values:
        raw["values"] = dict(step.values)
    raw.update(step.extra)
    return raw


def derived_prompt(model: PromptModel) -> str:
    """What `_spec_from_entry` will compute as the whole-task prompt — the string whose sha256 is
    the task id. Duplicating the join here would risk drifting from it, so the single source of
    truth stays in tasks.py and this only mirrors the no-slices case."""
    if not model.steps:
        return " ".join(model.prompt.split())
    from automation.tasks import _instantiated
    from automation.tasks import SubtaskDecl
    return " ".join(
        " ".join(_instantiated(SubtaskDecl(prompt=s.prompt, values=s.values or None)).split())
        for s in model.steps)
