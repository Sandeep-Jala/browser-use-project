"""Skill layer: the executable unit the hybrid engine runs for one subtask.

Two body tiers behind one loader/executor interface:

  * "code"  — tier 1: a generated `async def run(api, *, params...)` function
              (skills/codegen.py) plus its ANCHOR BUNDLE (<sid>.anchors.json). Preferred
              when present, linted at every load (codegen.lint_code), executed against a
              SkillApi bound to the live page. DOM drift heals into the anchors
              (promote_healed_anchors); the code text is never edited by machinery.
  * "steps" — tier 0: the compiled selector steps (script_compile.run_steps). Present
              ONLY for entries the transpiler couldn't express (codegen deletes the steps
              file once a code skill compiles) — the fallback when the code tier is
              missing or unlintable.

`load_skill` owns instantiation for both tiers through ONE alignment rule: a
parameterized entry's template is aligned against the subtask's concrete prompt
(adapt.match_template — deterministic regex), falling back to the decomposition's own
values by NAME only when they cover EVERY template param. None means "author instead" —
never replay wrong values.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from automation.pipeline import adapt
from automation.pipeline import subtask_store as sstore
from automation.pipeline.script_compile import (_FP_ATTRS, _atomic_write, _esc, _role_of,
                                                _selectors_from_parts,
                                                merge_promoted_selectors, run_steps)
from automation.skills import codegen
from automation.skills.api import SkillApi

logger = logging.getLogger("framework.skills")

# Ceiling for an anchor's candidate list after heal promotions (same policy as the steps
# tier's promote_healed): promoted candidates prepend, the oldest fallbacks drop.
_MAX_SELECTORS = 8


@dataclass
class Skill:
    """One executable skill resolved from the library for a concrete subtask."""

    sid: str
    body: str = "steps"                            # "steps" (tier 0) | "code" (tier 1)
    steps: list[dict[str, Any]] = field(default_factory=list)   # body == "steps"
    code: str = ""                                 # body == "code"
    anchors: dict[str, Any] = field(default_factory=dict)       # body == "code"
    params: dict[str, str] = field(default_factory=dict)        # run(**params)

    def __len__(self) -> int:
        if self.body == "code":
            return self.code.count("await api.")
        return len(self.steps)


# ------------------------------- value alignment (shared) -------------------------------


def _aligned_values(sid: str, sub: Any, run_resolver: Any = None
                    ) -> tuple[dict[str, str], dict[str, Any] | None] | None:
    """(values, template) for this entry aligned to `sub`; ({}, None) when the entry is
    not parameterized; None when the wording doesn't align and the decomposition's values
    don't cover every param — author instead, never replay wrong values.

    A template may carry `bindings`: params whose values come from THIS RUN's data
    (extract labels / create-response paths) via `run_resolver(spec)`, never from the
    prompt. Every binding must resolve or the whole load refuses — the params' stored
    defaults are the AUTHORING run's literals and replaying them would be the exact
    stale-data bug the provenance guard exists to stop."""
    entry = sstore.load_manifest().get(sid) or {}
    params = entry.get("params") or {}
    if not params or not sstore.template_path(sid).exists():
        return {}, None
    try:
        template = adapt.load_template(sstore.template_path(sid))
    except Exception as exc:  # noqa: BLE001
        logger.warning("library template %s unreadable: %s", sid, exc)
        return None
    bindings = template.get("bindings") or {}
    bound_values: dict[str, str] = {}
    for name, spec in bindings.items():
        value = run_resolver(spec) if run_resolver is not None else None
        if not value:
            logger.info("library entry %s: binding %s (%s) did not resolve from this "
                        "run's data; authoring instead", sid, name,
                        (spec or {}).get("kind"))
            return None
        bound_values[name] = str(value)
    prompt_params = {k: v for k, v in (template.get("params") or {}).items()
                     if k not in bindings}
    if not prompt_params:
        return bound_values, template
    match = adapt.match_template(sub.instantiated_prompt, [
        {"id": sid, "prompt": template.get("source_prompt", ""),
         "params": prompt_params},
    ])
    if match is not None:
        return {**match.values, **bound_values}, template
    tmpl_params = set(prompt_params.keys())
    by_name = {k: v for k, v in (sub.values or {}).items() if k in tmpl_params}
    if set(by_name.keys()) != tmpl_params:
        logger.info("library entry %s: prompt did not align and decomposition values do "
                    "not cover template params %s; authoring instead",
                    sid, sorted(tmpl_params - set(by_name)))
        return None
    return {**by_name, **bound_values}, template


# ------------------------------- tier 0: steps -------------------------------


def _instantiated_steps(sid: str, sub: Any,
                        run_resolver: Any = None) -> list[dict[str, Any]] | None:
    aligned = _aligned_values(sid, sub, run_resolver)
    if aligned is None:
        return None
    values, template = aligned
    if template is None:
        try:
            return json.loads(sstore.steps_path(sid).read_text())
        except Exception as exc:  # noqa: BLE001 - unreadable entry -> author
            logger.warning("library entry %s unreadable: %s", sid, exc)
            return None
    return adapt.instantiate(template, values)


# ------------------------------- tier 1: code -------------------------------


def _substituted_anchors(anchors: dict[str, Any],
                         values: dict[str, str]) -> dict[str, Any] | None:
    """Anchor selectors with {{param}} tokens replaced by (selector-escaped) values.
    None when any token stays unresolved — an incomplete anchor must not run.

    An anchor whose selectors carried exactly ONE param is stamped `expect_text` with the
    substituted value: it identifies an element NAMED by that value (a data row), and a
    value-swapped replay must never let its fallback selectors (positional xpath, the
    recording's stale href) land on a different-named row. Applied by api.click only —
    see SkillApi._step_for."""
    out: dict[str, Any] = {}
    for handle, anchor in anchors.items():
        raw_sels = list(anchor.get("selectors") or [])
        sels = [
            codegen._TOKEN.sub(
                lambda m: _esc(values[m.group(1)]) if m.group(1) in values else m.group(0),
                s)
            for s in raw_sels
        ]
        if any(codegen._TOKEN.search(s) for s in sels):
            logger.warning("unresolved anchor tokens for handle %r; refusing code tier",
                           handle)
            return None
        out[handle] = {**anchor, "selectors": sels}
        if isinstance(anchor.get("expect_text"), str):
            # A compile-time landed-name guard whose literal the parameterizer tokenized:
            # substitute it like a value (raw, not selector-escaped — it is compared
            # against rendered text, not used as a selector).
            out[handle]["expect_text"] = codegen._TOKEN.sub(
                lambda m: values.get(m.group(1), m.group(0)), anchor["expect_text"])
        token_names = {m.group(1) for s in raw_sels
                       for m in codegen._TOKEN.finditer(s)}
        if len(token_names) == 1:
            (name,) = token_names
            if name in values:
                out[handle]["expect_text"] = values[name]
    return out


def _load_code_skill(sid: str, sub: Any, run_resolver: Any = None) -> Skill | None:
    cpath, apath = sstore.code_path(sid), sstore.anchors_path(sid)
    if not cpath.exists() or not apath.exists():
        return None
    try:
        code = cpath.read_text()
        anchors = json.loads(apath.read_text())
    except Exception as exc:  # noqa: BLE001 - unreadable code tier -> steps tier
        logger.warning("code skill %s unreadable: %s", sid, exc)
        return None
    problems = codegen.lint_code(code)
    if problems:
        logger.warning("code skill %s failed lint (%s); falling back to steps",
                       sid, "; ".join(problems))
        return None
    aligned = _aligned_values(sid, sub, run_resolver)
    if aligned is None:
        return None
    values, template = aligned
    merged = dict((template or {}).get("params") or {})
    merged.update({k: v for k, v in values.items() if k in merged})
    concrete = _substituted_anchors(anchors, merged)
    if concrete is None:
        return None
    return Skill(sid=sid, body="code", code=code, anchors=concrete, params=merged)


async def _execute_code(skill: Skill, page: Page, timeout_ms: int,
                        api: Any = None) -> dict[str, Any]:
    """Exec the generated function against a SkillApi. The lint gate runs again here so a
    hand-edited file can never execute outside the whitelist; the namespace carries no
    builtins, so the code can literally only call what `api` offers."""
    problems = codegen.lint_code(skill.code)
    if problems:
        return {"executed": 0, "failed_at": 0, "log": [], "extracted": {},
                "error": f"code skill failed lint: {'; '.join(problems)}"}
    namespace: dict[str, Any] = {"__builtins__": {}}
    try:
        exec(compile(skill.code, f"<skill {skill.sid}>", "exec"), namespace)  # noqa: S102
        fn = namespace["run"]
    except Exception as exc:  # noqa: BLE001 - a broken artifact is a failed replay
        return {"executed": 0, "failed_at": 0, "log": [], "extracted": {},
                "error": f"code skill did not load: {type(exc).__name__}: {exc}"}
    if api is None:
        api = SkillApi(page, skill.anchors, timeout_ms=timeout_ms)
    try:
        await fn(api, **skill.params)
    except Exception as exc:  # noqa: BLE001 - report exactly which call broke
        logger.exception("code skill %s broke at call %d: %s", skill.sid, api.executed, exc)
        # Partial extractions are honest diagnostic data even on a failed run. getattr:
        # `api` is an injection seam and test doubles may be leaner than SkillApi.
        return {"executed": api.executed, "failed_at": api.executed,
                "error": f"{type(exc).__name__}: {exc}", "log": api.log,
                "extracted": dict(getattr(api, "extracted", None) or {})}
    return {"executed": api.executed, "failed_at": None, "error": None, "log": api.log,
            "extracted": dict(getattr(api, "extracted", None) or {})}


# ------------------------------- public interface -------------------------------


def load_skill(sid: str, sub: Any, run_resolver: Any = None) -> Skill | None:
    """Resolve library entry `sid` into an executable Skill for the concrete subtask
    `sub` — code tier preferred, steps tier as fallback — or None when neither can be
    trusted to carry the right values (author instead). `run_resolver` supplies values
    for the entry's runtime BINDINGS (hybrid._binding_resolver); a bound entry without
    a resolver, or with a binding this run's data cannot satisfy, refuses to load."""
    skill = _load_code_skill(sid, sub, run_resolver)
    if skill is not None:
        return skill
    steps = _instantiated_steps(sid, sub, run_resolver)
    if steps is None:
        return None
    return Skill(sid=sid, body="steps", steps=steps)


async def execute(skill: Skill, page: Page, timeout_ms: int = 15000,
                  api: Any = None) -> dict[str, Any]:
    """Run a skill's body on the live page. Returns the run_steps-shaped outcome:
    {executed, failed_at, error, log}."""
    if skill.body == "steps":
        return await run_steps(page, skill.steps, timeout_ms=timeout_ms)
    if skill.body == "code":
        return await _execute_code(skill, page, timeout_ms, api=api)
    return {"executed": 0, "failed_at": 0, "log": [],
            "error": f"unknown skill body {skill.body!r} for {skill.sid}"}


def promote_healed_anchors(anchors_path: str | Path,
                           replay_log: list[dict[str, Any]]) -> list[str]:
    """Persist a PASSED code-replay's healings into the anchor bundle (atomic rewrite).

    Handle-keyed — the code tier's structural advantage over the steps tier's index-keyed
    promotion: ledger positions never have to map back to file positions. Same policy as
    promote_healed: winner selectors prepend (old anchors stay as fallbacks, capped),
    fingerprint refreshes from the winner. Caller contract: only after the replay PASSED
    its gate."""
    path = Path(anchors_path)
    if not path.exists():
        return []
    anchors: dict[str, Any] = json.loads(path.read_text())
    promoted: list[str] = []
    for entry in replay_log or []:
        winner, handle = entry.get("healed"), entry.get("handle")
        if not winner or not handle or handle not in anchors:
            continue
        anchor = anchors[handle]
        tag = (winner.get("tag") or "").lower()
        attrs = dict(winner.get("attrs") or {})
        if winner.get("role"):
            attrs.setdefault("role", winner["role"])  # _role_of reads the explicit role
        text = (winner.get("text") or "").strip()

        new_sels = _selectors_from_parts(tag, attrs, text)
        old_sels = list(anchor.get("selectors") or [])
        anchor["selectors"] = merge_promoted_selectors(new_sels, old_sels,
                                                       _MAX_SELECTORS)

        fp = anchor.get("fingerprint") or {}
        fp["tag"] = tag or fp.get("tag")
        fp["role"] = winner.get("role") or _role_of(attrs, tag) or fp.get("role")
        if text and len(text) <= 60:  # a real label is short (see script_compile)
            fp["text"] = text
        fp["attrs"] = {**(fp.get("attrs") or {}),
                       **{k: attrs[k] for k in _FP_ATTRS if attrs.get(k)}}
        if winner.get("bounds"):
            fp["bounds"] = winner["bounds"]
        anchor["fingerprint"] = {k: v for k, v in fp.items() if v}
        promoted.append(handle)
        logger.info("⬆ promoted healed selectors into anchor %r", handle)

    if promoted:
        _atomic_write(path, json.dumps(anchors, indent=2))
    return promoted
