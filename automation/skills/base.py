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

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from automation.pipeline import adapt
from automation.pipeline import subtask_store as sstore
from automation.pipeline.script_compile import (_FP_ATTRS, _PUA, _atomic_write, _esc,
                                                _is_dynamic_id, _role_of,
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
            # Same rule as the tier-0 stamp (adapt.instantiate): a token that only scopes
            # a `:has-text(...)` ROW names the row, not the anonymous control inside it.
            if name in values and adapt._token_names_the_target(raw_sels, name):
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
        optional_from = getattr(api, "optional_from", None)
        if optional_from is not None and api.executed >= optional_from:
            # The failure is inside the slice's DECLARED error branch (api.begin_optional
            # marks where it starts). On a run that does not raise that branch there is
            # nothing for its closing steps to act on — which is the success case, not a
            # broken replay. The tail is always trailing, so the work all ran. Tier-0 twin:
            # script_compile.run_steps' `optional` handling.
            logger.info("code skill %s: the declared error branch did not apply this run "
                        "(%s); %d work call(s) completed", skill.sid, exc, api.executed)
            return {"executed": api.executed, "failed_at": None, "error": None,
                    "log": api.log,
                    "extracted": dict(getattr(api, "extracted", None) or {})}
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


# ------------------------------- the takeover brief -------------------------------
#
# When a replay breaks part-way the agent takes over IN PLACE, on the dirty page the
# replay abandoned (hybrid.run_hybrid_task). It used to be told only "a previous attempt
# partially completed this step and then stopped" — a boolean — so a recording that died
# on its LAST action and one that died on its second produced the identical prompt, and
# the agent had to reconstruct the boundary by inspection.
#
# The run's own ledger already knows: every executed verb is recorded (SkillApi._record /
# run_steps' log), and the skill body says what the whole sequence was. `replay_progress`
# turns those two into a truthful brief.
#
# TRUTHFULNESS: a ledger entry proves the action was DISPATCHED and its target resolved.
# It does NOT prove the intended effect happened — a click can land on a re-rendered
# control, a dialog can refuse a save. The wording (prompts.scoped_subtask_prompt) says
# exactly that, and the step that BROKE is reported as attempted-outcome-unknown, never
# as done or as not-done: a fill raises on its read-back after having typed.

_BRIEF_DONE_CAP = 8        # the tail is what places the agent; older actions elide
_BRIEF_REMAINING_CAP = 10
_BRIEF_VALUE_CAP = 40

# api verb -> the action name its ledger entry carries. Verbs missing here record
# nothing (begin_optional) and are skipped on BOTH sides of the match.
_VERB_ACTIONS = {
    "click": "click", "repeat_click": "click", "repeat_until_done": "click",
    "click_indexed": "click", "fill": "fill", "select": "select",
    "select_option": "select_option", "paste": "paste", "upload": "upload",
    "extract": "extract", "copy": "copy", "find_click": "find_click",
    "type_text": "type", "press": "press", "wait": "wait", "scroll": "scroll",
    "goto": "goto", "close_tab": "close_tab",
}
# Verbs that log one entry PER CLICK (api.repeat_click / repeat_until_done /
# click_indexed), so the ledger is longer than the call list and index arithmetic lies.
_REPEAT_VERBS = {"repeat_click", "repeat_until_done", "click_indexed"}
# Which positional argument carries the verb's value (the typed/picked text, the extract
# label, the url). Handle-bearing verbs put the handle at 0.
_VALUE_ARG = {"fill": 1, "select": 1, "paste": 1, "upload": 1, "extract": 1, "copy": 1,
              "select_option": 0, "find_click": 0, "type_text": 0, "press": 0,
              "goto": 0, "wait": 0}
_HANDLE_VERBS = {"click", "repeat_click", "repeat_until_done", "click_indexed", "fill",
                 "select", "paste", "upload", "extract", "copy"}


def _brief_text(value: Any) -> str:
    """Trim a name for the brief. Fluent icon fonts draw glyphs as literal
    Private-Use-Area TEXT NODES, so a raw name can be an invisible codepoint —
    strip them (script_compile._PUA) or the brief names a control called nothing."""
    return _PUA.sub("", str(value or "")).strip()


def _brief_name(node: dict[str, Any]) -> str:
    """The control's human name from an anchor or a compiled step: the recorded
    expect_text first (already param-substituted at load), then the fingerprint's text,
    then its labelling attributes. Empty when the control was genuinely nameless."""
    expect = _brief_text(node.get("expect_text"))
    if expect and "{{" not in expect:
        return expect
    fp = node.get("fingerprint") or {}
    text = _brief_text(fp.get("text"))
    if text:
        return text
    attrs = fp.get("attrs") or {}
    for key in ("aria-label", "name", "placeholder", "title", "id"):
        value = _brief_text(attrs.get(key))
        if not value:
            continue
        if key == "id" and _is_dynamic_id(value):
            # An auto-generated id ("react-select-11-input", "TextField99") is not a name
            # and does not survive a re-render — printing it would send the agent hunting
            # for a string that is not on this run's page. Observed live: run
            # 20260908_091814 seg 19 broke on exactly such a control.
            continue
        return value
    return ""


def _brief_role(node: dict[str, Any]) -> str:
    """What to call a control that has no name: its role/tag, so "an unnamed combobox"
    beats "an unnamed control" for an agent that has to find it on the page."""
    fp = node.get("fingerprint") or {}
    role = _brief_text(fp.get("role")) or _brief_text((fp.get("attrs") or {}).get("role"))
    if not role:
        tag = _brief_text(fp.get("tag")).lower()
        role = {"input": "input", "select": "dropdown", "textarea": "text box",
                "a": "link", "button": "button"}.get(tag, "")
    return role.lower()


def _brief_value(value: Any) -> str:
    text = _brief_text(value)
    if not text:
        return ""
    return f'"{text[:_BRIEF_VALUE_CAP]}…"' if len(text) > _BRIEF_VALUE_CAP else f'"{text}"'


def _brief_phrase(item: dict[str, Any], *, past: bool) -> str:
    """One line of the brief. `past` narrates what ran; the present tense is used for the
    recording's REMAINING actions, which are a plan, not a history."""
    action = item.get("action")
    name = item.get("name") or ""
    where = f'"{name}"' if name else f"an unnamed {item.get('role') or 'control'}"
    value = (_brief_text(item.get("value")) if item.get("value_phrase")
             else _brief_value(item.get("value")))
    label = _brief_text(item.get("label"))
    if action in ("click", "find_click"):
        repeat = item.get("count")
        times = f" {int(repeat)}×" if repeat and int(repeat) > 1 else ""
        return f"{'clicked' if past else 'click'} {where}{times}"
    if action == "fill":
        return (f"{'filled' if past else 'fill'} {where}"
                + (f" with {value}" if value else ""))
    if action == "paste":
        return f"{'pasted' if past else 'paste'} {value or 'a value'} into {where}"
    if action == "select":
        return f"{'selected' if past else 'select'} {value or 'an option'} in {where}"
    if action == "select_option":
        return f"{'picked' if past else 'pick'} the option {value or '(unnamed)'}"
    if action in ("extract", "copy"):
        what = label or "a value"
        got = f" = {value}" if past and value else ""
        return f"{'captured' if past else 'capture'} {what} from {where}{got}"
    if action == "upload":
        return f"{'uploaded' if past else 'upload'} {value or 'a file'} to {where}"
    if action == "type":
        return f"{'typed' if past else 'type'} {value or 'text'}"
    if action == "press":
        return f"{'pressed' if past else 'press'} {_brief_text(item.get('value')) or 'a key'}"
    if action == "scroll":
        return "scrolled the page" if past else "scroll the page"
    if action == "wait":
        return "waited" if past else "wait"
    if action == "goto":
        return f"{'navigated to' if past else 'navigate to'} {_brief_text(item.get('value'))}"
    if action == "close_tab":
        return "closed the tab" if past else "close the tab"
    return f"{action} {where}"


def _step_item(step: dict[str, Any]) -> dict[str, Any]:
    """A compiled tier-0 step as a brief item."""
    action = step.get("action")
    value = step.get("value")
    if value is None:
        value = step.get("text") or step.get("keys") or step.get("url")
    return {"action": action, "name": _brief_name(step), "role": _brief_role(step),
            "value": value, "label": step.get("label"), "count": step.get("count")}


def _arg_text(node: Any, params: dict[str, str]) -> str:
    """Render a call argument for the brief: a literal, a param (substituted with THIS
    run's value), or api.noted('x') — which has no value until the run reaches it, so it
    is named rather than guessed."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        return str(params.get(node.id, node.id))
    if isinstance(node, ast.Call):
        fn = node.func
        if (isinstance(fn, ast.Attribute) and fn.attr == "noted" and node.args
                and isinstance(node.args[0], ast.Constant)):
            return f"the value noted as {node.args[0].value}"
        return ""
    if isinstance(node, ast.JoinedStr):
        return "".join(_arg_text(part.value if isinstance(part, ast.FormattedValue)
                                 else part, params) for part in node.values)
    return ""


def _code_calls(code: str, anchors: dict[str, Any],
                params: dict[str, str]) -> list[dict[str, Any]]:
    """The skill's api.* calls in SOURCE order, as brief items. ast.walk is unordered, so
    the nodes are sorted by position; codegen.lint_code already guarantees every call is
    an `api.<verb>(...)`."""
    tree = ast.parse(code)
    nodes: list[Any] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == "api"):
            continue
        if fn.attr == "noted":
            continue  # an ARGUMENT of another call, not a step in the sequence
        nodes.append(node)
    nodes.sort(key=lambda n: (n.lineno, n.col_offset))
    calls: list[dict[str, Any]] = []
    for node in nodes:
        verb = node.func.attr
        handle = None
        if verb in _HANDLE_VERBS and node.args and isinstance(node.args[0], ast.Constant):
            handle = str(node.args[0].value)
        idx = _VALUE_ARG.get(verb)
        arg = node.args[idx] if idx is not None and len(node.args) > idx else None
        value = _arg_text(arg, params) if arg is not None else ""
        # A {{noted:label}} value has no text until the run reaches it, so the
        # brief names it in prose — quoting it would read as a literal to type.
        phrase = isinstance(arg, ast.Call)
        count = None
        if verb in ("repeat_click", "click_indexed") and len(node.args) > 2:
            arg = node.args[2] if verb == "click_indexed" else node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                count = arg.value
        anchor = (anchors.get(handle) or {}) if handle else {}
        # No handle fallback for the name: a handle is DERIVED from the control's
        # name (codegen._handle_for), so a nameless control gets a bare verb like
        # "click"/"copy" — printing that would invent a control called "click".
        calls.append({"verb": verb, "handle": handle, "count": count,
                      "action": _VERB_ACTIONS.get(verb),
                      "name": _brief_name(anchor), "role": _brief_role(anchor),
                      "value": value if verb not in ("extract", "copy") else "",
                      "value_phrase": phrase,
                      "label": value if verb in ("extract", "copy") else None})
    return calls


def _split_code(calls: list[dict[str, Any]], log: list[dict[str, Any]]
                ) -> tuple[list[dict], dict | None, list[dict]] | None:
    """(done, attempted, remaining) for a code skill, matched by HANDLE SEQUENCE.

    Not by index: repeat_click/repeat_until_done log one entry per click and
    begin_optional logs none, so api.executed is not a position in the call list. Fails
    CLOSED — a ledger that does not line up returns None and the caller falls back to the
    generic wording rather than naming actions that may not be the ones that ran."""
    pos = 0
    for i, call in enumerate(calls):
        action = call["action"]
        if action is None:
            continue  # begin_optional and friends record nothing
        if pos >= len(log):
            return calls[:i], call, calls[i + 1:]
        entry = log[pos]
        if entry.get("action") != action:
            return None
        if call["handle"] and entry.get("handle") and entry["handle"] != call["handle"]:
            return None
        pos += 1
        if call["verb"] in _REPEAT_VERBS:
            # One call, many entries. A literal count consumes at most that many, so a
            # plain click on the SAME control right after a repeat is not swallowed.
            want = int(call["count"]) if call.get("count") else None
            got = 1
            while (pos < len(log) and (want is None or got < want)
                   and log[pos].get("action") == action
                   and log[pos].get("handle") == entry.get("handle")):
                pos += 1
                got += 1
            if want is not None and got < want:
                # The repeat broke DURING its own clicks (2 of 3 landed). The action that
                # was attempted is this call, not the one after it — and anything left in
                # the ledger would mean the run continued past a call that raised, which
                # is not a shape we can narrate.
                return (calls[:i], call, calls[i + 1:]) if pos == len(log) else None
    if pos != len(log):
        return None  # more happened than the body accounts for: do not narrate
    return calls, None, []


def replay_progress(skill: Skill, outcome: dict[str, Any]) -> dict[str, Any] | None:
    """Where a broken replay of `skill` got to, as material for the takeover prompt.

    -> {"done": [phrase], "attempted": phrase|None, "remaining": [phrase],
        "ran_to_end": bool, "done_count": int, "total": int, "elided": int,
        "remaining_more": int, "summary": one-line}
    None when the body is empty or the ledger cannot be aligned with it. A replay that
    broke on its very FIRST action still gets a brief (nothing done, that action
    attempted): the run is not dirty, so no prompt shows it, but it is what progress.json
    keeps about why the entry stopped working.
    """
    try:
        return _replay_progress(skill, outcome)
    except Exception as exc:  # noqa: BLE001 - a brief must never break a recovery
        logger.debug("replay progress brief unavailable: %s", exc)
        return None


def _replay_progress(skill: Skill, outcome: dict[str, Any]) -> dict[str, Any] | None:
    log = list(outcome.get("log") or [])
    failed_at = outcome.get("failed_at")
    if skill.body == "steps":
        steps = list(skill.steps)
        if failed_at is None:
            split: tuple[list[dict], dict | None, list[dict]] | None = (steps, None, [])
        elif not isinstance(failed_at, int) or not 0 <= failed_at < len(steps):
            split = None
        else:
            # Tier-0 index arithmetic IS exact: run_steps stamps the step index on every
            # entry and counts `executed` once per step, repeats included.
            split = (steps[:failed_at], steps[failed_at], steps[failed_at + 1:])
        if split is None:
            return None
        raw_done, raw_attempted, raw_remaining = split
        done_items = [_step_item(s) for s in raw_done]
        attempted_item = _step_item(raw_attempted) if raw_attempted else None
        remaining_items = [_step_item(s) for s in raw_remaining]
        total = len(steps)
    elif skill.body == "code":
        calls = _code_calls(skill.code, skill.anchors, skill.params)
        split = _split_code(calls, log) if failed_at is not None else (calls, None, [])
        if split is None:
            return None
        done_items, attempted_item, remaining_items = split
        total = len([c for c in calls if c["action"] is not None])
    else:
        return None

    if not done_items and attempted_item is None and not remaining_items:
        return None  # an empty body has nothing to narrate
    # A replayed extract's value is real data the agent may need, and it is already in
    # this run's values.json — naming it here saves a re-read of a page that may be gone.
    extracted = outcome.get("extracted") or {}
    for item in done_items:
        if item.get("action") in ("extract", "copy") and item.get("label"):
            item["value"] = extracted.get(str(item["label"]), item.get("value"))

    # api.begin_optional() is a MARK, not an action (it records nothing and touches
    # no control), so it never appears in the narration.
    done = [_brief_phrase(i, past=True) for i in done_items if i.get("action")]
    elided = max(0, len(done) - _BRIEF_DONE_CAP)
    remaining = [_brief_phrase(i, past=False) for i in remaining_items
                 if i.get("action")]
    remaining_more = max(0, len(remaining) - _BRIEF_REMAINING_CAP)
    attempted = _brief_phrase(attempted_item, past=False) if attempted_item else None
    ran_to_end = failed_at is None
    if ran_to_end:
        summary = f"ran all {total} recorded action(s); the end check failed"
    elif attempted:
        summary = (f"ran {len(done)} of {total} recorded action(s); broke while "
                   f"attempting: {attempted}")
    else:
        # Every call is accounted for by the ledger, yet the replay reported a
        # failure: it broke after its last action, not on one of them.
        summary = (f"ran all {total} recorded action(s); the failure came after "
                   f"the last one")
    return {"done": done[-_BRIEF_DONE_CAP:], "attempted": attempted,
            "remaining": remaining[:_BRIEF_REMAINING_CAP], "ran_to_end": ran_to_end,
            "done_count": len(done), "total": total, "elided": elided,
            "remaining_more": remaining_more, "summary": summary}
