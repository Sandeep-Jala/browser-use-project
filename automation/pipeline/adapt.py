"""Parameterized replay: turn a golden script into a reusable template.

When a golden script is committed, `parameterize()` binds each typed value to a NAMED
parameter (one LLM call: fill step 13 types "5" → param "qty"), producing a template file:

    {"source_prompt": "...",
     "params": {"client": "290 CREW LIMITED", "customer": "Suresh Gopi", "qty": "5", ...},
     "steps": [... fill values and value-bearing selectors carry {{param}} tokens ...]}

When a new prompt later has no script of its own, `match_template()` (one LLM call) picks the
template with the same procedure and reads the new prompt's value for EVERY parameter into a
dictionary; `instantiate()` then swaps the tokens. Because each token was placed at a specific
step at parameterize time, substitution is purely mechanical and per-field — two fields that
happen to share a value (two "5"s) are separate parameters and can never cross-contaminate,
which is the weakness of diff-based old→new string replacement this replaces.

There is no pre-commit replay validation: a template is committed as soon as its authoring
run passes the segment gate, and an instantiated script is judged by that same gate the
moment it replays (hybrid.replay_segment). A wrong instantiation therefore fails its
replays and the entry self-evicts (subtask_store.archive_if_failing).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from browser_use.llm.messages import SystemMessage, UserMessage

from automation.pipeline.prompts import PARAMETERIZE_SYSTEM_PROMPT
from automation.pipeline.script_compile import _atomic_write, _esc

logger = logging.getLogger("framework.adapt")

_TOKEN = re.compile(r"\{\{([a-z0-9_]+)\}\}")
_VALID_PARAM = re.compile(r"^[a-z][a-z0-9_]*$")


def _token(name: str) -> str:
    return "{{" + name + "}}"


def _parse_json_reply(text: str) -> dict[str, Any] | None:
    """Parse the model's JSON reply, tolerating markdown fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


# --------------------------- parameterize (at golden-script commit) ---------------------------


async def parameterize(prompt: str, steps: list[dict[str, Any]], llm: Any) -> dict[str, Any] | None:
    """Build a template from a committed golden script, or None if nothing is parameterizable.

    Only fill values that appear verbatim in the prompt become parameters — a value the prompt
    never mentions (an auto-suggestion the agent picked, a default) could not be read out of a
    future prompt either, so it stays concrete. One LLM call names each candidate by role; the
    binding is per-step, so equal values in different fields become distinct parameters.
    """
    if llm is None:
        return None

    # Group fill steps by FIELD, not by step: the agent may fight a slow field and fill it
    # several times (observed: Qty filled 4x while its displayed value drifted). One param per
    # field, default = the FINAL value written, and every fill in the group gets the token —
    # otherwise instantiating a new value would be overwritten by a later duplicate fill of
    # the old one. The positional xpath (always last candidate) is the most field-identifying
    # anchor: volatile labels/values change between retries, the element's position does not.
    def _field_key(step: dict[str, Any]) -> str | None:
        if step.get("field_id"):  # synthetic dropdown type-steps carry an explicit identity
            return str(step["field_id"])
        if step.get("action") == "find_click":
            # Semantic clicks are grouped by their label: retried find_clicks of the same
            # target become one parameter.
            return f"find_click:{step.get('text', '')}"
        selectors = step.get("selectors") or []
        for s in reversed(selectors):
            if s.startswith("xpath="):
                return s
        return selectors[0] if selectors else None

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for i, step in enumerate(steps):
        if step.get("action") not in ("fill", "type", "find_click"):
            continue
        value = str(step.get("value") if step.get("action") == "fill"
                    else step.get("text") or "").strip()
        if not value:
            continue
        key = _field_key(step) or f"step-{i}"
        group = groups.setdefault(key, {"steps": [], "value": "",
                                        "field": (step.get("selectors") or [""])[0]})
        if not group["steps"]:
            order.append(key)
        group["steps"].append(i)
        group["value"] = value  # last write wins
    fills = [
        {"step": groups[key]["steps"][-1], "value": groups[key]["value"],
         "field": groups[key]["field"]}
        for key in order
        if groups[key]["value"].lower() in prompt.lower()
    ]
    group_of_step = {i: key for key in order for i in groups[key]["steps"]}

    # Untyped dropdown picks: a click on a react-select option with no filter typed before it
    # (observed: customer chosen by sight from the auto-opened list). The recorded element
    # carries NO label text, so the LLM infers which prompt value was picked there; the value
    # is then verified verbatim against the prompt before being trusted.
    _OPTION_SEL = re.compile(r'\[id\$="-option-\d+"\]')
    picks: list[dict[str, Any]] = []
    prev_action_step: dict[str, Any] | None = None
    for i, step in enumerate(steps):
        if step.get("action") == "wait":
            continue
        if step.get("action") == "click" and any(
            _OPTION_SEL.search(s) for s in step.get("selectors") or []
        ):
            typed_before = prev_action_step is not None and \
                prev_action_step.get("action") in ("fill", "type")
            if not typed_before:
                picks.append({"step": i, "value": None,
                              "field": "dropdown option picked without typing — infer its "
                                       "label from the task prompt"})
        prev_action_step = step
    pick_steps = {p["step"] for p in picks}

    if not fills and not picks:
        return None

    try:
        result = await llm.ainvoke(
            [
                SystemMessage(content=PARAMETERIZE_SYSTEM_PROMPT),
                UserMessage(content=f"TASK PROMPT:\n{prompt}\n\nINPUTS:\n"
                                    f"{json.dumps(fills + picks, indent=1)}"),
            ]
        )
        data = _parse_json_reply(result.completion or "")
    except Exception as exc:  # noqa: BLE001 - parameterization is best-effort
        logger.warning("parameterize failed (%s); no template saved", exc)
        return None
    if not data:
        return None

    # Validate bindings: known step index, sane name, and one value per name (a name reused
    # for a DIFFERENT value gets suffixed rather than silently merging two fields).
    value_by_step = {f["step"]: f["value"] for f in fills}
    params: dict[str, str] = {}
    step_param: dict[int, str] = {}
    pick_param: dict[int, str] = {}
    for binding in data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        idx, name = binding.get("step"), str(binding.get("param", "")).strip().lower()
        if not _VALID_PARAM.match(name):
            continue
        if idx in pick_steps:
            # Dropdown pick: the LLM supplies the label; trust it only if it appears
            # verbatim in the prompt (a hallucinated label would poison the template).
            value = str(binding.get("value") or "").strip()
            if not value or value.lower() not in prompt.lower():
                continue
            if name in params and params[name] != value:
                base, n = name, 2
                while name in params and params[name] != value:
                    name = f"{base}_{n}"
                    n += 1
            params[name] = value
            pick_param[idx] = name
            continue
        if idx not in value_by_step:
            continue
        value = value_by_step[idx]
        if name in params and params[name] != value:
            base, n = name, 2
            while name in params and params[name] != value:
                name = f"{base}_{n}"
                n += 1
        params[name] = value
        # Token every fill of this FIELD (not just the representative step), so replaying a
        # new value cannot be undone by a later duplicate fill of the old one.
        for i in groups[group_of_step[idx]]["steps"]:
            step_param[i] = name
    if not step_param and not pick_param:
        return None

    # Tokenize the steps: fill values by their per-step binding; selectors only for params
    # whose value is unique among params (a shared value inside a selector would be ambiguous).
    owners: dict[str, list[str]] = {}
    for name, value in params.items():
        owners.setdefault(value.strip().lower(), []).append(name)
    template_steps: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        new_step = dict(step)
        if i in pick_param:
            # Rewrite the positional option click as: type the (tokenized) label into the
            # open menu's focused filter, then click the option BY LABEL — first filtered
            # option as fallback. This is what makes an untyped pick value-substitutable.
            name = pick_param[i]
            template_steps.append({"action": "type", "text": _token(name)})
            template_steps.append({"action": "click", "selectors": [
                f'role=option[name="{_token(name)}"]',
                f'text="{_token(name)}"',
                'css=[id$="-option-0"]',
            ]})
            continue
        if i in step_param:
            if new_step.get("action") in ("type", "find_click"):
                new_step["text"] = _token(step_param[i])
            else:
                new_step["value"] = _token(step_param[i])
        selectors = new_step.get("selectors")
        if selectors:
            for name, value in params.items():
                if len(owners[value.strip().lower()]) != 1:
                    continue
                quoted_value, quoted_token = f'"{_esc(value)}"', f'"{_token(name)}"'
                selectors = [s.replace(quoted_value, quoted_token) for s in selectors]
            new_step["selectors"] = selectors
        template_steps.append(new_step)

    return {"source_prompt": prompt, "params": params, "steps": template_steps}


# --------------------------- match + instantiate (at replay time) ---------------------------


@dataclass
class TemplateMatch:
    """The template a new prompt matched, plus the new prompt's value for every parameter."""

    source_tid: str
    values: dict[str, str]


def _template_regex(template_prompt: str, params: dict[str, str]) -> re.Pattern | None:
    """Compile a template's prompt into a regex that matches any value-swapped variant.

    Literal text is escaped; each whole-value occurrence of a param default becomes a named
    capture group (later occurrences of the same param become backreferences, so a value that
    appears twice must change consistently). A prompt that matches yields every new value
    directly from the alignment — no LLM involved.
    """
    norm = " ".join(template_prompt.split())
    spans: list[tuple[int, int, str]] = []
    for name, default in (params or {}).items():
        for m in re.finditer(rf"(?<!\w){re.escape(default)}(?!\w)", norm, re.IGNORECASE):
            spans.append((m.start(), m.end(), name))
    spans.sort()
    pattern, last_end, seen = "", 0, set()
    for start, end, name in spans:
        if start < last_end:  # two params claiming the same text (shared default) — first wins
            continue
        pattern += re.escape(norm[last_end:start])
        pattern += rf"(?P={name})" if name in seen else rf"(?P<{name}>.+?)"
        seen.add(name)
        last_end = end
    if not seen:
        return None
    pattern += re.escape(norm[last_end:])
    return re.compile(f"^{pattern}$", re.IGNORECASE | re.DOTALL)


def match_template(
    new_prompt: str, candidates: list[dict[str, Any]], llm: Any = None
) -> TemplateMatch | None:
    """Match `new_prompt` against recorded templates; None means no value-only match.

    Fully deterministic (`llm` is accepted for API compatibility but unused): a candidate
    matches iff the new prompt equals its recorded prompt with only parameter values swapped,
    and the values are read straight out of the alignment. An earlier LLM-based matcher was
    unreliable in both directions — it matched prompts whose changes the template could not
    express (silently saving a wrong record), and refused clearly-covered prompts when several
    similar templates were listed.
    """
    norm_new = " ".join(new_prompt.split())
    matches: list[tuple[TemplateMatch, int]] = []
    for c in candidates:
        rx = _template_regex(c.get("prompt", ""), c.get("params") or {})
        if rx is None:
            continue
        m = rx.match(norm_new)
        if not m:
            continue
        values = {k: v.strip() for k, v in m.groupdict().items() if v is not None}
        matches.append((TemplateMatch(source_tid=c["id"], values=values),
                        len(c.get("params") or {})))
    if not matches:
        return None
    # Several templates of the same wording family can all match (e.g. two recorded variants
    # of the same task): prefer the one with the most parameters (the most expressive).
    matches.sort(key=lambda pair: -pair[1])
    if len(matches) > 1:
        logger.info("%d templates match; using the most parameterized (%s)",
                    len(matches), matches[0][0].source_tid)
    return matches[0][0]


def instantiate(template: dict[str, Any], values: dict[str, str]) -> list[dict[str, Any]] | None:
    """Fill a template's {{param}} tokens from `values` (defaults fill the gaps).

    Fill values get the raw string; selectors get the selector-escaped form. Returns None if
    any token is left unresolved (the match reply missed a parameter) — an incomplete script
    must not run.
    """
    merged = dict(template.get("params") or {})
    merged.update({k: v for k, v in (values or {}).items() if k in merged})

    def _sub(text: str, escape: bool) -> str:
        return _TOKEN.sub(
            lambda m: (_esc(merged[m.group(1)]) if escape else merged[m.group(1)])
            if m.group(1) in merged else m.group(0),
            text,
        )

    steps: list[dict[str, Any]] = []
    for step in template.get("steps") or []:
        new_step = dict(step)
        if isinstance(new_step.get("value"), str):
            new_step["value"] = _sub(new_step["value"], escape=False)
        if isinstance(new_step.get("text"), str):
            new_step["text"] = _sub(new_step["text"], escape=False)
        if new_step.get("selectors"):
            new_step["selectors"] = [_sub(s, escape=True) for s in new_step["selectors"]]
        steps.append(new_step)

    for step in steps:
        leftovers = ([str(step.get("value", "")), str(step.get("text", ""))]
                     + list(step.get("selectors") or []))
        if any(_TOKEN.search(text) for text in leftovers):
            logger.warning("unresolved template tokens in step %r; refusing to instantiate", step)
            return None
    return steps


# --------------------------- template file I/O ---------------------------


def load_template(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_template(path: str | Path, template: dict[str, Any]) -> None:
    _atomic_write(Path(path), json.dumps(template, indent=2))
