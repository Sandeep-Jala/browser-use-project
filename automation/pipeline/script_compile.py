"""Compile a recorded agent run into a lean, deterministic selector script.

browser-use's `rerun_history` replays at the ORIGINAL pace (it re-waits the agent's recorded
delays, including its LLM thinking time). This instead extracts just the essential actions and
a STABLE selector for each interacted element from a saved recording, so the flow can be
re-run fast over Playwright with auto-wait — no LLM, no recorded LLM-thinking delays. The
agent's deliberate `wait` steps ARE kept (capped): they are load-bearing on this slow React
app, and a short settle is added after each interaction so replay doesn't outrun the UI.

Selector strategy (durability > brevity): prefer a stable attribute the app is unlikely to
re-generate (id that doesn't look auto-numbered, then aria-label / name / title / placeholder),
and fall back to the recorded positional `x_path` only as a last resort.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger("framework.script")

# ids with a 3+ digit run look auto-generated (e.g. "SearchBox129") — don't anchor on them.
_DYNAMIC_ID = re.compile(r"\d{3,}")
# Framework-generated id families whose FULL id regenerates per render (a mount-order counter
# changes every time), so they are never a durable anchor even without a 3-digit run:
#   react-select-6-input / react-select-9-option-0   (react-select — the marquee offender here)
#   :r3: / :ra:                                       (React useId / Radix / MUI)
#   mui-42 / headlessui-menu-3                        (MUI / Headless UI)
_FRAMEWORK_ID = re.compile(
    r"^(react-select-\d+|:r[0-9a-z]+:|mui-\d+|headlessui-[\w-]*\d+|radix-[\w:-]+)",
    re.IGNORECASE,
)
# A react-select descendant id carries a STABLE suffix ("option-0", "listbox") we can anchor on
# independently of the volatile instance counter. GROUPED menus nest the index
# ("option-0-0" = first option of the first group — observed on the VAT select), hence the
# (?:-\d+)* tail.
_REACT_SELECT_PART = re.compile(r"^react-select-\d+-(?P<part>option-\d+(?:-\d+)*|listbox|placeholder)$")
_RS_OPTION = re.compile(r"^(?P<instance>react-select-\d+)-option-\d+(?:-\d+)*$")
_RS_INPUT = re.compile(r"^(?P<instance>react-select-\d+)-input$")
# Playwright ARIA roles we can target with get_by_role. Recorded elements carry either an explicit
# `role` attribute or a tag we can map to an implicit role.
_TAG_ROLE = {"a": "link", "button": "button"}


def _is_dynamic_id(idv: str) -> bool:
    """True if an id is auto-generated and unsafe to anchor on across re-renders."""
    return bool(_DYNAMIC_ID.search(idv) or _FRAMEWORK_ID.match(idv))


def _esc(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _attr_sel(name: str, value: str) -> str:
    return f'css=[{name}="{_esc(value)}"]'


def _role_of(attrs: dict[str, Any], tag: str) -> str | None:
    """Explicit ARIA role, else the implicit role of a tag we know how to target."""
    role = (attrs.get("role") or "").strip().lower()
    # Only roles that pair well with an accessible name via get_by_role.
    if role in {"link", "button", "menuitem", "tab", "checkbox", "radio", "option"}:
        return role
    return _TAG_ROLE.get(tag)


def _selectors(element: dict[str, Any]) -> list[str]:
    """Ranked list of Playwright selectors for a recorded element (most → least durable).

    Replay tries these in order and uses the first that resolves UNIQUELY, so a fragile primary
    anchor (a framework id, a moved node) degrades to a stabler fallback instead of stopping the
    run or clicking the wrong element. Order:
      1. get_by_role(role, name)  — semantic + unique, the most durable web locator
      2. a non-auto-generated id
      3. react-select option suffix ([id$="-option-0"]) — instance-counter-independent
      4. a distinguishing attribute (data-testid / name / aria-label / title / placeholder)
      5. href (links)
      6. exact accessible-name text
      7. the positional xpath (last resort)
    """
    attrs = element.get("attributes") or {}
    tag = (element.get("node_name") or "").lower()
    ax_name = (element.get("ax_name") or "").strip()
    cands = _selectors_from_parts(tag, attrs, ax_name)
    # 7. Positional xpath — last resort.
    xpath = element.get("x_path")
    if xpath:
        cands.append("xpath=/" + xpath.lstrip("/"))
    return cands


def _selectors_from_parts(tag: str, attrs: dict[str, Any], ax_name: str) -> list[str]:
    """Ranked candidates (ranks 1-6) from raw element parts. Shared by compile-time
    `_selectors` and replay-time heal promotion, so a healed winner is ranked through the
    exact same durability policy as a freshly recorded element."""
    # A real label is short. A long ax_name is a screen-reader announcement (react-select emits
    # "option Bike, selected. Select is focused, type to refine list, ..." onto its cell), which
    # changes every render and must never anchor a selector.
    ax_name = (ax_name or "").strip()
    if len(ax_name) > 60:
        ax_name = ""
    cands: list[str] = []

    # 1. Role + accessible name — Playwright's most durable, unambiguous locator.
    role = _role_of(attrs, tag)
    if role and ax_name:
        cands.append(f'role={role}[name="{_esc(ax_name)}"]')

    idv = attrs.get("id")
    if idv:
        # 2. A genuinely stable id.
        if not _is_dynamic_id(idv):
            cands.append(_attr_sel("id", idv))
        # 3. react-select option/listbox: volatile counter, stable suffix; one menu open at a
        #    time so an ends-with match is unambiguous.
        m = _REACT_SELECT_PART.match(idv)
        if m:
            cands.append(f'css=[id$="-{m.group("part")}"]')

    # 4. Distinguishing attributes.
    for key in ("data-testid", "name", "aria-label", "title", "placeholder"):
        if attrs.get(key):
            cands.append(_attr_sel(key, attrs[key]))
    # 5. href for links.
    if attrs.get("href"):
        cands.append(f'css={tag or "*"}[href="{_esc(attrs["href"])}"]')
    # 6. Exact accessible-name text.
    if ax_name:
        cands.append(f'text="{_esc(ax_name)}"')

    seen: set[str] = set()
    return [c for c in cands if not (c in seen or seen.add(c))]


# Tags that can actually receive typed text. A `fill` recorded against anything else (td, div,
# li...) is a MISCAPTURE: the agent picked a container's index, so the recorded element identity
# is wrong even though browser-use made the typing "work" at runtime (focus fell wherever it
# fell — observed: a description typed into a td landed in the Item react-select's filter).
_EDITABLE_TAGS = {"input", "textarea", "select"}

# Element lines in browser-use's state_message DOM listing, e.g.
#   \t\t\t[11419]<td />
#   \t\t\t\t|SHADOW(open)|[11367]<input type=text id=TextField1444 name=productItems.0.description />
_SM_LINE = re.compile(r"\[(?P<idx>\d+)\]<(?P<tag>[a-zA-Z][\w-]*)\b(?P<attrs>[^>]*?)/?>")
_SM_ATTR = re.compile(r"([a-zA-Z][\w-]*)=([^\s>]+)")
# A react-select filter input is never the intended target of a plain-text fill.
_RS_FILTER_ID = re.compile(r"^react-select-\d+-input$")
# How many listing lines below the miscaptured container to search for the real field.
_RECOVER_WINDOW = 8


def _recover_fill_target(state_message: str, recorded_index: Any) -> dict[str, str] | None:
    """Recover the REAL editable target of a miscaptured fill from browser-use's own recording.

    The recorded interacted_element only says which *index* the agent typed at; when that index
    was a container (td/div), the true field's identity is lost from the element — but NOT from
    the step's state_message, browser-use's serialized DOM listing, which names every editable
    element with its stable attributes. Scan a few lines below the container for the nearest
    <input>/<textarea> that is a data field (not a react-select combobox filter) and return its
    attributes. There is no pre-commit replay: the compiled selector is committed once the
    authoring run passes its segment gate, then validated by the entry's first real replay —
    a wrong recovery fails that replay and the entry self-evicts (subtask_store.archive_if_failing).
    """
    if not state_message or recorded_index is None:
        return None
    lines = state_message.splitlines()
    anchor = next((i for i, ln in enumerate(lines) if f"[{recorded_index}]<" in ln), None)
    if anchor is None:
        return None
    for ln in lines[anchor + 1: anchor + 1 + _RECOVER_WINDOW]:
        m = _SM_LINE.search(ln)
        if not m or m.group("tag").lower() not in ("input", "textarea"):
            continue
        attrs = dict(_SM_ATTR.findall(m.group("attrs")))
        if _RS_FILTER_ID.match(attrs.get("id", "")) or attrs.get("role") == "combobox":
            continue  # a dropdown filter, not a data field
        if attrs.get("type") in ("hidden", "checkbox", "radio", "button", "submit"):
            continue
        if any(attrs.get(k) for k in ("name", "placeholder", "aria-label")) or (
            attrs.get("id") and not _is_dynamic_id(attrs["id"])
        ):
            attrs["__tag__"] = m.group("tag").lower()
            return attrs
    return None


def _recovered_selectors(attrs: dict[str, str]) -> list[str]:
    """Ranked selectors for a recovered fill target (name first — the app's stablest anchor)."""
    cands: list[str] = []
    for key in ("name", "placeholder", "aria-label"):
        if attrs.get(key):
            cands.append(_attr_sel(key, attrs[key]))
    if attrs.get("id") and not _is_dynamic_id(attrs["id"]):
        cands.append(_attr_sel("id", attrs["id"]))
    return cands


# Attributes kept in a fingerprint — the durable, identifying ones (a self-healing scorer
# weighs them at replay). `class`/`value` are deliberately excluded: they churn on this React
# app and would drag the score toward the wrong element.
_FP_ATTRS = ("id", "name", "aria-label", "placeholder", "title", "data-testid", "type", "href")


def _fingerprint(element: dict[str, Any]) -> dict[str, Any]:
    """Distill a recorded element into a self-healing fingerprint.

    Everything here is already in the recorded element dict (see DOMInteractedElement.to_dict);
    compile just stops discarding it. Used ONLY as a replay fallback: when every ranked selector
    fails, `_heal_locate` scores the live DOM against this fingerprint and acts on the best,
    unambiguous match instead of aborting the whole script.
    """
    attrs = element.get("attributes") or {}
    tag = (element.get("node_name") or "").lower()
    ax_name = (element.get("ax_name") or "").strip()
    if len(ax_name) > 60:  # screen-reader announcement blob, not a label (see _selectors)
        ax_name = ""
    fp: dict[str, Any] = {
        "tag": tag or None,
        "role": _role_of(attrs, tag) or ((attrs.get("role") or "").strip().lower() or None),
        "text": ax_name or None,
        "attrs": {k: attrs[k] for k in _FP_ATTRS if attrs.get(k)},
        "xpath": element.get("x_path") or None,
        "bounds": element.get("bounds") or None,
    }
    return {k: v for k, v in fp.items() if v}


def _attach_fp(step: dict[str, Any], element: dict[str, Any]) -> dict[str, Any]:
    """Attach a self-healing fingerprint to a step (no-op if the element yields nothing useful)."""
    fp = _fingerprint(element)
    if fp:
        step["fingerprint"] = fp
    return step


def _push_step(steps: list[dict[str, Any]], step: dict[str, Any]) -> None:
    """Append a step, collapsing the agent's slow-app retries against the last real step.

    The app re-renders slowly, so during authoring the agent often clicks the same control
    several times before it registers (e.g. "+ Invoice" clicked twice). Fast replay doesn't
    need those retries — the first click already takes effect and NAVIGATES, so a second click
    on the same target then finds nothing and eats the full locator timeout. We therefore drop
    a click that repeats the last interaction's selector, and for a repeated fill on the same
    field we keep only the latest value (the last write wins). We compare against the last
    *non-wait* step so a retry separated by a recorded wait is still collapsed.
    """
    if step.get("action") in ("click", "fill"):
        prev_idx = next(
            (j for j in range(len(steps) - 1, -1, -1) if steps[j].get("action") != "wait"),
            None,
        )
        prev = steps[prev_idx] if prev_idx is not None else None
        if (
            prev is not None
            and prev.get("action") == step["action"]
            and prev.get("selectors") == step.get("selectors")
        ):
            if step["action"] == "fill":
                steps[prev_idx] = step  # same field typed again → keep the final value
            # repeated click → drop it (the first already fired)
            return
    steps.append(step)


# The app's react-select "+ Create \"<name>\"" option. Its label embeds the (per-replay
# changing) name, so replay clicks it via :has-text("Create") — the only option containing
# that word in an open menu — with position as last resort.
_CREATE_LINE = re.compile(r'^\W*Create\s*"\s*(?P<inline>[^"]*)')


def _create_option_name(state_message: str, backend_id: Any) -> str | None:
    """The record name if the clicked option is a '+ Create "<name>"' option, else None.

    The recorded option element carries no ax_name, but browser-use's DOM listing
    (state_message) renders the option's child text on the lines directly below its
    [backend_id] line — either inline (`Create "Bobby"`) or split across lines
    (`Create "` / `Bobby` / `"`). Anchoring on the element's own lines is what makes this
    precise: the word "Create" also appears in the task text embedded elsewhere in the
    state_message.
    """
    if not state_message or backend_id is None:
        return None
    lines = state_message.splitlines()
    anchor = next((i for i, ln in enumerate(lines) if f"[{backend_id}]" in ln), None)
    if anchor is None:
        return None
    seg = [ln.strip() for ln in lines[anchor + 1: anchor + 6]]
    for i, ln in enumerate(seg):
        m = _CREATE_LINE.match(ln)
        if not m:
            continue
        name = (m.group("inline") or "").strip()
        if not name and i + 1 < len(seg):  # name on the following listing line
            name = seg[i + 1].strip().strip('"').strip()
        return name or None
    return None


def _dropdown_option_steps(
    element: dict[str, Any], steps: list[dict[str, Any]], state_message: str = "",
) -> list[dict[str, Any]] | None:
    """For a click on a react-select OPTION, synthesize a click BY LABEL instead of the
    recorded positional option click. Returns the replacement step(s), or None to fall
    through to generic `_selectors` handling.

    A positional `option-3` click is fragile AND unparameterizable — replayed against
    different data it clicks "whatever now sits at position 3" (observed: the item 'bike'
    click was recorded with no readable label, compiled to `[id$="-option-1"]`, and failed
    replay validation). The label comes from either:
      * the option's own ax_name. If nothing was typed to filter first (the agent picked a
        visible option by sight), ALSO synthesize typing that label into the open menu's
        focused input, so the value is replayable by VALUE and parameterize() can bind it; or
      * the filter text just typed into the SAME react-select instance — covers options whose
        ax_name the recorder failed to capture. The recorded positional id stays as the LAST
        fallback in case the typed filter was only a partial label.
    """
    attrs = element.get("attributes") or {}
    match = _RS_OPTION.match(attrs.get("id") or "")
    if not match:
        return None
    # '+ Create "<name>"' option: creates a NEW record. Its label embeds the name — which
    # changes when the template tier swaps in a new value — and its position depends on how
    # many existing records the menu lists (observed at option-17). Click it by the constant
    # word "Create" instead; the recorded position is only a fallback.
    created = _create_option_name(state_message, element.get("backend_node_id"))
    if created:
        part = _REACT_SELECT_PART.match(attrs.get("id") or "")
        return [{"action": "click", "selectors": [
            'css=[id*="-option"]:has-text("Create")',
            f'css=[id$="-{part.group("part")}"]' if part else 'css=[id$="-option-0"]',
        ]}]
    ax_name = (element.get("ax_name") or "").strip()
    prev = next((s for s in reversed(steps) if s.get("action") != "wait"), None)
    typed = ""
    if prev is not None and prev.get("action") in ("fill", "type") and \
            prev.get("field_id") == match.group("instance"):
        typed = str(prev.get("value") or prev.get("text") or "").strip()
    label = ax_name or typed
    if not label:
        return None  # no label anywhere — generic (positional) handling is all we have
    part = _REACT_SELECT_PART.match(attrs.get("id") or "")
    positional = f'css=[id$="-{part.group("part")}"]' if part else 'css=[id$="-option-0"]'
    click_step = {"action": "click", "selectors": [
        f'role=option[name="{_esc(label)}"]',
        f'text="{_esc(label)}"',
        positional if typed else 'css=[id$="-option-0"]',
    ]}
    if typed:
        # The filter is already a recorded fill step; just click the option it filtered to.
        return [click_step]
    type_step = {
        "action": "type", "text": label,
        # Field identity for parameterize(): the select instance this menu belongs to.
        "field_id": match.group("instance"),
    }
    return [type_step, click_step]


def compile_recording(
    recording_path: str | Path, max_steps: int | None = None, *,
    emit_start_goto: bool = True,
) -> list[dict[str, Any]]:
    """Turn a saved agent history JSON into an ordered list of {action, ...} steps.

    `max_steps` keeps only the first N agent steps. Used to cut a recording at the step where
    the create-write fired (see ground_truth["write_step"]), so an agent that flailed AFTER
    the record was actually saved never gets its post-save junk into the script.

    `emit_start_goto=False` skips the leading goto to the recording's start URL. Mid-flow
    subtask segments need this: on an SPA a reload destroys live form state, and the segment's
    context-keyed lookup already guarantees the page is in its start state when it replays.
    """
    data = json.loads(Path(recording_path).read_text())
    steps: list[dict[str, Any]] = []
    history = data.get("history", [])
    if max_steps is not None and max_steps > 0:
        history = history[:max_steps]
    # Assert a known starting page: the agent's flow began on this URL, so replay must too.
    # Without it, replay silently depends on wherever the browser happened to be left.
    if emit_start_goto:
        start_url = ((history[0].get("state") or {}).get("url") if history else None)
        if start_url and start_url.startswith("http"):
            steps.append({"action": "goto", "url": start_url})
    for item in history:
        actions = (item.get("model_output") or {}).get("action") or []
        elements = (item.get("state") or {}).get("interacted_element") or []
        # ActionResults for this step, aligned to actions (one action per step in this app).
        # find_by_text stashes the element it clicked here (agent_tools.py) since a custom
        # action gets no state.interacted_element.
        results = item.get("result") or []
        for i, action in enumerate(actions):
            if not action:
                continue
            name = next(iter(action))
            params = action[name] or {}
            element = elements[i] if i < len(elements) else None
            if name == "find_by_text" and params.get("click_first"):
                # A navigation/click made via find_by_text: recover its target element from the
                # recorded metadata and compile it exactly like a built-in click. Without this,
                # every find_by_text click (menus, Sales, btnInvoice, Save, ...) is dropped and
                # the replay skeleton collapses.
                element = None
                if i < len(results) and isinstance(results[i], dict):
                    md = results[i].get("metadata")
                    if isinstance(md, dict):
                        element = md.get("interacted_element")
                query = str(params.get("text") or "").strip()
                if element and not element.get("hidden_click"):
                    synth = _dropdown_option_steps(element, steps,
                                                   item.get("state_message") or "")
                    if synth is not None:
                        for s in synth:
                            _push_step(steps, s)
                    else:
                        sels = _selectors(element)
                        if sels:
                            # hidden_ok: find_by_text can reach controls a re-render hides;
                            # replay keeps the hover/dispatch recovery as a safety net.
                            _push_step(steps, _attach_fp(
                                {"action": "click", "selectors": sels,
                                 "hidden_ok": True}, element))
                elif query:
                    # The click went through the tool's hidden-control path (or the saved
                    # history lacks the element entirely): no selector+pointer translation
                    # is stable for such controls, so replay the INTENT — a find_click step
                    # runs the exact same in-page algorithm the tool used (RAW_FIND_JS).
                    _push_step(steps, {"action": "find_click", "text": query})
            elif name == "navigate" and params.get("url"):
                _push_step(steps, {"action": "goto", "url": params["url"]})
            elif name == "click" and element:
                synth = _dropdown_option_steps(element, steps,
                                               item.get("state_message") or "")
                if synth is not None:
                    for s in synth:
                        _push_step(steps, s)
                    continue
                sels = _selectors(element)
                if sels:
                    _push_step(steps, _attach_fp(
                        {"action": "click", "selectors": sels}, element))
            elif name == "input" and element:
                # Miscapture repair: a fill recorded against a non-editable container means the
                # element identity is wrong (agent typed at a td/div index). Recover the real
                # field from this step's state_message instead of compiling the container.
                if (element.get("node_name") or "").lower() not in _EDITABLE_TAGS and \
                        "contenteditable" not in (element.get("attributes") or {}):
                    rec = _recover_fill_target(item.get("state_message") or "",
                                               params.get("index"))
                    sels = _recovered_selectors(rec) if rec else []
                    if sels:
                        logger.warning(
                            "fill %r was recorded against <%s> (not editable); recovered real "
                            "target %s from state_message", str(params.get("text", ""))[:40],
                            element.get("node_name"), sels[0])
                        _push_step(steps, {
                            "action": "fill", "selectors": sels,
                            "value": str(params.get("text", "")),
                            "clear": params.get("clear", True), "recovered": True,
                            "fingerprint": {"tag": rec.pop("__tag__", "input"),
                                            "attrs": {k: v for k, v in rec.items()
                                                      if k in _FP_ATTRS}},
                        })
                        continue
                    logger.warning(
                        "fill %r recorded against non-editable <%s> and no recovery target "
                        "found; compiling container selectors (replay may descend)",
                        str(params.get("text", ""))[:40], element.get("node_name"))
                sels = _selectors(element)
                if sels:
                    step = _attach_fp({"action": "fill", "selectors": sels,
                            "value": str(params.get("text", "")), "clear": params.get("clear", True)},
                            element)
                    # Typing into a react-select filter input: stamp the select instance so a
                    # following option click can be resolved BY the typed label (and so
                    # parameterize() groups retried fills of this field together).
                    m = _RS_INPUT.match((element.get("attributes") or {}).get("id") or "")
                    if m:
                        step["field_id"] = m.group("instance")
                    _push_step(steps, step)
            elif name == "send_keys" and params.get("keys"):
                _push_step(steps, {"action": "press", "keys": params["keys"]})
            elif name in ("capped_scroll", "scroll"):
                # Discovery scrolling is load-bearing: the target section must be scrolled
                # into view before the following click can resolve (observed: the Reviews
                # panel's "View all" icon). capped_scroll is our tool ({pages}); the
                # built-in scroll uses {num_pages}.
                pages = params.get("pages", params.get("num_pages", 0.5))
                try:
                    pages = float(pages)
                except (TypeError, ValueError):
                    pages = 0.5
                _push_step(steps, {"action": "scroll",
                                   "down": bool(params.get("down", True)),
                                   "pages": min(pages, 1.0)})
            elif name == "wait":
                # Keep the agent's deliberate pauses (capped). They are load-bearing on this
                # slow React app: they let the invoice form and its react-select menus finish
                # rendering before the next click. Dropping them makes fast replay outrun the UI
                # (menu not open yet → click times out; Save fires before state commits → no POST).
                secs = params.get("seconds")
                if isinstance(secs, (int, float)) and secs > 0:
                    _push_step(steps, {"action": "wait", "seconds": min(float(secs), 3.0)})
            # `done` is intentionally dropped — Playwright auto-waits on locators.
    return steps


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a half-written file."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_steps(
    recording_path: str | Path, steps_path: str | Path, max_steps: int | None = None, *,
    emit_start_goto: bool = True,
) -> list[dict[str, Any]]:
    """Compile `recording_path` and write the step list to `steps_path` atomically."""
    steps = compile_recording(recording_path, max_steps=max_steps,
                              emit_start_goto=emit_start_goto)
    _atomic_write(Path(steps_path), json.dumps(steps, indent=2))
    return steps


# The raw-DOM find+click algorithm, SHARED between find_by_text (authoring, agent_tools)
# and the `find_click` replay step: token match over title/aria-label/name/text plus child
# icon hints, visible-first ranking, scrollIntoView (which scrolls the CORRECT container —
# unlike window.scrollBy, a no-op inside Fluent ScrollablePanes), then the element's own
# click handler. Using the identical implementation at author and replay time is what makes
# hover-revealed/0-size controls (the Reviews "View all" icon) replayable at all.
# Placeholders: %s = JSON token list, %s = "true"/"false" for click.
RAW_FIND_JS = r"""
(function () {
  try {
    var TOKENS = %s, DOCLICK = %s;
    var sel = 'button,a,[role=button],[role=menuitem],[role=tab],[role=link],' +
              'input[type=button],input[type=submit],[data-is-focusable],[onclick]';
    var out = [];
    document.querySelectorAll(sel).forEach(function (e) {
      var hay = [e.getAttribute('title'), e.getAttribute('aria-label'), e.getAttribute('name'),
                 e.innerText].filter(Boolean).join(' ');
      e.querySelectorAll('[data-icon-name],[title],[aria-label]').forEach(function (c) {
        hay += ' ' + [c.getAttribute('data-icon-name'), c.getAttribute('title'),
                      c.getAttribute('aria-label')].filter(Boolean).join(' ');
      });
      hay = hay.toLowerCase();
      if (TOKENS.every(function (t) { return hay.indexOf(t) !== -1; })) {
        var r = e.getBoundingClientRect();
        var icon = e.querySelector('[data-icon-name]');
        var name = e.getAttribute('title') || e.getAttribute('aria-label') ||
                   e.getAttribute('name') || (e.innerText || '').trim() ||
                   (icon && icon.getAttribute('data-icon-name')) || '';
        out.push({ el: e, name: name.trim().slice(0, 80),
                   visible: r.width > 0 && r.height > 0 });
      }
    });
    if (!out.length) return { count: 0 };
    out.sort(function (a, b) { return (b.visible ? 1 : 0) - (a.visible ? 1 : 0); });
    var top = out[0], clicked = false;
    if (DOCLICK) { try { top.el.scrollIntoView({ block: 'center' }); top.el.click(); clicked = true; } catch (e) {} }
    var attrs = {};
    ['id', 'aria-label', 'title', 'name', 'placeholder', 'data-testid', 'href', 'role']
      .forEach(function (a) { var v = top.el.getAttribute(a); if (v) attrs[a] = v; });
    return { count: out.length, clicked: clicked, name: top.name,
             names: out.slice(0, 8).map(function (o) { return o.name; }),
             element: { tag: top.el.tagName.toLowerCase(), attrs: attrs } };
  } catch (e) { return { error: String(e) }; }
})()
"""

# After an interaction, give the slow React app a beat to open a menu / commit react-select
# state / re-render before the next locator query, so replay doesn't outrun the UI.
_SETTLE_MS = 400
# Budget for probing a non-final candidate selector: it should fail fast so a stale anchor falls
# through to the durable fallback instead of eating the whole timeout.
_PROBE_MS = 2500


def _step_selectors(step: dict[str, Any]) -> list[str]:
    """Candidate selectors for a step (supports the legacy single-`selector` form too)."""
    sels = step.get("selectors")
    if sels:
        return list(sels)
    one = step.get("selector")
    return [one] if one else []


_EDITABLE_SEL = "input, textarea, select, [contenteditable='true'], [contenteditable='']"

# --- Self-healing fallback -------------------------------------------------------------------
# When every ranked selector for a step fails, score the live DOM against the step's recorded
# fingerprint and act on the single best, UNAMBIGUOUS match (mirrors _resolve's "refuse an
# ambiguous locator" rule). This rescues elements whose durable anchors all regenerated (the
# xpath-only react-select fields that otherwise abort a whole replay).
_HEAL_ATTR = "data-heal-target"   # temporary marker the scorer stamps on its winner
# A heal commits only when best >= THRESHOLD and best beats the runner-up by >= MARGIN, so a
# lone strong signal (exact id/name/aria-label, weight 4) or a coherent bundle of weaker ones
# wins, but two lookalike fields (small margin) are refused — replay-validation then fails safe.
_HEAL_THRESHOLD = 4.0
_HEAL_MARGIN = 1.5

# In-page scorer. Receives {fp, editable, attr, threshold, margin}; clears any prior marker,
# scores visible (and, if editable, editable) candidates by attribute/role/tag/text overlap plus
# a small position tiebreak, and stamps the winner with `attr` when it clears the gates.
_HEAL_JS = r"""
(a) => {
  var fp = a.fp || {}, attrs = fp.attrs || {};
  var wantTag = fp.tag || null, wantText = (fp.text || '').trim().toLowerCase();
  var ATTR = a.attr;
  var prev = document.querySelectorAll('[' + ATTR + ']');
  for (var p = 0; p < prev.length; p++) prev[p].removeAttribute(ATTR);
  function norm(s){ return (s == null ? '' : '' + s).trim().toLowerCase(); }
  function toks(s){ return norm(s).split(/[^a-z0-9]+/).filter(function(w){ return w.length > 1; }); }
  function visible(el){
    var s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
    var r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }
  function editableEl(el){
    var t = el.tagName.toLowerCase();
    if (t === 'input') return el.type !== 'hidden';
    if (t === 'textarea' || t === 'select') return true;
    return !!el.isContentEditable;
  }
  var nodes;
  if (a.editable) nodes = document.querySelectorAll("input, textarea, select, [contenteditable='true'], [contenteditable='']");
  else if (wantTag) nodes = document.getElementsByTagName(wantTag);
  else nodes = document.querySelectorAll('*');
  var wantToks = toks(wantText);
  var cx = null, cy = null;
  if (fp.bounds) { cx = fp.bounds.x + fp.bounds.width / 2; cy = fp.bounds.y + fp.bounds.height / 2; }
  var scored = [];
  for (var i = 0; i < nodes.length; i++) {
    var el = nodes[i];
    if (!el || !el.getAttribute) continue;
    if (a.editable && !editableEl(el)) continue;
    if (!visible(el)) continue;
    var sc = 0;
    if (attrs['data-testid'] && el.getAttribute('data-testid') === attrs['data-testid']) sc += 5;
    if (attrs['id'] && el.id === attrs['id']) sc += 4;
    if (attrs['name'] && el.getAttribute('name') === attrs['name']) sc += 4;
    if (attrs['aria-label'] && el.getAttribute('aria-label') === attrs['aria-label']) sc += 3;
    if (attrs['placeholder'] && el.getAttribute('placeholder') === attrs['placeholder']) sc += 3;
    if (attrs['href'] && el.getAttribute('href') === attrs['href']) sc += 2;
    if (attrs['title'] && el.getAttribute('title') === attrs['title']) sc += 1.5;
    if (attrs['type'] && el.getAttribute('type') === attrs['type']) sc += 0.5;
    if (fp.role) { var r = norm(el.getAttribute('role')); if (r && r === fp.role) sc += 2; }
    if (wantTag && el.tagName.toLowerCase() === wantTag) sc += 1;
    if (wantText) {
      var et = norm(el.innerText || el.textContent || el.getAttribute('aria-label') || el.value || el.getAttribute('placeholder'));
      if (et === wantText) sc += 3;
      else if (wantToks.length) {
        var ets = toks(et), ov = 0;
        for (var k = 0; k < wantToks.length; k++) if (ets.indexOf(wantToks[k]) >= 0) ov++;
        sc += 1.5 * (ov / wantToks.length);
      }
    }
    if (cx !== null) {
      var rr = el.getBoundingClientRect();
      var d = Math.sqrt(Math.pow(rr.left + rr.width / 2 - cx, 2) + Math.pow(rr.top + rr.height / 2 - cy, 2));
      sc += Math.max(0, 2 - d / 300);
    }
    if (sc > 0) scored.push({ el: el, sc: sc });
  }
  if (!scored.length) return { healed: false };
  scored.sort(function(x, y){ return y.sc - x.sc; });
  var best = scored[0], margin = best.sc - (scored.length > 1 ? scored[1].sc : 0);
  if (best.sc < a.threshold || margin < a.margin) return { healed: false, score: best.sc, margin: margin };
  best.el.setAttribute(ATTR, '1');
  // Winner identity, so a successful heal can be promoted into the golden script's selector
  // list (same attribute family as the fingerprint).
  var FP = ['id', 'name', 'aria-label', 'placeholder', 'title', 'data-testid', 'type', 'href'];
  var wAttrs = {};
  for (var f = 0; f < FP.length; f++) {
    var av = best.el.getAttribute(FP[f]);
    if (av) wAttrs[FP[f]] = av;
  }
  var br = best.el.getBoundingClientRect();
  return { healed: true, score: best.sc, margin: margin,
           desc: best.el.tagName.toLowerCase() + (best.el.id ? '#' + best.el.id : ''),
           winner: {
             tag: best.el.tagName.toLowerCase(),
             role: norm(best.el.getAttribute('role')) || null,
             text: ('' + (best.el.innerText || best.el.textContent || '')).trim().slice(0, 80),
             attrs: wAttrs,
             bounds: { x: br.left, y: br.top, width: br.width, height: br.height }
           } };
}
"""


async def _heal_locate(page: Page, fingerprint: dict[str, Any], editable: bool):
    """Score the live DOM against `fingerprint` and return (locator, label, winner) for a
    confident, unambiguous match, else None. The winner element is stamped with _HEAL_ATTR so
    we can locate it without a durable selector; scoring clears any prior stamp so only one is
    ever tagged. `winner` is the element's identity ({tag, role, text, attrs, bounds}) so a
    successful heal can later be promoted into the golden script (see promote_healed)."""
    try:
        info = await page.evaluate(_HEAL_JS, {
            "fp": fingerprint, "editable": bool(editable), "attr": _HEAL_ATTR,
            "threshold": _HEAL_THRESHOLD, "margin": _HEAL_MARGIN,
        })
    except Exception as exc:  # noqa: BLE001 - a heal attempt must never crash the replay
        logger.debug("heal scorer failed: %s", exc)
        return None
    if not info or not info.get("healed"):
        return None
    loc = page.locator(f"[{_HEAL_ATTR}]")
    try:
        if await loc.count() != 1:
            return None
    except Exception:  # noqa: BLE001
        return None
    logger.info("↺ healed via fingerprint (score=%.1f margin=%.1f -> %s)",
                info.get("score", 0.0), info.get("margin", 0.0), info.get("desc", "?"))
    return loc.first, f"healed:{info.get('desc', '?')}", info.get("winner") or None


async def _resolve(page: Page, step: dict[str, Any], timeout_ms: int,
                   require_editable: bool = False):
    """Return (locator, selector_label, healed_winner) for the first candidate that resolves
    to EXACTLY ONE visible element. `healed_winner` is None unless the self-healing fallback
    located the element (then it carries the winner's identity for promotion).

    Refusing to act on an ambiguous match is what prevents "clicks somewhere else": rather
    than silently taking `.first`, we require a unique hit — judged among VISIBLE matches
    only. A hidden twin earlier in DOM order (Fluent keeps collapsed panels' header buttons
    in the DOM — observed live: [title="View all"] resolving to an invisible header button
    while the real icon sat at nth(1)) must neither shadow the real target nor make it
    "ambiguous". Non-final candidates get a short probe budget; the last candidate gets the
    full timeout and, only then, a logged first-visible concession. With `require_editable`
    (fill steps), a candidate that resolves to a non-editable node is skipped so replay
    falls through to a candidate that hits the real input.
    """
    sels = _step_selectors(step)
    if not sels:
        raise RuntimeError("step has no selector")
    errors: list[str] = []
    for i, sel in enumerate(sels):
        last = i == len(sels) - 1
        budget = timeout_ms if last else _PROBE_MS
        loc = page.locator(sel)
        try:
            await loc.first.wait_for(state="visible", timeout=budget)
        except Exception:  # noqa: BLE001 - .first may be a hidden twin; scan the rest below
            pass
        count = await loc.count()
        if count == 0:
            errors.append(f"{sel} -> no match")
            continue
        visible: list[int] = []
        for n in range(min(count, 8)):
            try:
                if await loc.nth(n).is_visible():
                    visible.append(n)
            except Exception:  # noqa: BLE001 - node vanished mid-scan
                continue
        if not visible:
            errors.append(f"{sel} -> {count} match(es), none visible")
            continue
        if len(visible) == 1:
            candidate = loc.nth(visible[0])
        elif last:
            # Exhausted durable candidates; act on the first VISIBLE match but record it.
            logger.warning("ambiguous selector %r: %d visible matches; using the first",
                           sel, len(visible))
            candidate = loc.nth(visible[0])
        else:
            errors.append(f"{sel} -> {len(visible)} visible matches (ambiguous)")
            continue
        if require_editable:
            try:
                if not await candidate.is_editable():
                    inner = candidate.locator(_EDITABLE_SEL)
                    if await inner.count() == 0:
                        errors.append(f"{sel} -> visible but not an editable field")
                        continue
                    # The recorded element was the field's container; type into its input.
                    return inner.first, f"{sel} >> {_EDITABLE_SEL}", None
            except Exception:  # noqa: BLE001 - element vanished mid-check; try the next
                errors.append(f"{sel} -> vanished during editability check")
                continue
        return candidate, sel, None
    # Every ranked selector failed. Before giving up, try the self-healing fallback: score the
    # live DOM against the step's recorded fingerprint and act on a confident, unique match.
    fingerprint = step.get("fingerprint")
    if fingerprint:
        healed = await _heal_locate(page, fingerprint, require_editable)
        if healed is not None:
            return healed
    raise RuntimeError("no unique candidate matched: " + " | ".join(errors))


# How many times to re-resolve + re-act a step that fails transiently before giving up.
_MAX_ATTEMPTS = 3
# Substrings of errors caused by the app RE-RENDERING a row/menu mid-interaction: the node we
# located (or was about to act on) detaches before we can use it. This is the dominant replay
# failure on this React app — selecting a line-item Item auto-fills Account/VAT and rebuilds the
# whole <tr>, so a field located a beat too early vanishes. Re-resolving after a short settle,
# once React has finished swapping the subtree, almost always succeeds. We retry ONLY these:
# a genuine logic error (wrong value, missing field) still fails, so no bad script is committed.
_TRANSIENT = (
    "vanished during editability check",
    "no unique candidate matched",
    "not attached",
    "element is not attached",
    "element is not stable",
    "detached",
    "element was detached",
    # Resolved, then hidden by a re-render before the click landed: re-resolving picks the
    # currently-visible instance (see _resolve's visible-match scan).
    "is not visible",
)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in _TRANSIENT)


# How many progressive scroll rounds _reveal_hidden_click hunts before giving up: a Fluent
# virtualized ScrollablePane renders a section's DOM only when the viewport nears it.
_REVEAL_ROUNDS = 5

# Center of the target's nearest VISIBLE ancestor (its section/card), scrolled into view —
# the thing a user hovers to make a hover-revealed icon appear.
_VISIBLE_ANCESTOR_JS = """
(el) => {
  let n = el.parentElement;
  while (n) {
    const r = n.getBoundingClientRect();
    if (r.width > 4 && r.height > 4) {
      n.scrollIntoView({block: 'center'});
      const r2 = n.getBoundingClientRect();
      return {x: r2.left + r2.width / 2, y: r2.top + Math.min(r2.height / 2, 40)};
    }
    n = n.parentElement;
  }
  return null;
}
"""


async def _reveal_hidden_click(page: Page, step: dict[str, Any]) -> str | None:
    """Recovery for hidden_ok steps (recorded through find_by_text's hidden-control path)
    using the mechanics a USER would use on a hover-revealed control — observed live: the
    Reviews "View all" icon exists 0-size until its section is hovered.

      1. If no candidate selector matches yet, scroll down progressively: a virtualized
         pane renders the section's DOM only near the viewport.
      2. Scroll the target's nearest VISIBLE ancestor (its section) into view and HOVER it
         with real mouse movement — the reveal trigger.
      3. Real-click the now-visible target (a trusted, user-equivalent click).
      4. Only if hovering never reveals it: dispatch the element's own click handler.

    Unique-match rule throughout: hidden recovery never guesses among elements. Returns
    the used selector label, or None."""
    target, used_sel = None, None
    for _ in range(_REVEAL_ROUNDS):
        for sel in _step_selectors(step):
            loc = page.locator(sel)
            try:
                if await loc.count() == 1:
                    target, used_sel = loc.first, sel
                    break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
        if target is not None:
            break
        try:
            await page.evaluate("() => window.scrollBy(0, 0.6 * window.innerHeight)")
        except Exception:  # noqa: BLE001 - page gone; nothing to recover
            return None
        await page.wait_for_timeout(400)
    if target is None:
        return None
    try:
        point = await target.evaluate(_VISIBLE_ANCESTOR_JS)
    except Exception:  # noqa: BLE001 - detached target
        point = None
    if point:
        try:
            await page.mouse.move(point["x"], point["y"], steps=6)
            await page.wait_for_timeout(_SETTLE_MS)
            if await target.is_visible():
                await target.click(timeout=5000)
                logger.info("↺ hover-revealed and clicked via %r", used_sel)
                return f"{used_sel} (hover reveal)"
            # The icon may appear adjacent to the hovered point; nudge onto the target's
            # own position (it has a box once the section is hovered on some skins).
            box = await target.bounding_box()
            if box and box["width"] > 0:
                await page.mouse.move(box["x"] + box["width"] / 2,
                                      box["y"] + box["height"] / 2, steps=4)
                await page.wait_for_timeout(_SETTLE_MS)
                if await target.is_visible():
                    await target.click(timeout=5000)
                    logger.info("↺ hover-revealed and clicked via %r", used_sel)
                    return f"{used_sel} (hover reveal)"
        except Exception as exc:  # noqa: BLE001 - fall through to dispatch
            logger.debug("hover-reveal attempt failed via %r: %s", used_sel, exc)
    try:
        await target.dispatch_event("click")
        logger.info("↺ hidden-click dispatched via %r", used_sel)
        return f"{used_sel} (hidden dispatch)"
    except Exception:  # noqa: BLE001 - recovery failed; caller raises the original error
        return None


async def _click_with_retry(page: Page, step: dict[str, Any], timeout_ms: int) -> tuple[str, dict[str, Any] | None]:
    """Resolve + click, re-resolving after a settle if the target detaches mid-render.
    Returns (selector_label, healed_winner_or_None)."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            loc, sel, healed = await _resolve(page, step, timeout_ms)
            try:
                await loc.click(timeout=5000)
            except Exception:  # noqa: BLE001 - typically "another element intercepts pointer events"
                # react-select renders a placeholder div UNDER an input container that intercepts
                # pointer events; a forced click dispatches at the element's position — i.e. onto
                # the overlaying control, which is the real target.
                logger.warning("click on %r intercepted/failed; retrying with force", sel)
                await loc.click(timeout=timeout_ms, force=True)
            return sel, healed
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_ATTEMPTS - 1 and _is_transient(exc):
                logger.info("click step transient (%s); settling %dms then re-resolving (attempt %d/%d)",
                            exc, _SETTLE_MS * 2, attempt + 2, _MAX_ATTEMPTS)
                await page.wait_for_timeout(_SETTLE_MS * 2)
                continue
            if step.get("hidden_ok"):
                used = await _reveal_hidden_click(page, step)
                if used is not None:
                    return used, None
            raise
    raise RuntimeError("unreachable")  # loop either returns or raises


async def _fill_with_retry(page: Page, step: dict[str, Any], timeout_ms: int) -> tuple[str, dict[str, Any] | None]:
    """Resolve + fill, re-resolving after a settle if the field detaches mid-render.
    Returns (selector_label, healed_winner_or_None)."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            loc, sel, healed = await _resolve(page, step, timeout_ms, require_editable=True)
            if step.get("clear", True):
                await loc.fill("", timeout=timeout_ms)
            await loc.fill(step.get("value", ""), timeout=timeout_ms)
            return sel, healed
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_ATTEMPTS - 1 and _is_transient(exc):
                logger.info("fill step transient (%s); settling %dms then re-resolving (attempt %d/%d)",
                            exc, _SETTLE_MS * 2, attempt + 2, _MAX_ATTEMPTS)
                await page.wait_for_timeout(_SETTLE_MS * 2)
                continue
            raise
    raise RuntimeError("unreachable")  # loop either returns or raises


async def _wheel_scroll(page: Page, pages: float, down: bool = True) -> None:
    """User-faithful scroll: real wheel input at the viewport center. window.scrollBy is a
    NO-OP inside Fluent ScrollablePanes (the app scrolls an inner container, not the
    window) — wheel events land on whatever is under the cursor, exactly like a user."""
    size = page.viewport_size or {"width": 1280, "height": 800}
    await page.mouse.move(size["width"] / 2, size["height"] / 2)
    await page.mouse.wheel(0, (1 if down else -1) * float(pages) * size["height"])
    await page.wait_for_timeout(_SETTLE_MS)


# Wheel-scroll rounds find_click hunts before giving up: a virtualized pane renders a
# section's DOM only once the viewport nears it.
_FIND_CLICK_ROUNDS = 4


async def _find_click(page: Page, text: str) -> str:
    """Replay a find_by_text click SEMANTICALLY: run the same in-page algorithm the tool
    used at record time (RAW_FIND_JS — token match, visible-first, scrollIntoView, direct
    handler click), wheel-scrolling between rounds when nothing matches yet. Returns the
    clicked element's reported name; raises when no round finds a match."""
    import json as _json

    tokens = [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]
    if not tokens:
        raise RuntimeError(f"find_click: no searchable text in {text!r}")
    expr = RAW_FIND_JS % (_json.dumps(tokens), "true")
    for round_no in range(_FIND_CLICK_ROUNDS + 1):
        try:
            raw = await page.evaluate(expr)
        except Exception as exc:  # noqa: BLE001 - page navigating; settle and retry
            logger.debug("find_click eval failed (%s); settling", exc)
            raw = None
            await page.wait_for_timeout(_SETTLE_MS)
        if raw and raw.get("clicked"):
            name = str(raw.get("name") or text)
            logger.info("🔎 find_click(%r): clicked %r (round %d)", text, name, round_no)
            return name
        if round_no < _FIND_CLICK_ROUNDS:
            await _wheel_scroll(page, 0.6)
    raise RuntimeError(f"find_click: no clickable match for {text!r} "
                       f"after {_FIND_CLICK_ROUNDS} scroll rounds")


# Budget for each click of the flyout-reopen recovery (predecessor + retried target). Shorter
# than the main timeout: the recovery either works quickly or the failure was real.
_REOPEN_MS = 8000


async def _click_with_flyout_recovery(
    page: Page, steps: list[dict[str, Any]], idx: int, timeout_ms: int
) -> tuple[str, dict[str, Any] | None]:
    """Click step `idx`, and if its target is unreachable, re-click the nearest PREVIOUS click
    step once, then retry the target.

    This is the replay-engine version of the FLYOUT SUBMENUS recovery the agent prompt documents:
    submenu items (e.g. Sales under Inputs) exist only while their parent flyout is open, and any
    app re-render closes it. On a slow render the flyout can close between our predecessor click
    and this step's probe; re-probing the target alone (what _click_with_retry does) can never
    bring it back — only re-clicking its opener can. If the recovery also fails, the ORIGINAL
    error is raised so the report shows the real failure.
    """
    step = steps[idx]
    try:
        return await _click_with_retry(page, step, timeout_ms)
    except Exception as exc:  # noqa: BLE001
        prev = next((steps[j] for j in range(idx - 1, -1, -1)
                     if steps[j].get("action") == "click"), None)
        if prev is None:
            raise
        logger.info("click step %d unreachable (%s); re-clicking predecessor to reopen its "
                    "flyout, then retrying the target once", idx, str(exc)[:120])
        try:
            await _click_with_retry(page, prev, _REOPEN_MS)
            await page.wait_for_timeout(_SETTLE_MS)
            sel, healed = await _click_with_retry(page, step, _REOPEN_MS)
        except Exception:  # noqa: BLE001 - recovery failed; surface the original failure
            raise exc
        logger.info("↺ flyout recovery succeeded for step %d (%s)", idx, sel)
        return sel, healed


async def run_steps(page: Page, steps: list[dict[str, Any]], timeout_ms: int = 15000) -> dict[str, Any]:
    """Execute compiled steps over a Playwright page. Returns {executed, failed_at, error, log}.

    `log` records which concrete selector resolved for each interaction, so a healer/validator can
    see exactly how each step was located (and which candidate won).

    NOTE on create-record scripts: values are replayed EXACTLY as compiled. A "create X" flow
    only replays while X does not exist — run it with a different name via the template tier,
    or delete the record in the app first.
    """
    executed = 0
    log: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        try:
            action = step["action"]
            if action == "goto":
                await page.goto(step["url"], wait_until="domcontentloaded", timeout=timeout_ms)
            elif action == "click":
                sel, healed = await _click_with_flyout_recovery(page, steps, idx, timeout_ms)
                entry = {"step": idx, "action": action, "used": sel}
                if healed:
                    entry["healed"] = healed
                log.append(entry)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "fill":
                sel, healed = await _fill_with_retry(page, step, timeout_ms)
                entry = {"step": idx, "action": action, "used": sel}
                if healed:
                    entry["healed"] = healed
                log.append(entry)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "press":
                await page.keyboard.press(step["keys"])
            elif action == "type":
                # Types into the FOCUSED element — used for an open react-select menu, whose
                # filter input owns focus. If focus is elsewhere the subsequent click-by-label
                # times out and validation fails safely (no bad script is ever committed).
                await page.keyboard.type(step["text"], delay=30)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "scroll":
                await _wheel_scroll(page, float(step.get("pages", 0.5)),
                                    down=bool(step.get("down", True)))
            elif action == "find_click":
                name = await _find_click(page, step.get("text", ""))
                log.append({"step": idx, "action": action, "used": f"find_click:{name}"})
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "wait":
                await page.wait_for_timeout(int(step.get("seconds", 0) * 1000))
            executed += 1
        except Exception as exc:  # noqa: BLE001 - report where the script broke (app changed?)
            logger.exception("script execution broke at step %d: %s", idx, exc)
            return {"executed": executed, "failed_at": idx,
                    "error": f"{type(exc).__name__}: {exc}", "log": log}
    return {"executed": executed, "failed_at": None, "error": None, "log": log}


# Ceiling for a step's candidate list after heal promotions, so repeated healings of a churny
# element can't grow it without bound (promoted candidates prepend; the oldest fallbacks drop).
_MAX_SELECTORS = 8


def promote_healed(steps_path: str | Path, replay_log: list[dict[str, Any]]) -> list[int]:
    """Persist successful replay healings into the golden script (atomic rewrite).

    For each replay-log entry that carries a healed winner, synthesize durable selectors from
    the winner's identity via the same ranking policy as compile time and PREPEND them to that
    step's candidate list (the old anchors stay as fallbacks), then refresh the step's
    fingerprint from the winner. A winner with no durable anchor only refreshes the
    fingerprint. Returns the indices of the updated steps ([] leaves the file untouched).

    Caller contract: only invoke after the replay PASSED its ground-truth gate — a failed run
    must never rewrite a golden script.
    """
    steps_path = Path(steps_path)
    steps: list[dict[str, Any]] = json.loads(steps_path.read_text())
    promoted: list[int] = []
    for entry in replay_log or []:
        winner = entry.get("healed")
        idx = entry.get("step")
        if not winner or idx is None or not 0 <= idx < len(steps):
            continue
        step = steps[idx]
        tag = (winner.get("tag") or "").lower()
        attrs = dict(winner.get("attrs") or {})
        if winner.get("role"):
            attrs.setdefault("role", winner["role"])  # _role_of reads the explicit role here
        text = (winner.get("text") or "").strip()

        new_sels = _selectors_from_parts(tag, attrs, text)
        old_sels = _step_selectors(step)  # normalizes the legacy single-`selector` form
        seen: set[str] = set()
        merged = [s for s in new_sels + old_sels
                  if not (s in seen or seen.add(s))][:_MAX_SELECTORS]
        if merged != old_sels:
            step["selectors"] = merged
            step.pop("selector", None)

        fp = step.get("fingerprint") or {}
        fp["tag"] = tag or fp.get("tag")
        fp["role"] = winner.get("role") or _role_of(attrs, tag) or fp.get("role")
        if text and len(text) <= 60:  # same "a real label is short" rule as _selectors
            fp["text"] = text
        fp["attrs"] = {**(fp.get("attrs") or {}),
                       **{k: attrs[k] for k in _FP_ATTRS if attrs.get(k)}}
        if winner.get("bounds"):
            fp["bounds"] = winner["bounds"]
        step["fingerprint"] = {k: v for k, v in fp.items() if v}
        promoted.append(idx)
        logger.info("⬆ promoted healed selectors into step %d: %s", idx, new_sels or "(fingerprint only)")

    if promoted:
        _atomic_write(steps_path, json.dumps(steps, indent=2))
    return promoted
