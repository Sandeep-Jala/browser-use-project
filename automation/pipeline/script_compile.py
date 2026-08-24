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
from urllib.parse import urlparse

from playwright.async_api import Locator, Page

logger = logging.getLogger("framework.script")

# ids with a 3+ digit run look auto-generated (e.g. "SearchBox129") — don't anchor on them.
_DYNAMIC_ID = re.compile(r"\d{3,}")
# Framework-generated id families whose FULL id regenerates per render (a mount-order counter
# changes every time), so they are never a durable anchor even without a 3-digit run:
#   react-select-6-input / react-select-9-option-0   (react-select — the marquee offender here)
#   :r3: / :ra:                                       (React useId / Radix / MUI)
#   mui-42 / headlessui-menu-3                        (MUI / Headless UI)
#   TextField99 / Toggle21 / Dropdown4                Fluent getId() — a MOUNT counter
_FRAMEWORK_ID = re.compile(
    r"^(react-select-\d+|:r[0-9a-z]+:|mui-\d+|headlessui-[\w-]*\d+|radix-[\w:-]+)",
    re.IGNORECASE,
)
# Fluent's getId('TextField') ids are a per-mount counter, so the SAME control is
# TextField69 one run and TextField99 the next (run 20260818, Net-to-Gross popup: the
# fill's only anchors were css=[id="TextField99"] and an xpath gated on that id — "no
# match" and "positional drift", a step that could never replay). _DYNAMIC_ID's 3+ digit
# rule misses a young counter, so the family is listed explicitly. NOT a generic
# letters+digits rule: that also swallows btnReverseCalc10, the app-authored pencil id
# this very skill anchors its working click steps on.
_FLUENT_COUNTER_ID = re.compile(
    r"^(TextField|Toggle|Dropdown|ComboBox|Checkbox|ChoiceGroup|DatePicker|SpinButton"
    r"|Slider|Callout|Layer|Tooltip|id__)\d+$"
)
# A react-select descendant id carries a STABLE suffix ("option-0", "listbox") we can anchor on
# independently of the volatile instance counter. GROUPED menus nest the index
# ("option-0-0" = first option of the first group — observed on the VAT select), hence the
# (?:-\d+)* tail.
_REACT_SELECT_PART = re.compile(r"^react-select-\d+-(?P<part>option-\d+(?:-\d+)*|listbox|placeholder)$")
# Generic twin of the react-select rule: an auto-generated id whose LEADING token carries
# the volatile digit run but whose tail is stable ("row26102-0-checkbox" → "-0-checkbox").
# Virtualized grids (Fluent DetailsList) regenerate the row-block prefix per data load
# while the tail — including the row INDEX — is the element's durable identity.
_TRAILING_STABLE_SUFFIX = re.compile(r"^[A-Za-z_]+\d{3,}(?P<suffix>(?:-[A-Za-z0-9_]+)+)$")
_RS_OPTION = re.compile(r"^(?P<instance>react-select-\d+)-option-\d+(?:-\d+)*$")
_RS_INPUT = re.compile(r"^(?P<instance>react-select-\d+)-input$")
# Playwright ARIA roles we can target with get_by_role. Recorded elements carry either an explicit
# `role` attribute or a tag we can map to an implicit role.
_TAG_ROLE = {"a": "link", "button": "button"}


def _is_dynamic_id(idv: str) -> bool:
    """True if an id is auto-generated and unsafe to anchor on across re-renders."""
    return bool(_DYNAMIC_ID.search(idv) or _FRAMEWORK_ID.match(idv)
                or _FLUENT_COUNTER_ID.match(idv))


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
      0. the positional xpath — deterministic-first (2026-08-12, user choice): the
         recorded position leads, sanity-gated at resolve time against the fingerprint
         (_resolve refuses a drifted hit and falls through to the semantic ladder)
      1. get_by_role(role, name)  — semantic + unique, the most durable web locator
      2. a non-auto-generated id
      3. react-select option suffix ([id$="-option-0"]) — instance-counter-independent
      4. a distinguishing attribute (data-testid / name / aria-label / title / placeholder)
      5. href (links)
      6. exact accessible-name text
    """
    attrs = element.get("attributes") or {}
    tag = (element.get("node_name") or "").lower()
    ax_name = (element.get("ax_name") or "").strip()
    cands: list[str] = []
    xpath = element.get("x_path")
    if xpath:
        cands.append("xpath=/" + xpath.lstrip("/"))
    cands.extend(_selectors_from_parts(tag, attrs, ax_name))
    return cands


def _selectors_from_parts(tag: str, attrs: dict[str, Any], ax_name: str) -> list[str]:
    """Ranked HARD-IDENTITY candidates from raw element parts. Shared by compile-time
    `_selectors` and replay-time heal promotion, so a healed winner is ranked through the
    exact same durability policy as a freshly recorded element.

    Attribute identity ONLY — no `role=[name=…]`, no `text="…"` (2026-08-13, user
    directive "use xpaths for everything, no random searches"). Those two locate by
    what an element SAYS, which is what made replays hunt for elements and land on the
    wrong one; the recorded xpath (rank 0, added by `_selectors`) plus these attributes
    locate WHERE it is and WHAT it is. An element with neither is unanchorable by
    design — compile marks it and the segment authors live rather than replaying a
    search. (Dropdown OPTION picks keep name matching in `_dropdown_option_steps`: an
    option is chosen BY ITS VALUE, and a positional xpath there would pick whatever row
    happens to sit in that slot.)"""
    # A real label is short. A long ax_name is a screen-reader announcement (react-select emits
    # "option Bike, selected. Select is focused, type to refine list, ..." onto its cell), which
    # changes every render and must never anchor a selector.
    ax_name = (ax_name or "").strip()
    if len(ax_name) > 60:
        ax_name = ""
    cands: list[str] = []

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
        # 3b. Generic volatile-prefix id with a stable tail (virtualized grid rows).
        #     An ambiguous suffix is safe: _resolve requires a unique match per candidate
        #     and falls through to the next selector on a miss.
        elif _is_dynamic_id(idv) and not _FRAMEWORK_ID.match(idv):
            m2 = _TRAILING_STABLE_SUFFIX.match(idv)
            if m2:
                cands.append(f'css=[id$="{_esc(m2.group("suffix"))}"]')

    # 4. Distinguishing attributes.
    for key in ("data-testid", "data-automationid", "name", "aria-label", "title",
                "placeholder"):
        if attrs.get(key):
            cands.append(_attr_sel(key, attrs[key]))
    # 5. href for links.
    if attrs.get("href"):
        cands.append(f'css={tag or "*"}[href="{_esc(attrs["href"])}"]')

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
# How many listing lines below an element's own line may carry its child text.
_SM_TEXT_LINES = 3
# Listing lines that are page-edge markers, not element text.
_SM_EDGE = re.compile(r"pixels? (above|below)|^\[?(Start|End) of page|^\.\.\.")


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
_FP_ATTRS = ("id", "name", "aria-label", "placeholder", "title", "data-testid",
             "data-automationid", "type", "href")


def _sm_child_text(state_message: str, backend_id: Any) -> str:
    """A recorded element's visible child text, recovered from browser-use's DOM listing.

    browser-use leaves ax_name null on plain containers whose text the AX tree does not
    NAME (observed live: the 'Sent' status-chip <span>), and the element dict carries no
    innerText — so compile used to collapse such targets to a bare positional xpath,
    which resolves to whatever sits at that path next run and clicks it silently. The
    listing renders an element's child text on the line(s) directly below its
    [backend_id] line; recover it so the step still gets a text= candidate, a text
    fingerprint, and a name to verify the landed click against."""
    if not state_message or backend_id is None:
        return ""
    lines = state_message.splitlines()
    anchor = next((i for i, ln in enumerate(lines) if f"[{backend_id}]<" in ln), None)
    if anchor is None:
        return ""
    elem_indent = len(lines[anchor]) - len(lines[anchor].lstrip())
    parts: list[str] = []
    for ln in lines[anchor + 1: anchor + 1 + _SM_TEXT_LINES]:
        if _SM_LINE.search(ln):
            break  # the next element's line — end of this element's own text
        text = ln.strip()
        # An element's OWN text is rendered strictly DEEPER than its line; same-or-
        # shallower text belongs to a sibling (observed live: 'Select file' from the
        # adjacent upload control, listed flat right under the Notes textarea).
        if len(ln) - len(ln.lstrip()) <= elem_indent:
            break
        if _SM_EDGE.search(text):
            break
        if text:
            parts.append(text)
    text = " ".join(parts).strip()
    # Same policy as ax_name: a long blob is a container/announcement, not a label.
    return text if 0 < len(text) <= 60 else ""


def _with_recovered_text(element: dict[str, Any] | None,
                         state_message: str) -> dict[str, Any] | None:
    """The recorded element, with ax_name recovered from the DOM listing when the
    recorder left it empty (no-op otherwise). Feeds _selectors / _fingerprint /
    _dropdown_option_steps, so a label-less capture still anchors semantically instead
    of xpath-only. Editable elements are exempt: they have no text children of their
    own — anything below their listing line is a label/placeholder or a sibling."""
    if not element or (element.get("ax_name") or "").strip() \
            or (element.get("node_name") or "").lower() in _EDITABLE_TAGS:
        return element
    text = _sm_child_text(state_message, element.get("backend_node_id"))
    if not text:
        return element
    enriched = dict(element)
    enriched["ax_name"] = text
    return enriched


def _refused_scroll(results: list[Any], i: int) -> bool:
    """Did the scroll action at history position `i` refuse instead of moving anything?
    (agent_tools stamps metadata no_scroll when a dismiss-on-scroll popup was open.)"""
    md = results[i].get("metadata") if i < len(results) and \
        isinstance(results[i], dict) else None
    return bool(isinstance(md, dict) and md.get("no_scroll"))


def _ax_label(element: dict[str, Any]) -> str:
    """The element's short accessible name for landed-click verification ('' when the
    recorder captured none, or only a >60-char announcement blob). Editable elements
    never qualify: the verifier reads inner_text/aria-label, and a field's name lives
    in an external <label> the landed check cannot see — expecting it would refuse the
    RIGHT field."""
    if (element.get("node_name") or "").lower() in _EDITABLE_TAGS:
        return ""
    ax = (element.get("ax_name") or "").strip()
    return ax if len(ax) <= 60 else ""


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
    """Append a step, distinguishing slow-app retries from intentional repeats.

    An ADJACENT click on the same target (no recorded wait between) is the agent
    retrying a click the slow app hadn't registered yet ("+ Invoice" clicked twice) —
    replay doesn't need it, so it is dropped as before. A same-target click SEPARATED
    by a recorded wait is a deliberate cadence ("click Save & Next exactly 5 times,
    waiting after each") — dropping those compiled the 5x/14x counter recordings to a
    single click (observed 2026-08-12). Such repeats now absorb into the first click's
    `count`, and the intervening waits become its `repeat_wait_s` floor instead of
    bare wait steps. For a repeated fill on the same field the latest value still
    wins (the last write is the one the form kept).
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
                return
            waits_between = steps[prev_idx + 1:]   # only waits, by prev_idx construction
            if not waits_between:
                return  # adjacent retry → the first click already fired
            prev["count"] = int(prev.get("count", 1)) + 1
            wait_s = max((float(w.get("seconds") or 0) for w in waits_between), default=0.0)
            prev["repeat_wait_s"] = max(float(prev.get("repeat_wait_s", 0.0)), wait_s)
            del steps[prev_idx + 1:]
            return
    steps.append(step)


# Playwright key names accepted by keyboard.press (plus single characters and F1-F12);
# anything else in a send_keys payload is TEXT to be typed, not a key to be pressed.
_KEY_NAMES = {
    "enter", "tab", "escape", "esc", "backspace", "delete", "insert", "home", "end",
    "pageup", "pagedown", "arrowup", "arrowdown", "arrowleft", "arrowright", "space",
    "capslock", "contextmenu", "numlock", "pause", "printscreen",
}
_MODIFIER_NAMES = {"control", "ctrl", "shift", "alt", "meta", "command", "cmd"}
_FKEY_RE = re.compile(r"f[1-9]|f1[0-2]", re.IGNORECASE)


def _is_key_chord(keys: str) -> bool:
    """Is this a keyboard.press key or chord ("Enter", "Control+a", "ArrowDown")?"""
    parts = [p.strip() for p in str(keys).split("+") if p.strip()]
    if not parts:
        return False
    *mods, last = parts
    if any(m.lower() not in _MODIFIER_NAMES for m in mods):
        return False
    return (len(last) == 1 or last.lower() in _KEY_NAMES
            or _FKEY_RE.fullmatch(last) is not None)


def _mirror_enter(steps: list[dict[str, Any]], enter_after: bool) -> None:
    """Mirror the auto-Enter input tool's Enter (agent_tools) as a replay `press` step."""
    if enter_after:
        _push_step(steps, {"action": "press", "keys": "Enter"})


def _indexed_id_readings(step: dict[str, Any]) -> list[tuple[int, str]] | None:
    """All (index, tail) readings of a click step's element id.

    "row26102-0-checkbox" reads as (0, "checkbox"); an id with several integer tokens
    yields one reading per token and the run matcher keeps whichever tail stays
    constant. None for non-clicks, clicks already carrying a repeat count, and ids
    without a pure-integer token."""
    if step.get("action") != "click" or step.get("count"):
        return None
    idv = str(((step.get("fingerprint") or {}).get("attrs") or {}).get("id") or "")
    tokens = idv.split("-")
    if len(tokens) < 2:
        return None
    readings = [(int(tok), "-".join(tokens[pos + 1:]))
                for pos, tok in enumerate(tokens) if tok.isdigit()]
    return readings or None


def _collapse_indexed_runs(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a run of ≥3 clicks over id-indexed siblings (virtualized grid rows,
    "row{volatile}-{n}-checkbox") into ONE `click_indexed` template step.

    The rows' structural xpaths are identical and their id prefixes regenerate per
    data load, so N baked anchors cannot replay; the id's index component is the only
    durable per-row identity. Interleaved waits/scrolls INSIDE the run are absorbed
    (replay scrolls each index into view itself); a wait after the last member
    survives as the run's settle. Observed indices are normalized to the contiguous
    span from the lowest (length = number of recorded clicks): an authoring quirk — a
    skipped or doubled row — must not become the routine."""
    out: list[dict[str, Any]] = []
    i, n = 0, len(steps)
    while i < n:
        best: tuple[int, list[int], dict[str, Any], str] | None = None
        for idx0, tail in _indexed_id_readings(steps[i]) or []:
            indices, j, last_end = [idx0], i + 1, i + 1
            while j < n:
                st = steps[j]
                if st.get("action") in ("wait", "scroll"):
                    j += 1
                    continue
                match = next((v for v, t in (_indexed_id_readings(st) or [])
                              if t == tail and v > indices[-1]), None)
                if match is None:
                    break
                indices.append(match)
                j += 1
                last_end = j
            if len(indices) >= 3 and (best is None or len(indices) > len(best[1])):
                best = (last_end, indices, steps[i], tail)
        if best is None:
            out.append(steps[i])
            i += 1
            continue
        last_end, indices, first, tail = best
        collapsed: dict[str, Any] = {
            "action": "click_indexed",
            "selector_template": f'css=[id$="-{{n}}{("-" + tail) if tail else ""}"]',
            "start": min(indices),
            "count": len(indices),
        }
        if first.get("fingerprint"):
            collapsed["fingerprint"] = first["fingerprint"]
        out.append(collapsed)
        i = last_end
    return out


# The click count a slice's wording pins: "exactly 5 clicks" / "exactly 14 more
# clicks" / "exactly 3 times".
_REPEAT_HINT_RE = re.compile(r"\bexactly\s+(\d+)\s+(?:more\s+)?(?:clicks?|times)\b",
                             re.IGNORECASE)


def repeat_hint_from_wording(prompt: str) -> int | None:
    """The repeat count the slice wording states, or None. Compile normalizes a lone
    repeat cluster to this number: the recorded count can be off by a collapsed
    retry, and the wording is the contract the counter slices are written against."""
    m = _REPEAT_HINT_RE.search(prompt or "")
    return int(m.group(1)) if m else None


def _apply_repeat_hint(steps: list[dict[str, Any]], hint: int | None) -> list[dict[str, Any]]:
    """Reconcile recorded repeat clusters with the slice WORDING.

    Wording that pins a count ("exactly 5 clicks") keeps its single cluster and has the
    number normalized to the wording — the recorded count can be off by a collapsed
    retry, and the wording is the contract the counter slices are written against. Never
    invents a cluster (a lone click stays a lone click) and never guesses among several.

    Wording that declares NO cadence DISSOLVES its clusters back to one click. _push_step
    reads two same-target clicks separated by a wait as a deliberate rhythm, which is only
    true when the slice asked for a rhythm. On a TOGGLE it is actively wrong: the Download
    menu button recorded click/wait/click (the agent re-clicking what it thought had not
    registered) compiled to repeat_click('download', 2) — open the menu, then shut it —
    and every replay then hunted Excel on a closed menu, which is the reported symptom
    (segment e665b42d2c22fcee, 2026-08-24; the gate never caught it because the skill
    still downloaded *something*: uses=2, fail_count=0). Undeclared same-target repeats
    are retries, and one click is the faithful reading of a retry.

    The recorded inter-click wait survives as a plain wait step: the second click is gone,
    but the app still earned that settle time before whatever follows."""
    clusters = [s for s in steps
                if s.get("action") == "click" and int(s.get("count", 1)) > 1]
    if not clusters:
        return steps
    if hint:
        if len(clusters) == 1 and int(clusters[0]["count"]) != int(hint):
            logger.warning("repeat cluster recorded %s clicks but the wording says "
                           "exactly %s — pinning to the wording",
                           clusters[0]["count"], hint)
            clusters[0]["count"] = int(hint)
        return steps
    out: list[dict[str, Any]] = []
    for step in steps:
        if step.get("action") == "click" and int(step.get("count", 1)) > 1:
            logger.info("compile: dissolving a %sx repeat on one target — the slice "
                        "wording declares no cadence, so the extra clicks are retries "
                        "(a toggle re-clicked twice would undo itself)",
                        step.get("count"))
            wait_s = float(step.get("repeat_wait_s", 0.0))
            step = {k: v for k, v in step.items()
                    if k not in ("count", "repeat_wait_s")}
            out.append(step)
            if wait_s > 0:
                out.append({"action": "wait", "seconds": min(wait_s, 3.0)})
            continue
        out.append(step)
    return out


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
        ], "expect_text": "Create"}]
    ax_name = (element.get("ax_name") or "").strip()
    # Skip wait AND press steps: the auto-Enter input tool interleaves `press` after fills,
    # and the filter fill this option click belongs to may sit behind one.
    prev = next((s for s in reversed(steps) if s.get("action") not in ("wait", "press")), None)
    typed = ""
    if prev is not None and prev.get("action") in ("fill", "type") and \
            prev.get("field_id") == match.group("instance"):
        typed = str(prev.get("value") or prev.get("text") or "").strip()
    label = ax_name or typed
    if not label:
        return None  # no label anywhere — generic (positional) handling is all we have
    part = _REACT_SELECT_PART.match(attrs.get("id") or "")
    positional = f'css=[id$="-{part.group("part")}"]' if part else 'css=[id$="-option-0"]'
    sels = [
        f'role=option[name="{_esc(label)}"]',
        f'text="{_esc(label)}"',
        positional if typed else 'css=[id$="-option-0"]',
    ]
    xpath = element.get("x_path")
    if xpath and ax_name:
        # Exact location first — safe here ONLY because the pick is name-guarded below:
        # if the option now sitting at that position is a different one, expect_text
        # refuses it and the by-name candidate takes over. Without a recorded name
        # (filter-typed picks) the xpath stays out: nothing would catch a wrong row.
        sels.insert(0, "xpath=/" + str(xpath).lstrip("/"))
    click_step = {"action": "click", "selectors": sels}
    if ax_name:
        # The pick must land on an option NAMED what was recorded — the positional
        # fallback otherwise clicks whatever now sits at that index. Only the element's
        # own name qualifies: a typed filter may be a partial label, and expecting it
        # verbatim would refuse the legitimately-filtered option.
        click_step["expect_text"] = ax_name
    if typed:
        # The filter is already a recorded fill step; just click the option it filtered to.
        return [click_step]
    type_step = {
        "action": "type", "text": label,
        # Field identity for parameterize(): the select instance this menu belongs to.
        "field_id": match.group("instance"),
    }
    return [type_step, click_step]


# Actions that only OBSERVE the page. They cannot have caused an off-site navigation, so
# the site-boundary invariant never drops them: if the site redirected spontaneously around
# one (an ad firing during the settle), dropping the read would not prevent the redirect at
# replay — and it may carry an observation later steps need.
_READ_ONLY_ACTIONS = {"extract_data", "capped_scroll", "scroll", "scroll_panels",
                      "wait", "done"}


def _reg_host(url: Any) -> str:
    """Registrable-ish host for the site-boundary check: the last two labels, so
    www.fakenamegenerator.com == fakenamegenerator.com but != accounts.google.com.
    Non-http(s)/unparsable urls yield '' (the check disables itself)."""
    u = str(url or "")
    if not u.startswith("http"):
        return ""
    try:
        host = (urlparse(u).hostname or "").lower()
    except Exception:  # noqa: BLE001 - malformed url in an old recording
        return ""
    return ".".join(host.split(".")[-2:]) if host else ""


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
    # Site-boundary invariant: a segment is single-site by construction (main segments live
    # on the app, aux segments on their tab_url host — cross-site work is split into
    # separate subtasks). So a recorded interaction whose CONSEQUENCE was leaving the
    # segment's site is never load-bearing: it was a stray hit on an ad/SSO overlay, and the
    # recording's own recovery (the explicit navigate back, which compiles to a goto) is the
    # authoritative continuation. Compiling the exit click makes replay REQUIRE the overlay
    # (observed live: a committed click on a "Reload"-named control that had bounced the
    # helper tab to accounts.google.com; healthy replays then died hunting it). The
    # recording itself stays untouched — only the compiled script skips the step.
    seg_host = _reg_host((history[0].get("state") or {}).get("url") if history else "")
    for item_idx, item in enumerate(history):
        actions = (item.get("model_output") or {}).get("action") or []
        elements = (item.get("state") or {}).get("interacted_element") or []
        # ActionResults for this step, aligned to actions (one action per step in this app).
        # find_by_text stashes the element it clicked here (agent_tools.py) since a custom
        # action gets no state.interacted_element.
        results = item.get("result") or []
        exits_site = False
        if seg_host and item_idx + 1 < len(history):
            here = _reg_host((item.get("state") or {}).get("url"))
            after = _reg_host((history[item_idx + 1].get("state") or {}).get("url"))
            exits_site = here == seg_host and bool(after) and after != seg_host
        for i, action in enumerate(actions):
            if not action:
                continue
            name = next(iter(action))
            params = action[name] or {}
            # An action with NO result slot never ran: multi_act stops the queue on a
            # refusal or error, leaving the later actions of that step undispatched.
            # Compiling them bakes phantom steps — a find_by_text that never happened
            # became find_click('Aaran Duncan') and failed every replay of the
            # pay-forecast segment (run 20260813_132507). Results are aligned to
            # actions by index; a wholly result-less item is legacy shape, left alone.
            if results and i >= len(results) and name != "done":
                logger.info("compile: skipping recorded %s at step %d — it never "
                            "executed (the step's action queue stopped earlier)",
                            name, item_idx)
                continue
            if exits_site and name not in _READ_ONLY_ACTIONS:
                logger.warning(
                    "compile: dropping recorded %s at step %d — it navigated the tab off "
                    "the segment's site (%s); the recorded recovery goto that follows is "
                    "the replayable path", name, item_idx, seg_host)
                continue
            element = elements[i] if i < len(elements) else None
            if name == "find_by_text" and params.get("click_first"):
                # A navigation/click made via find_by_text: recover its target element from the
                # recorded metadata and compile it exactly like a built-in click. Without this,
                # every find_by_text click (menus, Sales, btnInvoice, Save, ...) is dropped and
                # the replay skeleton collapses.
                element, md = None, None
                if i < len(results) and isinstance(results[i], dict):
                    md = results[i].get("metadata")
                    if isinstance(md, dict):
                        element = md.get("interacted_element")
                if isinstance(md, dict) and md.get("no_click"):
                    # The tool clicked NOTHING (a 0-match probe or a candidate listing).
                    # Compiling it would bake a phantom click: a conditional guard's
                    # closed-panel probe became find_click('save') and failed every
                    # replay on the healthy page. Metadata-less results keep the legacy
                    # semantic-find_click fallback (save_history used to drop metadata).
                    continue
                query = str(params.get("text") or "").strip()
                element = _with_recovered_text(element,
                                               item.get("state_message") or "")
                synth = (_dropdown_option_steps(element, steps,
                                                item.get("state_message") or "")
                         if element else None)
                if synth is not None:
                    for s in synth:
                        _push_step(steps, s)
                else:
                    # hidden_ok: find_by_text can reach controls a re-render hides;
                    # replay keeps the hover/dispatch recovery as a safety net. A hidden
                    # capture that carries identity is anchored like any other click —
                    # only a capture with NO identity at all is unanchorable.
                    sels = _selectors(element) if element else []
                    if sels:
                        step = {"action": "click", "selectors": sels, "hidden_ok": True}
                        # The landed element must carry the recorded name (or, for a
                        # blob-named row, the query the tool matched on): an xpath or
                        # stale-href fallback resolving into a DIFFERENT control must
                        # refuse, not click (the wrong-row guard, now also for
                        # unparameterized clicks).
                        ax = _ax_label(element)
                        label = ax or query
                        if label:
                            step["expect_text"] = label
                        if label and not ax:
                            # The name is the agent's QUERY, and find_by_text matched it
                            # on a haystack (text + every descendant icon label, each
                            # token anywhere) — not as the element's own name. Replay's
                            # consecutive-token rule can never satisfy that: the Feb-27
                            # grid row took 'Feb-27' from its month cell and 'Net to
                            # gross' from the pencil's aria-label, so every replay
                            # refused it and only the hover fallback clicked (run
                            # 20260818). Record the provenance so _resolve verifies the
                            # way the query matched.
                            step["expect_scattered"] = True
                        _push_step(steps, _attach_fp(step, element))
                    else:
                        # No anchorable identity was captured. Recording the tool's TEXT
                        # SEARCH instead is what put find_click('Net to gross') — 36
                        # matches — into replays; a recording must locate elements, never
                        # hunt for them. Mark it: _author_segment refuses the commit and
                        # this segment authors live every run until it can be anchored.
                        _push_step(steps, {
                            "action": "unanchorable",
                            "why": f"find_by_text click on {query!r} captured no anchorable "
                                   f"element identity (no xpath, no distinguishing attribute)"})
            elif name == "extract_data":
                # The tool records {label, value, query, interacted_element} in its result
                # metadata (persisted by runner.restore_result_metadata); a valueless call
                # records NO metadata and so compiles to nothing. The recorded value is
                # provenance only — replay re-reads the element's CURRENT text, which is
                # the whole point of an extract step (fresh data on every run).
                md = (results[i].get("metadata")
                      if i < len(results) and isinstance(results[i], dict) else None)
                ext = md.get("extract") if isinstance(md, dict) else None
                if not isinstance(ext, dict):
                    continue
                ext_label = str(ext.get("label") or "value")
                ext_query = str(ext.get("query") or params.get("text") or "").strip()
                ext_element = ext.get("interacted_element")
                sels = _selectors(ext_element) if ext_element else []
                if sels:
                    step = _attach_fp({"action": "extract", "selectors": sels,
                                       "label": ext_label}, ext_element)
                    if ext_query:
                        # Kept as the semantic fallback when every selector goes stale.
                        step["query"] = ext_query
                    _push_step(steps, step)
                elif ext_query:
                    # No stable element identity: replay re-finds the value by query with
                    # the same in-page algorithm the tool used (RAW_FIND_JS, no click).
                    _push_step(steps, {"action": "extract", "label": ext_label,
                                       "query": ext_query})
            elif name in ("select_dropdown", "select_dropdown_option"):
                # A pick on a NATIVE <select> (helper/public pages — the app's react-selects
                # go through click steps instead). Dropping these used to compile the
                # surrounding flow WITHOUT the picks, so replay submitted the form with its
                # defaults (observed live: fakenamegenerator replayed onto gen-random-us-us
                # instead of gd-uk). Replay picks BY LABEL via select_option, which fires
                # the change events the page's own scripts listen for.
                option = str(params.get("text", ""))
                if option and element and \
                        (element.get("node_name") or "").lower() == "select":
                    sels = _selectors(element)
                    if sels:
                        _push_step(steps, _attach_fp(
                            {"action": "select", "selectors": sels, "value": option},
                            element))
                elif option and element:
                    # Custom-combobox pick: the tool records the OPTION element it clicked
                    # (react-select option id + the option text as ax_name). Reuse the same
                    # by-label synthesis as a recorded option click; for non-react-select
                    # widgets fall back to a role=option click by name.
                    synth = _dropdown_option_steps(element, steps,
                                                   item.get("state_message") or "")
                    if synth is None and option:
                        # Type-then-pick, never a bare option click: the menu's list is
                        # only populated once the filter text is typed, so a lone click
                        # hunts for an option that does not exist yet (the pay-forecast
                        # employee pick missed on two consecutive runs —
                        # `role=option[name="Abdullah Reilly"] -> no match`). codegen
                        # collapses this pair into api.select_option(value), which opens
                        # the menu, filters, picks by VALUE and reads the value back.
                        attrs_el = element.get("attributes") or {}
                        is_option_el = (
                            str(attrs_el.get("role") or "").lower() == "option"
                            or "-option-" in str(attrs_el.get("id") or ""))
                        synth = []
                        opt_sels = [f'role=option[name="{_esc(option)}"]',
                                    f'text="{_esc(option)}"']
                        xp = element.get("x_path")
                        if is_option_el:
                            if xp:
                                # The recorded row itself — guarded by expect_text below,
                                # so a shifted list refuses rather than picking whoever
                                # now sits in that slot.
                                opt_sels.insert(0, "xpath=/" + str(xp).lstrip("/"))
                        else:
                            # The recorded element is the COMBOBOX, not a row: the tool
                            # opened the menu, typed and picked inside one action, so
                            # nothing else in the script opens it. Emit that opener —
                            # without it replay searched a CLOSED menu and the gender
                            # pick failed every replay (run 20260814_113403). The
                            # combobox's own xpath must never anchor the option row.
                            opener = _selectors(element)
                            if opener:
                                synth.append(_attach_fp(
                                    {"action": "click", "selectors": opener}, element))
                        synth += [
                            {"action": "type", "text": option},
                            {"action": "click", "selectors": opt_sels,
                             "expect_text": option},
                        ]
                    for s in synth:
                        _push_step(steps, s)
                elif option:
                    logger.warning(
                        "select_dropdown %r recorded without a <select> element identity; "
                        "step dropped (replay gate will catch a wrong end state)", option)
            elif name == "navigate" and params.get("url"):
                _push_step(steps, {"action": "goto", "url": params["url"]})
            elif name == "click" and element:
                element = _with_recovered_text(element,
                                               item.get("state_message") or "")
                synth = _dropdown_option_steps(element, steps,
                                               item.get("state_message") or "")
                if synth is not None:
                    for s in synth:
                        _push_step(steps, s)
                    continue
                sels = _selectors(element)
                if sels:
                    step = {"action": "click", "selectors": sels}
                    label = _ax_label(element)
                    if label:
                        # Landed-click verification: the acted-on element must be NAMED
                        # what was recorded, whichever selector resolved it (see _resolve).
                        step["expect_text"] = label
                    _push_step(steps, _attach_fp(step, element))
            elif name == "input" and element:
                # The auto-Enter `input` tool (agent_tools.py) presses Enter after typing
                # into non-dropdown fields and flags it in its result metadata (put back
                # into saved recordings by runner.restore_result_metadata). Mirror it with
                # a `press` step; recordings without the flag — dropdown fills and those
                # made with the old tool — compile exactly as they ran.
                md = (results[i].get("metadata")
                      if i < len(results) and isinstance(results[i], dict) else None)
                if isinstance(md, dict) and md.get("no_fill"):
                    # The tool typed NOTHING (dropdown-filter refusal, agent_tools) —
                    # compiling this would bake a phantom fill into the replay, the
                    # fill-shaped twin of the no_click phantom click above.
                    continue
                enter_after = isinstance(md, dict) and bool(md.get("auto_enter"))
                # Date pickers never get the mirrored Enter, whatever the recording says:
                # recordings from the brief window when the live tool pressed Enter after
                # dates (2026-08-07 morning) would otherwise bake that press into every
                # replay. Same rule as the live path — the date commits on blur.
                if (str((element.get("attributes") or {}).get("aria-haspopup") or "")
                        .strip().lower() == "dialog"):
                    enter_after = False
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
                        _mirror_enter(steps, enter_after)
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
                    _mirror_enter(steps, enter_after)
            elif name == "upload_file" and element and params.get("path"):
                # File attach (browser-use upload_file -> set files via CDP, no native
                # dialog). Only the BASENAME is kept: task files live in
                # files.UPLOADS_DIR by convention, which keeps the step portable and
                # its value identical to the string the prompt spells (so parameterize's
                # verbatim-in-prompt lift rule applies). hidden_ok: upload inputs are
                # legitimately display:none behind styled drop zones.
                sels = _selectors(element)
                if sels:
                    _push_step(steps, _attach_fp(
                        {"action": "upload", "selectors": sels,
                         "value": Path(str(params["path"])).name,
                         "hidden_ok": True}, element))
            elif name == "send_keys" and params.get("keys"):
                keys = str(params["keys"])
                if _is_key_chord(keys):
                    _push_step(steps, {"action": "press", "keys": keys})
                else:
                    # Raw TEXT sent through the keyboard tool (the agent typing into an
                    # already-focused field — how the net-to-gross popup got its amount).
                    # Replay must type it: keyboard.press("4000") raises `Unknown key`
                    # and failed that segment on every replay (run 20260814_105247).
                    _push_step(steps, {"action": "type", "text": keys})
            elif name in ("capped_scroll", "scroll", "scroll_panels") and \
                    _refused_scroll(results, i):
                # The tool was REFUSED while a popup was open (agent_tools
                # _refuse_if_callout): nothing moved. Compiling it would bake the very
                # page-scroll that dismisses that popup into every future replay — the
                # phantom-action rule the no_click probes and no_fill refusals follow.
                continue
            elif name in ("capped_scroll", "scroll"):
                # Discovery scrolling is load-bearing: the target section must be scrolled
                # into view before the following click can resolve (observed: the Reviews
                # panel's "View all" icon). capped_scroll was OUR tool ({pages}) until
                # 2026-08-18, when the built-in scroll ({num_pages}) replaced it — the
                # name stays handled so recordings made before that still replay.
                pages = params.get("pages", params.get("num_pages", 0.5))
                try:
                    pages = float(pages)
                except (TypeError, ValueError):
                    pages = 0.5
                _push_step(steps, {"action": "scroll",
                                   "down": bool(params.get("down", True)),
                                   "pages": min(pages, 1.0)})
            elif name == "scroll_panels":
                # Container scrolling must stay container scrolling at replay: folding it
                # into a viewport "scroll" step would wheel the page BEHIND the fixed
                # panel the agent was scrolling (run 20260817_110501 — the Add Data
                # Request employee list owns its scroll box).
                pages = params.get("pages", 0.8)
                try:
                    pages = float(pages)
                except (TypeError, ValueError):
                    pages = 0.8
                _push_step(steps, {"action": "scroll_panels",
                                   "pages": max(0.2, min(pages, 1.0))})
            elif name == "wait":
                # Keep the agent's deliberate pauses (capped). They are load-bearing on this
                # slow React app: they let the invoice form and its react-select menus finish
                # rendering before the next click. Dropping them makes fast replay outrun the UI
                # (menu not open yet → click times out; Save fires before state commits → no POST).
                secs = params.get("seconds")
                if isinstance(secs, (int, float)) and secs > 0:
                    _push_step(steps, {"action": "wait", "seconds": min(float(secs), 3.0)})
            # `done` is intentionally dropped — Playwright auto-waits on locators.
    return _collapse_indexed_runs(steps)


def normalize_block_text(text: Any) -> str:
    """Whitespace-collapse WITHIN each line, drop blank lines, keep the line breaks.

    Extract values must normalize identically whether they were read by the authoring
    tool or by a replayed skill: bindings slice fields out by LINE POSITION, so a
    replay that flattened the block ("Name 12 High St TOWN" instead of three lines)
    silently made every line-transform binding unresolvable — the form-fill segment
    then re-authored with the LLM on every run (observed between runs 20260813_153838
    and _155450)."""
    return "\n".join(
        ln for ln in (" ".join(part.split()) for part in str(text or "").splitlines())
        if ln)


def merge_extract(store: dict[str, str], label: str, value: str) -> str:
    """Record an extract under `label` WITHOUT clobbering a different value already
    there — a collision stores under `label_2`, `label_3`, … and a re-read of an
    already-stored value is a no-op (agent retries must not multiply keys).

    Values are stored WHOLE (line structure preserved). Auto-splitting them into
    per-field keys was tried 2026-08-12 and removed 2026-08-13: the sub-keys came from
    page CONTENT, so an address block yielded `pilton` on one run and `catterline` on
    the next, and any binding keyed on one could never resolve again. Fields are sliced
    out of the whole block by the binding transforms instead (line index / date
    reformat) — a line POSITION is stable across identities, a town name is not.

    The clobbering this replaces lost real data (observed live: the generated NAME was
    extracted as 'identity_block', then the ADDRESS extract reused the label and
    silently overwrote it — replays then fed consumers a nameless identity and the
    Add-Employee step failed honestly with 'missing generated name'). Returns the key
    used. Shared by every aggregation point: SkillApi.extract (tier 1), run_steps
    (tier 0), and hybrid._history_extracts (agent histories)."""
    label, value = str(label), str(value)
    for n in range(1, 10):
        key = label if n == 1 else f"{label}_{n}"
        if key not in store:
            store[key] = value
            return key
        if str(store[key]).strip() == value.strip():
            return key
    store[key] = value  # pathological collision depth: last slot wins
    return key


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a half-written file."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_steps(
    recording_path: str | Path, steps_path: str | Path, max_steps: int | None = None, *,
    emit_start_goto: bool = True, repeat_hint: int | None = None,
) -> list[dict[str, Any]]:
    """Compile `recording_path` and write the step list to `steps_path` atomically.
    `repeat_hint` is the slice wording's "exactly N clicks" count (see
    repeat_hint_from_wording) — it pins a lone repeat cluster's count."""
    steps = _apply_repeat_hint(
        compile_recording(recording_path, max_steps=max_steps,
                          emit_start_goto=emit_start_goto),
        repeat_hint)
    _atomic_write(Path(steps_path), json.dumps(steps, indent=2))
    return steps


# Composed-tree helpers, spliced verbatim into BOTH raw finders below. The app renders
# some surfaces (the Add Expenses or Benefits modal) behind an OPEN shadow root; a
# body-scoped querySelectorAll cannot see them, so find_by_text answered "the element is
# not in this page's DOM" for a label visibly on screen (run 20260805_123407_334719) and
# the agent fell back to guessing anonymous combobox indexes. walkAll descends through
# open shadow roots (closed ones expose no .shadowRoot and stay invisible by
# construction); composedParent/composedContains cross the same boundaries the walk does.
# No literal '%' may appear here — the host templates are %-formatted.
_COMPOSED_WALK_JS = """\
    var walkAll = function (visit) {
      var stack = [document.body], n, kids, i;
      while (stack.length) {
        n = stack.pop();
        if (n.shadowRoot) stack.push(n.shadowRoot);
        kids = n.children || [];
        for (i = 0; i < kids.length; i++) stack.push(kids[i]);
        if (n !== document.body && n.nodeType === 1) visit(n);
      }
    };
    var composedParent = function (e) {
      if (!e) return null;
      if (e.parentElement) return e.parentElement;
      var r = e.getRootNode ? e.getRootNode() : null;
      return (r && r.host) ? r.host : null;
    };
    var composedContains = function (a, b) {
      for (var p = composedParent(b); p; p = composedParent(p)) {
        if (p === a) return true;
      }
      return false;
    };
    var inShadowTree = function (e) {
      return !!(e.getRootNode && e.getRootNode() !== document);
    };
    // Positional path of an element, document-rooted. Shared by BOTH raw finders so
    // every captured element carries the exact location (2026-08-13: the control
    // finder used to return none, which is why find_by_text clicks could not be
    // anchored and compiled to text searches instead).
    var xpathOf = function (e) {
      if (!e || e === document.documentElement) return '/html';
      var ix = 1, sib, n = 0;
      for (sib = e.parentNode ? e.parentNode.firstElementChild : null; sib;
           sib = sib.nextElementSibling) {
        if (sib.tagName === e.tagName) { n++; if (sib === e) ix = n; }
      }
      return xpathOf(e.parentElement) + '/' + e.tagName.toLowerCase() +
             (n > 1 ? '[' + ix + ']' : '');
    };
    // A positional xpath resolves against the DOCUMENT at replay time; for a
    // shadow-tree element it would land on some unrelated light-DOM node. No anchor
    // is honest — a wrong-element anchor is not.
    var safeXpath = function (e) {
      if (inShadowTree(e)) return '';
      try { return xpathOf(e); } catch (err) { return ''; }
    };
"""

def _query_tokens(s: str) -> list[str]:
    """A query in the raw finders' token form: lowercased, split on non-alphanumerics.
    ONE tokenizer for authoring probes, replay re-finds, and gate checks — it must stay
    in lockstep with the JS-side norm()/toks() in the probes below."""
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]


# The raw-DOM find+click algorithm, SHARED between find_by_text (authoring, agent_tools)
# and the `find_click` replay step: token match over title/aria-label/name/text plus child
# icon hints, ranked visible-first then by NAME SPECIFICITY (exact > word-aligned prefix >
# whole-phrase substring > scattered tokens — so a query 'Reviews' prefers an element NAMED
# 'Reviews' over 'Add reviews', which the substring match alone blind-clicked in live runs),
# scrollIntoView (which scrolls the CORRECT container — unlike window.scrollBy, a no-op
# inside Fluent ScrollablePanes), then the element's own click handler. Using the identical
# implementation at author and replay time is what makes hover-revealed/0-size controls
# (the Reviews "View all" icon) replayable at all.
# Placeholders: %s = JSON token list, %s = "true"/"false" for click.
RAW_FIND_JS = r"""
(function () {
  try {
    var TOKENS = %s, DOCLICK = %s;
    var sel = 'button,a,[role=button],[role=menuitem],[role=tab],[role=link],' +
              'input[type=button],input[type=submit],[data-is-focusable],[onclick]';
__WALK__
    var out = [];
    walkAll(function (e) {
      if (!e.matches || !e.matches(sel)) return;
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
    // Rank by how specifically the accessible NAME matches the query phrase (normalized
    // with the same token grammar the callers use). Visibility stays the primary key.
    var PHRASE = TOKENS.join(' ');
    var normName = function (s) {
      return String(s || '').toLowerCase().split(/[^a-z0-9]+/)
        .filter(function (t) { return t; }).join(' ');
    };
    out.forEach(function (o) {
      var nm = normName(o.name);
      o.rank = nm === PHRASE ? 0
             : nm.indexOf(PHRASE + ' ') === 0 ? 1
             : (' ' + nm + ' ').indexOf(' ' + PHRASE + ' ') !== -1 ? 2
             : 3;
    });
    out.sort(function (a, b) {
      var v = (b.visible ? 1 : 0) - (a.visible ? 1 : 0);
      return v !== 0 ? v : a.rank - b.rank;
    });
    var top = out[0], clicked = false, refused = '';
    // An INVISIBLE candidate whose name does not match the query (rank 2+) is never the
    // intended target — observed live: a 0-size background grid row "clicked" four times
    // with a ✅ receipt while the agent hunted a dropdown option that never existed. The
    // legit hidden-click cases (Fluent 0x0 icons) carry the query as their name (rank<=1).
    if (DOCLICK && !top.visible && top.rank > 1) {
      refused = 'invisible-name-mismatch';
    } else if (DOCLICK) {
      try {
        // A Fluent Callout on screen closes itself the moment anything outside it
        // scrolls, and this click needs no scroll at all: the events below are
        // DISPATCHED, not aimed at coordinates, so they land wherever the element
        // sits. Guarding inside the shared JS keeps authoring (find_by_text) and
        // replay (find_click) byte-identical — a behaviour only one side has is a
        // step that cannot replay.
        var calloutOpen = false, cs = document.querySelectorAll('.ms-Callout');
        for (var ci = 0; ci < cs.length; ci++) {
          var cr = cs[ci].getBoundingClientRect();
          if (cr.width > 0 && cr.height > 0) { calloutOpen = true; break; }
        }
        if (!calloutOpen) top.el.scrollIntoView({ block: 'center' });
        // Full mousedown→mouseup→click sequence: React widgets (react-select options)
        // select on mousedown and ignore a bare .click().
        ['mousedown', 'mouseup', 'click'].forEach(function (t) {
          top.el.dispatchEvent(new MouseEvent(t, { bubbles: true, cancelable: true, view: window }));
        });
        clicked = true;
      } catch (e) {}
    }
    var attrs = {};
    ['id', 'aria-label', 'title', 'name', 'placeholder', 'data-testid', 'href', 'role']
      .forEach(function (a) { var v = top.el.getAttribute(a); if (v) attrs[a] = v; });
    return { count: out.length, clicked: clicked, refused: refused, name: top.name,
             names: out.slice(0, 8).map(function (o) { return o.name; }),
             element: { tag: top.el.tagName.toLowerCase(), attrs: attrs,
                        xpath: safeXpath(top.el) } };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)

# Static-TEXT finder for extract_data and the `extract` replay step. RAW_FIND_JS above
# deliberately scans only control-like elements (it exists to CLICK things); a value shown
# in a plain <h3>/<div> — the generated identity on fakenamegenerator.com was the live
# failure: 15+ extract_data attempts, all "nothing matched" — is invisible to it AND to the
# interactive snapshot. This walks the whole DOM for the DEEPEST visible element whose text
# contains every token and returns that element's rendered text as the value (shortest text
# wins among several deepest matches, i.e. the tightest element around the value; for a
# <select> the SELECTED option's label — its rendered text is every option concatenated,
# which is never the value a user sees chosen). One exception to "tightest wins": when the
# tightest capture is EXACTLY the query (zero information gained — a bare name in its own
# <h3>, the fakenamegenerator identity card), it expands to the nearest ancestor that adds
# text (`expanded: true` in the result), capped at 1000 chars so a page-sized container can
# never replace a tight match. Read-only
# by construction: it clicks nothing. Shared verbatim between authoring (agent_tools) and
# replay (_extract_value) so a text-anchored extract re-reads identically on every run.
#
# The element identity INCLUDES the positional xpath. This is load-bearing for fresh-data
# pages: the captured value's own text is often the ONLY text anchor (an attribute-less
# <h3> holding a generated name), and a text="<old value>" selector can never re-find NEXT
# run's value — the xpath re-reads whatever the same slot shows now (observed live: extract
# anchors text="Kirsty Crawford" were dead on every later run).
# Placeholder: %s = JSON token list.
RAW_TEXT_FIND_JS = r"""
(function () {
  try {
    var TOKENS = %s;
    var body = document.body;
    if (!body) return { count: 0 };
__WALK__
    // OPTION/OPTGROUP are skipped so a <select> stays the DEEPEST match for its own
    // option text (options are zero-rect while the menu is closed and would otherwise
    // knock the select out of the deepest-only filter below, losing the match entirely).
    var SKIP = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1, TEMPLATE: 1, OPTION: 1, OPTGROUP: 1 };
    // The text a capture REPORTS. A <select>'s innerText is every option label
    // concatenated — never the value a user sees chosen — and that is true whether the
    // select IS the match or merely sits INSIDE it. Run 20260818_103055 subtask 1:
    // extract_data('Gender') matched the <label>, expanded to the enclosing form block
    // (below), and reported 'Gender Random Male Female Name set American ... Country
    // Australia ...' — the option lists of all three dropdowns. Substituting each
    // descendant select with its selected option turns that same block into
    // 'Gender Male Name set Scottish Country United Kingdom'.
    // innerText stays the path for select-free elements: it respects visibility, which a
    // textContent walk cannot. The walk runs only where a select is actually present.
    var renderedText = function (el) {
      if (el.tagName === 'SELECT') {
        var sel = el.selectedOptions && el.selectedOptions[0];
        return (sel && sel.text) || el.value || '';
      }
      if (!el.querySelector || !el.querySelector('select')) return el.innerText || '';
      var out = [];
      var walk = function (n) {
        if (n.nodeType === 3) { out.push(n.data); return; }
        if (n.nodeType !== 1) return;
        if (n.tagName === 'SELECT') {
          var o = n.selectedOptions && n.selectedOptions[0];
          out.push((o && o.text) || n.value || '');
          return;
        }
        if (SKIP[n.tagName]) return;
        for (var c = n.firstChild; c; c = c.nextSibling) walk(c);
      };
      walk(el);
      return out.join(' ');
    };
    // One composed pass collects the candidate elements AND the whole-page text for the
    // every-token gate (body.textContent alone misses shadow content).
    var all = [], whole = '';
    walkAll(function (e) {
      all.push(e);
      var cn = e.childNodes;
      for (var k = 0; k < cn.length; k++) {
        if (cn[k].nodeType === 3) whole += cn[k].data + ' ';
      }
    });
    whole = whole.toLowerCase();
    if (!TOKENS.every(function (t) { return whole.indexOf(t) !== -1; })) return { count: 0 };
    var matches = [];
    for (var i = 0; i < all.length; i++) {
      var e = all[i];
      if (SKIP[e.tagName]) continue;
      var t = (e.textContent || '').toLowerCase();
      var ok = true;
      for (var j = 0; j < TOKENS.length; j++) {
        if (t.indexOf(TOKENS[j]) === -1) { ok = false; break; }
      }
      if (ok) matches.push(e);
    }
    if (!matches.length) return { count: 0 };
    // Deepest only: every match's ancestors also match (their text is a superset), so drop
    // any element that contains another match. Containment must cross shadow boundaries
    // the same way the walk does, or a host and its shadow content both survive.
    var deepest = matches.filter(function (e) {
      return !matches.some(function (o) { return o !== e && composedContains(e, o); });
    });
    var scored = [];
    deepest.forEach(function (e) {
      var r = e.getBoundingClientRect();
      if (!(r.width > 0 && r.height > 0)) return;      // display:none / detached
      var text = renderedText(e).replace(/\s+/g, ' ').trim();
      if (!text) return;                                // hidden by an ancestor
      scored.push({ el: e, text: text });
    });
    if (!scored.length) return { count: 0 };
    scored.sort(function (a, b) { return a.text.length - b.text.length; });
    var top = scored[0];
    // (A <select> match already carries its SELECTED option as its text — renderedText
    // resolved it when the candidate was scored. Inputs cannot reach here at all: their
    // textContent is empty, so they never match the tokens.)
    // Zero-information-gain expansion: a capture that is EXACTLY the query teaches nothing
    // (a generated name alone in its own <h3> — the caller wanted the card AROUND it).
    // Climb to the nearest ancestor that adds text; if that first-gaining ancestor is
    // bigger than the 1000-char value cap, keep the tight capture instead of a page blob.
    var expanded = false;
    if (top.el.tagName !== 'SELECT') {
      var normText = function (s) {
        return (s || '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).join(' ');
      };
      var queryNorm = TOKENS.join(' ');
      if (normText(top.text) === queryNorm) {
        var anc = composedParent(top.el);
        while (anc && anc !== document.body) {
          var ancText = renderedText(anc).replace(/\s+/g, ' ').trim();
          if (normText(ancText) !== queryNorm) {
            if (ancText.length && ancText.length <= 1000) {
              top = { el: anc, text: ancText };
              expanded = true;
            }
            break;
          }
          anc = composedParent(anc);
        }
      }
    }
    var attrs = {};
    ['id', 'aria-label', 'title', 'name', 'placeholder', 'data-testid', 'href', 'role']
      .forEach(function (a) { var v = top.el.getAttribute(a); if (v) attrs[a] = v; });
    var xpath = safeXpath(top.el);   // shared walker, spliced in above
    // Per-line text of the chosen block: extract_data stores this line structure so
    // bindings can slice stable line POSITIONS out of it. The NAME stays
    // whitespace-flattened — every matcher assumes flat names.
    // A <select>'s innerText is every option label concatenated — never the chosen
    // value — so its reported text (the SELECTED option, resolved above) is the whole
    // value and there are no block lines to keep.
    // A block holding a <select> has no trustworthy line structure once the options are
    // replaced by the chosen value (the walk joins text, it cannot reproduce layout line
    // breaks), so it reports as one line. The consumer of `lines` is bindings' stable
    // line POSITIONS, and its subject — the generated identity card — holds no selects,
    // so that path keeps innerText's exact line split.
    var blockLines = top.el.tagName === 'SELECT' ? [] :
      String(renderedText(top.el) || '').split('\n')
        .map(function (s) { return s.replace(/\s+/g, ' ').trim(); })
        .filter(function (s) { return s.length > 0; }).slice(0, 60);
    return { count: scored.length, name: top.text.slice(0, 1000), expanded: expanded,
             lines: blockLines,
             names: scored.slice(0, 5).map(function (o) { return o.text.slice(0, 80); }),
             element: { tag: top.el.tagName.toLowerCase(), attrs: attrs, xpath: xpath } };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)

# Live-twin re-find for stale fills (agent_tools `input`). When an earlier action in the
# same step re-renders a form (dropdown picks re-mounting a modal's type-specific fields
# — the observed duplicate-add loop, run 20260805_131827_339055), the fill's index points
# at a DETACHED node: its CDP object id still resolves, keystrokes land in whatever holds
# focus, and a read-back on the dead node either reads '' (false "did NOT take") or the
# typed text (false clean receipt). This finds the CURRENT element with the same tag and
# identity attribute — composed-tree walk, so shadow-rooted forms work — and, op 'focus',
# focuses it (selecting existing text when CLEAR) so a CDP Input.insertText lands there;
# op 'read' reads it back. Exactly one visible match or the caller refuses: a guessed
# twin is the wrong-field bug this exists to kill.
# Placeholders (named): %(tag)s %(attr)s %(value)s %(op)s %(clear)s — all json.dumps'd.
FIELD_REFIND_JS = r"""
(function () {
  try {
    var TAG = %(tag)s, ATTR = %(attr)s, VALUE = %(value)s, OP = %(op)s, CLEAR = %(clear)s;
__WALK__
    var matches = [];
    walkAll(function (e) {
      if (String(e.tagName || '').toLowerCase() !== TAG) return;
      if (String(e.getAttribute(ATTR) || '') !== VALUE) return;
      var r = e.getBoundingClientRect();
      if (!(r.width > 0 && r.height > 0)) return;
      matches.push(e);
    });
    if (matches.length !== 1) return { count: matches.length };
    var el = matches[0];
    if (OP === 'focus') {
      try { el.focus(); } catch (e) {}
      if (CLEAR && el.select) { try { el.select(); } catch (e) {} }
      else if (!CLEAR && el.setSelectionRange) {
        try {
          var L = (el.value || '').length;
          el.setSelectionRange(L, L);
        } catch (e) {}
      }
      var attrs = {};
      ['id', 'role', 'aria-autocomplete', 'placeholder', 'aria-label', 'name', 'type']
        .forEach(function (a) { var v = el.getAttribute(a); if (v) attrs[a] = v; });
      var root = el.getRootNode ? el.getRootNode() : document;
      return { count: 1, focused: root.activeElement === el, attrs: attrs,
               label: el.getAttribute('placeholder') || el.getAttribute('aria-label') ||
                      el.getAttribute('name') || el.getAttribute('id') || TAG };
    }
    if (OP === 'read') {
      return { count: 1,
               value: el.value !== undefined ? String(el.value) : (el.textContent || '') };
    }
    return { error: 'bad-op' };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)

# Dialog probes for the click receipt's DIALOG OUTCOME (agent_tools click override).
# A modal Save in this app closes the dialog and writes over a channel the network
# collector cannot see; with no receipt saying "the dialog closed", the agent re-opened
# and re-saved the same benefit ten times. DIALOG_COUNT_JS counts VISIBLE dialogs
# (role=dialog/alertdialog, aria-modal, or a modal/dialog class token — Fluent's
# ms-Modal/ms-Dialog), composed-tree so shadow-rooted dialogs count too.
DIALOG_COUNT_JS = r"""
(function () {
  try {
__WALK__
    var n = 0;
    walkAll(function (e) {
      var role = String(e.getAttribute('role') || '').toLowerCase();
      var modal = String(e.getAttribute('aria-modal') || '').toLowerCase() === 'true';
      var cls = String(e.getAttribute('class') || '');
      if (!(role === 'dialog' || role === 'alertdialog' || modal ||
            /(^|[\s-])(modal|dialog)([\s-]|$)/i.test(cls))) return;
      var r = e.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) n = n + 1;
    });
    return { open: n };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)

# Identity-tracked in-dialog probe: the global dialog COUNT lies when panels CHAIN — this app's
# Save closes its dialog and immediately opens the next panel (Add-Request → Send-Email,
# run 20260807_095537), so count-delta reported "STILL OPEN" against a save that
# succeeded and the agent redid it. Stamp THE dialog the clicked element lives in;
# the post-click question is then "did THAT dialog close", immune to whatever opened.
DIALOG_WATCH_ATTR = "data-ao-dialog-watch"

# Runs ON the clicked node (this = element): sweep stale stamps everywhere, then stamp
# the node's dialog ancestor (role dialog/alertdialog, aria-modal, or modal/dialog class,
# walking composed parents so shadow-rendered dialogs count).
DIALOG_STAMP_JS = r"""
function () {
  try {
__WALK__
    walkAll(function (e) {
      if (e.hasAttribute && e.hasAttribute('WATCH')) e.removeAttribute('WATCH');
    });
    for (var p = this; p; p = composedParent(p)) {
      if (!p.getAttribute) continue;
      var role = String(p.getAttribute('role') || '').toLowerCase();
      var modal = String(p.getAttribute('aria-modal') || '').toLowerCase() === 'true';
      var cls = String(p.getAttribute('class') || '');
      if (role === 'dialog' || role === 'alertdialog' || modal ||
          /(^|[\s-])(modal|dialog)([\s-]|$)/i.test(cls)) {
        p.setAttribute('WATCH', '1');
        return { stamped: true };
      }
    }
    return { stamped: false };
  } catch (e) { return { error: String(e) }; }
}
""".replace("__WALK__", _COMPOSED_WALK_JS).replace("WATCH", DIALOG_WATCH_ATTR)

# Document-level: is the stamped dialog still mounted with layout? Unmounted or zero-rect
# means THE dialog closed, regardless of chained panels or toasts.
DIALOG_STAMPED_OPEN_JS = r"""
(function () {
  try {
__WALK__
    var found = null, n = 0;
    walkAll(function (e) {
      if (!e.getAttribute) return;
      if (!found && e.hasAttribute && e.hasAttribute('WATCH')) {
        var fr = e.getBoundingClientRect();
        if (fr.width > 0 && fr.height > 0) found = e;
      }
      var role = String(e.getAttribute('role') || '').toLowerCase();
      var modal = String(e.getAttribute('aria-modal') || '').toLowerCase() === 'true';
      var cls = String(e.getAttribute('class') || '');
      if (!(role === 'dialog' || role === 'alertdialog' || modal ||
            /(^|[\s-])(modal|dialog)([\s-]|$)/i.test(cls))) return;
      var r = e.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) n = n + 1;
    });
    // `open` (same predicate as DIALOG_COUNT_JS) corroborates a missing stamp: React
    // re-renders REPLACE the stamped node, so "stamp gone" alone cannot mean "closed".
    return { present: !!found, open: n };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS).replace("WATCH", DIALOG_WATCH_ATTR)

# Reveal stylesheet: the app hides several REAL controls until hover by collapsing their
# wrappers to 0-size (the Reviews "Send NPS survey request" / "Add reviews" icons). Zero-
# layout controls never enter browser-use's interactive snapshot (the agent cannot click
# them by index) and fail replay _resolve's visibility gate — both drivers then depend on
# the RAW_FIND_JS blind-click fallback, whose substring match has misfired in live runs.
# Forcing the wrappers visible gives the controls layout, so both drivers act on them
# natively; the fallbacks above stay untouched for old recordings. Class patterns are
# app-specific and owner-supplied — keep verbatim. Every injection site is gated by
# Config.reveal_hidden_controls.
REVEAL_STYLE_ID = "__ao_reveal_css"

REVEAL_CSS = """\
.buttons-wrapper,
[class*="headerButtonWrapper"],
[class*="buttons-wrapper"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
[class*="headerButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
.hover-item,
[class*="hover-item"],
.containerHover .hover-item {
    display: inline !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
"""

# Guarded installer: appends the <style> once per document, no-op when already present.
# The guard is the element's presence (not a window flag) so a framework that rebuilds
# <head> self-heals on the next injection pass. As a context init script this runs at
# document start where <head> may not exist yet — then it retries on DOMContentLoaded.
# Never throws: styling must not be able to break a step, a login, or a replay.
# (%-formatted ONCE below; if the CSS ever gains a literal '%', escape it as '%%'.)
_REVEAL_INSTALL_TEMPLATE = r"""
(function () {
  try {
    var ID = %s, CSS = %s;
    var install = function () {
      try {
        if (document.getElementById(ID)) return true;
        var root = document.head || document.documentElement;
        if (!root) return false;
        var style = document.createElement('style');
        style.id = ID;
        style.textContent = CSS;
        root.appendChild(style);
        return true;
      } catch (e) { return false; }
    };
    if (!install())
      document.addEventListener('DOMContentLoaded', install, { once: true });
  } catch (e) {}
})()
"""
REVEAL_CSS_JS = _REVEAL_INSTALL_TEMPLATE % (json.dumps(REVEAL_STYLE_ID), json.dumps(REVEAL_CSS))

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


def _names_value(text: str, value: str) -> bool:
    """True when `text` NAMES `value`: the value's tokens appear as a CONSECUTIVE token
    subsequence of the text's tokens (same token grammar as RAW_FIND_JS/_norm_phrase —
    lowercase, split on non-alphanumerics).

    'FUNFOOD LIMITED' does NOT name 'FOOD LIMITED' ('funfood' is not the token 'food' —
    the live wrong-business click this guards), while 'FOOD LIMITED 0123 Monthly' does:
    row links concatenate several cells' text, so plain equality would reject the RIGHT
    row."""
    want = [t for t in re.split(r"[^a-z0-9]+", (value or "").lower()) if t]
    if not want:
        return True
    have = [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]
    return any(have[i:i + len(want)] == want
               for i in range(len(have) - len(want) + 1))


# Identity attributes the xpath sanity gate compares. TEXT is deliberately absent: an
# element's text is the DATA it displays, and the fresh-data pages xpath exists for
# (the generated identity block) show different text every run — comparing it vetoed
# exactly the extracts the anchor was added to serve (run 20260813_132507, seg 1).
_FP_GATE_ATTRS = ("id", "name", "aria-label", "data-testid", "data-automationid")

_FP_GATE_JS = (
    "el => ({tag: el.tagName.toLowerCase(), attrs: Object.fromEntries("
    "['id','name','aria-label','data-testid','data-automationid']"
    ".map(a => [a, el.getAttribute(a)]))})"
)


async def _xpath_matches_fingerprint(candidate: Any,
                                     fingerprint: dict[str, Any] | None) -> bool:
    """Sanity gate for a positional-xpath hit: the landed element must still BE the
    recorded one — same tag and same hard identity attributes. A positional path
    resolves confidently on a shifted layout, onto the WRONG element; identity is what
    tells them apart, and unlike text it does not change when the page's data does.
    Volatile ids are skipped (they are regenerated per render). Nothing to compare →
    allow; unverifiable → refuse (the attribute ladder is right behind)."""
    fp = fingerprint or {}
    if not fp:
        return True
    tag = str(fp.get("tag") or "")
    want = {k: str(v) for k, v in ((fp.get("attrs") or {}).items())
            if k in _FP_GATE_ATTRS and v
            and not (k == "id" and _is_dynamic_id(str(v)))}
    if not tag and not want:
        return True
    try:
        got = await candidate.evaluate(_FP_GATE_JS)
    except Exception:  # noqa: BLE001 - unverifiable = fail this candidate
        return False
    if tag and str((got or {}).get("tag") or "").lower() != tag.lower():
        return False
    got_attrs = (got or {}).get("attrs") or {}
    return all(str(got_attrs.get(k) or "") == v for k, v in want.items())


# Accessible-name approximation for controls whose name lives OUTSIDE the element:
# aria-labelledby targets, then label[for=id], then an enclosing <label>, then title.
# Fluent Toggles are the motivating case (runs 20260817_114057/115232): the recorded
# fingerprint text "Student loan" is the toggle's ACCESSIBILITY name via label[for],
# but the button's own inner_text/aria-label are empty — the gate refused the RIGHT
# element on every replay and archive_if_failing retired the entry after 2 runs.
_CAND_ACC_NAME_JS = (
    "el => { const t = s => (s || '').replace(/\\s+/g, ' ').trim();"
    " const ids = el.getAttribute('aria-labelledby');"
    " let out = '';"
    " if (ids) out = t(ids.split(/\\s+/).map(i => {"
    "   const n = document.getElementById(i); return n ? n.textContent : ''; })"
    "   .join(' '));"
    " if (!out && el.id) { const l = document.querySelector("
    "   'label[for=' + JSON.stringify(el.id) + ']'); if (l) out = t(l.textContent); }"
    " if (!out) { const l = el.closest('label'); if (l) out = t(l.textContent); }"
    " if (!out) out = t(el.getAttribute('title'));"
    " return out; }"
)


# Accessible-name approximation for an element with NO text of its own that holds a
# LABELLED child — an icon-only table cell / wrapper (run 20260818_091836 seg 4: the
# Net-to-Gross grid cell, whose pencil renders as a CSS glyph, so inner_text is '').
# Capture reads exactly this layer: browser-use builds a cell's ax_name from its
# CONTENTS, so compile stamps expect_text="Net to gross" for an element whose own name
# is empty, and the gate refused it on every replay. The empty-own-text condition is
# what keeps this narrow: an element that DOES render text has already been judged on
# that text by the inner_text reader, and a child icon must not overrule it.
_CAND_DESC_NAME_JS = (
    "el => { const t = s => (s || '').replace(/\\s+/g, ' ').trim();"
    " if (t(el.innerText)) return '';"
    " const kids = el.querySelectorAll('[aria-label],[title],[data-icon-name]');"
    " const out = [];"
    " for (let i = 0; i < kids.length && i < 12; i++) {"
    "   out.push(t(kids[i].getAttribute('aria-label')),"
    "            t(kids[i].getAttribute('title')),"
    "            t(kids[i].getAttribute('data-icon-name'))); }"
    " return out.filter(Boolean).join(' ').slice(0, 200); }"
)


# The haystack RAW_FIND_JS matches a find_by_text QUERY against: the element's own
# label-ish attributes and text, PLUS every descendant's icon/title/label hints. Replay
# reads the same surface back for a step whose expect_text is a query rather than a name
# (`expect_scattered`), so a click is re-verified by the predicate that made it.
_CAND_HAYSTACK_JS = (
    "el => { const t = s => (s || '').replace(/\\s+/g, ' ').trim();"
    " const bits = [el.getAttribute('title'), el.getAttribute('aria-label'),"
    "   el.getAttribute('name'), el.innerText];"
    " el.querySelectorAll('[data-icon-name],[title],[aria-label]').forEach(c => {"
    "   bits.push(c.getAttribute('data-icon-name'), c.getAttribute('title'),"
    "             c.getAttribute('aria-label')); });"
    " return bits.filter(Boolean).map(t).join(' '); }"
)


def _holds_tokens(text: str, expect: str) -> bool:
    """Every token of `expect` present SOMEWHERE in `text` — RAW_FIND_JS's own rule.
    Weaker than _names_value's consecutive run, and deliberately so: it is only used
    where the recorded name came from a scattered-token search in the first place. Each
    token must still be there, so a neighbouring row missing one of them is refused."""
    have = set(_query_tokens(text))
    want = _query_tokens(expect)
    return bool(want) and all(tok in have for tok in want)


async def _candidate_holds_tokens(loc: Any, expect: str) -> bool:
    """The scattered twin of _candidate_names_value. Unreadable nodes fail closed."""
    try:
        hay = str(await loc.evaluate(_CAND_HAYSTACK_JS) or "")
    except Exception:  # noqa: BLE001 - a candidate we cannot read is never acted on
        return False
    return _holds_tokens(hay, expect)


async def _candidate_names_value(loc: Any, expect: str) -> bool:
    """Does this resolved candidate visibly carry `expect` as its name? Checks rendered
    text first, aria-label second (icon-ish controls), then the ASSOCIATED name
    (aria-labelledby / label[for] / enclosing label / title) — the name layer record-time
    capture reads for toggles and checkboxes, whose own text is empty — and finally, for
    an element with no text at all, the name of the LABELLED CHILD it holds (an icon-only
    cell; see _CAND_DESC_NAME_JS). Every reader mirrors a layer capture's ax_name is built
    from: a name this gate cannot read back is a step that can never replay. Unreadable
    nodes fail closed — a value-anchored click must never act on an element it cannot
    verify."""
    for reader in ("inner_text", "aria", "assoc", "desc"):
        try:
            if reader == "inner_text":
                text = await loc.inner_text(timeout=1000)
            elif reader == "aria":
                text = (await loc.get_attribute("aria-label")) or ""
            elif reader == "assoc":
                text = str(await loc.evaluate(_CAND_ACC_NAME_JS) or "")
            else:
                text = str(await loc.evaluate(_CAND_DESC_NAME_JS) or "")
        except Exception:  # noqa: BLE001 - unreadable this way; try the next reader
            continue
        if _names_value(text, expect):
            return True
    return False


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

    A step stamped `expect_text` (its selectors carried an instantiated {{param}}: "click
    the element NAMED <value>") additionally requires every acted-on candidate to carry
    that value as its name (_names_value) — whatever selector found it. This is what
    keeps a value-swapped replay from clicking the wrong DATA ROW: substring collisions
    (FUNFOOD LIMITED vs FOOD LIMITED), a stale recorded href uniquely matching the OLD
    record's row, and the positional row anchor all resolve confidently to an element —
    the wrong one — and only the name check can tell.
    """
    sels = _step_selectors(step)
    if not sels:
        raise RuntimeError("step has no selector")
    expect = str(step.get("expect_text") or "")
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
        if expect:
            verify = (_candidate_holds_tokens if step.get("expect_scattered")
                      else _candidate_names_value)
            named = [n for n in visible if await verify(loc.nth(n), expect)]
            if not named:
                errors.append(f'{sel} -> {len(visible)} visible match(es), '
                              f'none named "{expect}"')
                continue
            visible = named
        if len(visible) == 1:
            candidate = loc.nth(visible[0])
        elif last:
            # Exhausted durable candidates; act on the first VISIBLE match (already
            # name-filtered when the step is value-anchored) but record it.
            logger.warning("ambiguous selector %r: %d visible matches; using the first",
                           sel, len(visible))
            candidate = loc.nth(visible[0])
        else:
            errors.append(f"{sel} -> {len(visible)} visible matches (ambiguous)")
            continue
        if sel.startswith("xpath=") and not await _xpath_matches_fingerprint(
                candidate, step.get("fingerprint")):
            errors.append(f"{sel} -> landed element no longer matches the recorded "
                          f"identity (positional drift)")
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
            winner = healed[2] or {}
            winner_text = str(winner.get("text") or "")
            expect_ok = (_holds_tokens(winner_text, expect)
                         if step.get("expect_scattered")
                         else _names_value(winner_text, expect))
            if expect and not expect_ok:
                # A fingerprint heal scores STRUCTURE, not the value — on a value-anchored
                # step a confident structural match to the wrong-named row is exactly the
                # wrong-business click this gate exists for.
                errors.append(f'healed match {str(winner.get("text"))[:40]!r} is not '
                              f'named "{expect}"')
            else:
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
                    if sel.startswith("xpath=") and not await _xpath_matches_fingerprint(
                            loc.first, step.get("fingerprint")):
                        continue
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


# Non-digits a page may add when it reformats a value it accepted ("£5,000.00" for "5000").
_NUM_NOISE_RE = re.compile(r"[^\d.\-]")


def _as_number(s: str) -> float | None:
    """`s` as a float once currency/grouping noise is stripped, else None."""
    stripped = _NUM_NOISE_RE.sub("", s or "")
    if not stripped or stripped in ("-", ".", "-."):
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def value_took(typed: str, actual: str) -> bool:
    """Did the field accept `typed`? Tolerant of the page reformatting what we wrote
    (amounts regrouped, an autocomplete completing it) but NOT of it keeping some other
    value: two numbers must be EQUAL, which is what catches a cell that reverted.

    Shared with the live agent tool (agent_tools.input) so an authored run and its replay
    judge "did this fill take?" by the same rule."""
    want, got = (typed or "").strip(), (actual or "").strip()
    if want == got:
        return True
    if not want:
        return not got
    want_n, got_n = _as_number(want), _as_number(got)
    if want_n is not None and got_n is not None:
        return want_n == got_n
    if want_n is not None or got_n is not None:
        return False  # one side numeric, the other not — a real mismatch
    return want.lower() in got.lower()  # autocompleted around what we wrote


async def _keyboard_refill(loc: Locator, value: str, timeout_ms: int) -> None:
    """Clear + retype the field with real keystrokes, for fields `fill()` can't move.

    React re-renders its own state over a value the framework never saw change; a
    select-all + Delete + character-by-character type is indistinguishable from a user
    and survives that (same reasoning as agent_tools' stubborn-field notes)."""
    await loc.click(timeout=timeout_ms)
    await loc.press("ControlOrMeta+a", timeout=timeout_ms)
    await loc.press("Delete", timeout=timeout_ms)
    if value:
        await loc.press_sequentially(value, delay=20, timeout=timeout_ms)


async def _current_value(loc: Locator) -> str | None:
    """The field's value as the page holds it now (None when it can't be read)."""
    try:
        return await loc.input_value(timeout=2000)
    except Exception:  # noqa: BLE001 - contenteditable has no value; fall back to its text
        try:
            return await loc.inner_text(timeout=2000)
        except Exception:  # noqa: BLE001 - unreadable: skip verification, don't fail the fill
            return None


async def _fill_with_retry(page: Page, step: dict[str, Any], timeout_ms: int) -> tuple[str, dict[str, Any] | None]:
    """Resolve + fill, re-resolving after a settle if the field detaches mid-render.
    Returns (selector_label, healed_winner_or_None)."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            loc, sel, healed = await _resolve(page, step, timeout_ms, require_editable=True)
            value = step.get("value", "")
            if step.get("clear", True):
                await loc.fill("", timeout=timeout_ms)
            await loc.fill(value, timeout=timeout_ms)
            # Verify and repair: a replayed fill that silently didn't take corrupts every
            # step after it (a wrong amount saves just as happily as a right one).
            actual = await _current_value(loc)
            if actual is not None and not value_took(value, actual):
                logger.warning("fill %r left the field reading %r; clearing and retyping "
                               "with keystrokes", value, actual)
                await _keyboard_refill(loc, value, timeout_ms)
                actual = await _current_value(loc)
                if actual is not None and not value_took(value, actual):
                    raise RuntimeError(
                        f"fill did not take: field still reads {actual!r}, expected {value!r}")
            return sel, healed
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_ATTEMPTS - 1 and _is_transient(exc):
                logger.info("fill step transient (%s); settling %dms then re-resolving (attempt %d/%d)",
                            exc, _SETTLE_MS * 2, attempt + 2, _MAX_ATTEMPTS)
                await page.wait_for_timeout(_SETTLE_MS * 2)
                continue
            raise
    raise RuntimeError("unreachable")  # loop either returns or raises


async def _select_with_retry(page: Page, step: dict[str, Any], timeout_ms: int
                             ) -> tuple[str, dict[str, Any] | None]:
    """Resolve a native <select> and pick the recorded option BY LABEL, falling back to
    the option's value attribute when the visible label drifted. select_option fires the
    input/change events the page's own scripts listen for — the reason a replayed pick
    actually changes what the form submits. Returns (selector_label, healed_winner)."""
    loc, sel, healed = await _resolve(page, step, timeout_ms)
    option = str(step.get("value", ""))
    try:
        await loc.select_option(label=option, timeout=timeout_ms)
    except Exception:  # noqa: BLE001 - label text changed; the value attr is the fallback
        await loc.select_option(value=option, timeout=timeout_ms)
    return sel, healed


# Scroll every scrollable CONTAINER (and the document) by a fraction of its own height.
# A side panel / dialog list owns its scroll box, and it usually renders only the rows
# near its scroll position — so a row further down is not in the DOM at all until that
# box moves. Wheeling at the viewport centre moves whatever sits under the cursor (the
# page behind the panel), which is why the employee picker's later rows were
# unreachable (run 20260814_105247, seg 6). Returns how many boxes actually moved, so a
# caller can tell "nothing left to scroll" from "scrolled, try the search again".
# Placeholder: %s = fraction of each container's height to advance.
SCROLL_CONTAINERS_JS = r"""
(function () {
  try {
    var FRAC = %s;
    var moved = 0;
__WALK__
    var boxes = [];
    walkAll(function (e) {
      var st;
      try { st = getComputedStyle(e); } catch (err) { return; }
      var oy = st.overflowY;
      if ((oy === 'auto' || oy === 'scroll') && e.scrollHeight > e.clientHeight + 4) {
        boxes.push(e);
      }
    });
    boxes.forEach(function (e) {
      var before = e.scrollTop;
      e.scrollTop = before + Math.max(40, e.clientHeight * FRAC);
      if (e.scrollTop !== before) moved++;
    });
    var doc = document.scrollingElement || document.documentElement;
    if (doc && doc.scrollHeight > doc.clientHeight + 4) {
      var b4 = doc.scrollTop;
      doc.scrollTop = b4 + Math.max(40, doc.clientHeight * FRAC);
      if (doc.scrollTop !== b4) moved++;
    }
    return moved;
  } catch (e) { return 0; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)


async def _scroll_containers(page: Page, fraction: float = 0.8) -> int:
    """Advance every scrollable container; returns how many moved (0 = nothing left)."""
    try:
        moved = await page.evaluate(SCROLL_CONTAINERS_JS % json.dumps(float(fraction)))
        await page.wait_for_timeout(_SETTLE_MS)
        return int(moved or 0)
    except Exception as exc:  # noqa: BLE001 - scrolling is best-effort
        logger.debug("container scroll failed: %s", exc)
        return 0


# Reset every scrollable container AND the page to the top. The hunt companion to
# SCROLL_CONTAINERS_JS: a down-only sweep from an arbitrary position can never reach a
# target ABOVE it, and a failed sweep used to leave the page parked at the very bottom
# (run 20260817_133135 — the user had to scroll back up by hand). Sweeps now START here,
# and failed hunts END here.
SCROLL_TOPS_JS = r"""
(function () {
  try {
    var moved = 0;
__WALK__
    walkAll(function (e) {
      var st;
      try { st = getComputedStyle(e); } catch (err) { return; }
      var oy = st.overflowY;
      if ((oy === 'auto' || oy === 'scroll') && e.scrollTop > 0) {
        e.scrollTop = 0; moved++;
      }
    });
    var doc = document.scrollingElement || document.documentElement;
    if (doc && doc.scrollTop > 0) { doc.scrollTop = 0; moved++; }
    return moved;
  } catch (e) { return 0; }
})()
""".replace("__WALK__", _COMPOSED_WALK_JS)


async def _scroll_tops(page: Page) -> int:
    """Scroll page + containers to the top; returns how many scrollers moved."""
    try:
        moved = await page.evaluate(SCROLL_TOPS_JS)
        await page.wait_for_timeout(_SETTLE_MS)
        return int(moved or 0)
    except Exception as exc:  # noqa: BLE001 - scrolling is best-effort
        logger.debug("scroll-to-top failed: %s", exc)
        return 0


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


async def _resolve_with_scroll(page: Page, step: dict[str, Any], timeout_ms: int,
                               rounds: int = _FIND_CLICK_ROUNDS) -> tuple[Any, str]:
    """_resolve with find_click's wheel-scroll rounds: a virtualized pane renders a
    row's DOM only once the viewport nears it, so a miss wheel-scrolls down and
    retries. Returns (locator, selector_label); raises after the last round."""
    last: Exception | None = None
    for round_no in range(rounds + 1):
        try:
            loc, sel, _healed = await _resolve(page, step, timeout_ms)
            return loc, sel
        except Exception as exc:  # noqa: BLE001 - miss; scroll and retry
            last = exc
            if round_no == 0:
                # Sweep from the TOP: a down-only hunt from mid-page can never reach
                # a target rendered above it (run 20260817_133135).
                await _scroll_tops(page)
            elif round_no < rounds:
                await _wheel_scroll(page, 0.6)
    await _scroll_tops(page)   # a failed hunt must not leave the page at the bottom
    raise RuntimeError(f"indexed click: {step.get('selectors')} not found after "
                       f"{rounds} scroll rounds") from last


async def _find_click(page: Page, text: str, verify_name: bool = False) -> str:
    """Replay a find_by_text click SEMANTICALLY: run the same in-page algorithm the tool
    used at record time (RAW_FIND_JS — token match, visible-first, scrollIntoView, direct
    handler click), wheel-scrolling between rounds when nothing matches yet. Returns the
    clicked element's reported name; raises when no round finds a match.

    `verify_name` (steps whose text is an instantiated {{param}}): probe first with the
    click disabled and require the top candidate to be NAMED the text (_names_value)
    before the real click fires — a hint-ranked near-miss must fail the step, never
    click a wrong-named element. Plain find_click steps keep today's behavior (icon
    hints legitimately click elements not named by the query)."""
    import json as _json

    tokens = _query_tokens(text)
    if not tokens:
        raise RuntimeError(f"find_click: no searchable text in {text!r}")
    expr = RAW_FIND_JS % (_json.dumps(tokens), "true")
    probe_expr = RAW_FIND_JS % (_json.dumps(tokens), "false")
    for round_no in range(_FIND_CLICK_ROUNDS + 1):
        raw = None
        try:
            if verify_name:
                probe = await page.evaluate(probe_expr)
                if probe and probe.get("count"):
                    if not _names_value(str(probe.get("name") or ""), str(text)):
                        raise RuntimeError(
                            f"find_click: best match {str(probe.get('name'))[:40]!r} is "
                            f"not named {text!r}; refusing a wrong-named click")
                    raw = await page.evaluate(expr)
            else:
                raw = await page.evaluate(expr)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - page navigating; settle and retry
            logger.debug("find_click eval failed (%s); settling", exc)
            raw = None
            await page.wait_for_timeout(_SETTLE_MS)
        if raw and raw.get("clicked"):
            name = str(raw.get("name") or text)
            logger.info("🔎 find_click(%r): clicked %r (round %d)", text, name, round_no)
            return name
        if round_no == 0:
            # Sweep from the TOP: a target above the current scroll position is
            # otherwise unreachable — virtualized rows above the window are pruned
            # from the DOM just like the ones below it (run 20260817_133135).
            await _scroll_tops(page)
        elif round_no < _FIND_CLICK_ROUNDS:
            # Panels and dialog lists own their scroll boxes and render only the rows
            # near their scroll position; move those FIRST, then the page under the
            # cursor for everything else.
            if not await _scroll_containers(page, 0.8):
                await _wheel_scroll(page, 0.6)
    await _scroll_tops(page)   # a failed hunt must not leave the page at the bottom
    raise RuntimeError(f"find_click: no clickable match for {text!r} "
                       f"after {_FIND_CLICK_ROUNDS} scroll rounds")


async def _extract_value(page: Page, step: dict[str, Any], timeout_ms: int
                         ) -> tuple[str, str, dict[str, Any] | None]:
    """Read the CURRENT text of an extract step's target: (value, used_selector, healed).

    Selector-anchored steps resolve through the full `_resolve` ladder (unique match,
    fingerprint heal), so extract steps self-heal exactly like clicks; when every selector
    fails and the step carries its recorded `query`, the value is re-found semantically
    with the same in-page algorithms the authoring tool used (RAW_FIND_JS over controls,
    then RAW_TEXT_FIND_JS over static text, click disabled) — that is also the whole path
    for query-only steps. An EMPTY read raises: a valueless extraction means the page no
    longer shows the data where the recording found it, and the segment must fail into the
    agent-recovery path instead of reporting a hollow pass.
    """
    value, used, healed = "", "", None
    if _step_selectors(step):
        try:
            loc, used, healed = await _resolve(page, step, timeout_ms)

            # A form control's value first: a <select>'s inner_text is its option labels
            # concatenated (never the chosen value), so the text readers below would
            # report the blob. Non-controls return '' here and fall through unchanged.
            async def _control_value() -> str:
                return await loc.evaluate(
                    "el => el.tagName === 'SELECT'"
                    " ? ((el.selectedOptions[0] && el.selectedOptions[0].text)"
                    "    || el.value || '')"
                    " : (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')"
                    " ? (el.value || '') : ''")

            # inner_text is the honest read (what a user sees); text_content rescues
            # 0-size/hover-revealed targets; input_value covers form fields.
            for reader in (_control_value, loc.inner_text, loc.text_content,
                           loc.input_value):
                try:
                    value = normalize_block_text(await reader() or "")
                except Exception:  # noqa: BLE001 - e.g. input_value on a non-input
                    value = ""
                if value:
                    break
        except Exception as exc:  # noqa: BLE001 - fall to the semantic re-find if possible
            if not step.get("query"):
                raise
            logger.info("extract %r: selectors failed (%s); re-finding by query",
                        step.get("label"), str(exc)[:120])
    if not value and step.get("query"):
        import json as _json

        tokens = _query_tokens(step["query"])
        if tokens:
            raw = await page.evaluate(RAW_FIND_JS % (_json.dumps(tokens), "false"))
            if not (raw and not raw.get("error") and raw.get("count")):
                # The value lives in plain text, not in a control (the authoring path
                # that captured it) — re-find it with the same static-text algorithm.
                raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(tokens))
            if raw and not raw.get("error") and raw.get("count"):
                lines = [str(ln) for ln in (raw.get("lines") or []) if str(ln).strip()]
                value = ("\n".join(lines) if len(lines) >= 2
                         else normalize_block_text(raw.get("name") or ""))
                used = used or f"find:{step['query']}"
    if not value:
        raise RuntimeError(f"extract {str(step.get('label') or 'value')!r}: no visible "
                           f"text at the recorded location")
    # Cap sized for BLOCK captures (one extract on the card showing several facts is the
    # preferred authoring shape — the consuming agent parses the blob).
    return value[:1000], used, healed


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


async def _upload_with_retry(page: Page, step: dict[str, Any], timeout_ms: int
                             ) -> tuple[str, dict[str, Any] | None]:
    """Attach the step's file (files.UPLOADS_DIR/<value>) to its upload control.
    Returns (used_selector, healed).

    The file must exist NON-EMPTY or this raises before touching the page: a ghost
    upload is worse than a failed segment — observed live, a nonexistent path attached
    "successfully", then Save made the page READ it and Chrome killed the renderer
    (RESULT_CODE_KILLED_BAD_MESSAGE). The CONTROL runs the normal _resolve ladder, then
    walks to the real <input type=file>: the resolved element itself, a descendant, or
    the page's file input — set_input_files works on display:none inputs, which is what
    keeps the native dialog closed. An unresolvable recorded control falls through to
    the page-wide input rather than failing: the recorded element was only ever a
    pointer to the true target.
    """
    from automation.pipeline.files import UPLOADS_DIR, find_file

    name = str(step.get("value") or "")
    path = find_file(name)
    if path is None:
        raise RuntimeError(f"upload file {name!r} not found (or empty) in "
                           f"{UPLOADS_DIR}/ — place it there")

    loc, sel, healed = None, "", None
    try:
        loc, sel, healed = await _resolve(page, step, timeout_ms)
    except Exception as exc:  # noqa: BLE001 - the file input is the true target
        logger.info("upload control did not resolve (%s); falling back to the page's "
                    "file input", str(exc)[:120])
    target = None
    if loc is not None:
        try:
            if await loc.evaluate("el => el.tagName === 'INPUT' && el.type === 'file'"):
                target = loc
        except Exception:  # noqa: BLE001 - fall through the ladder
            target = None
        if target is None:
            inner = loc.locator("input[type=file]")
            if await inner.count() > 0:
                target = inner.first
    if target is None:
        page_wide = page.locator("input[type=file]")
        if await page_wide.count() > 0:
            target = page_wide.first
            sel = sel or "input[type=file]"
    if target is None:
        raise RuntimeError("no <input type=file> found at or near the recorded control")
    await target.set_input_files(str(path.resolve()))
    return sel or "input[type=file]", healed


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
    extracted: dict[str, str] = {}
    for idx, step in enumerate(steps):
        try:
            action = step["action"]
            if action == "goto":
                await page.goto(step["url"], wait_until="domcontentloaded", timeout=timeout_ms)
            elif action == "click":
                # Tier-0 parity for compiled repeats: N clicks with the recorded wait
                # between (the tier-1 verb adds the readiness poll on top).
                for rep in range(int(step.get("count", 1))):
                    sel, healed = await _click_with_flyout_recovery(page, steps, idx,
                                                                    timeout_ms)
                    entry = {"step": idx, "action": action, "used": sel}
                    if healed:
                        entry["healed"] = healed
                    log.append(entry)
                    await page.wait_for_timeout(_SETTLE_MS)
                    if rep < int(step.get("count", 1)) - 1 and step.get("repeat_wait_s"):
                        await page.wait_for_timeout(
                            int(min(float(step["repeat_wait_s"]), 3.0) * 1000))
            elif action == "click_indexed":
                template = str(step.get("selector_template") or "")
                for n in range(int(step.get("start", 0)),
                               int(step.get("start", 0)) + int(step.get("count", 0))):
                    isel = template.replace("{n}", str(n))
                    loc, sel = await _resolve_with_scroll(
                        page, {"selectors": [isel]}, 2500)
                    try:
                        await loc.click(timeout=5000)
                    except Exception:  # noqa: BLE001 - pointer interception; force like clicks do
                        await loc.click(timeout=5000, force=True)
                    log.append({"step": idx, "action": "click", "used": sel})
                    await page.wait_for_timeout(_SETTLE_MS)
            elif action == "fill":
                sel, healed = await _fill_with_retry(page, step, timeout_ms)
                entry = {"step": idx, "action": action, "used": sel}
                if healed:
                    entry["healed"] = healed
                log.append(entry)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "select":
                sel, healed = await _select_with_retry(page, step, timeout_ms)
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
            elif action == "scroll_panels":
                await _scroll_containers(page, float(step.get("pages", 0.8)))
            elif action == "find_click":
                name = await _find_click(page, step.get("text", ""),
                                         verify_name=bool(step.get("verify_name")))
                log.append({"step": idx, "action": action, "used": f"find_click:{name}"})
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "extract":
                value, used, healed = await _extract_value(page, step, timeout_ms)
                merge_extract(extracted, str(step.get("label") or "value"), value)
                entry = {"step": idx, "action": action, "used": used,
                         "value": value[:200]}
                if healed:
                    entry["healed"] = healed
                log.append(entry)
            elif action == "upload":
                sel, healed = await _upload_with_retry(page, step, timeout_ms)
                entry = {"step": idx, "action": action, "used": sel}
                if healed:
                    entry["healed"] = healed
                log.append(entry)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "wait":
                await page.wait_for_timeout(int(step.get("seconds", 0) * 1000))
            executed += 1
        except Exception as exc:  # noqa: BLE001 - report where the script broke (app changed?)
            logger.exception("script execution broke at step %d: %s", idx, exc)
            return {"executed": executed, "failed_at": idx,
                    "error": f"{type(exc).__name__}: {exc}", "log": log,
                    "extracted": extracted}
    return {"executed": executed, "failed_at": None, "error": None, "log": log,
            "extracted": extracted}


# Ceiling for a step's candidate list after heal promotions, so repeated healings of a churny
# element can't grow it without bound (promoted candidates prepend; the oldest fallbacks drop).
_MAX_SELECTORS = 8


def merge_promoted_selectors(new_sels: list[str], old_sels: list[str],
                             cap: int) -> list[str]:
    """Healed selectors prepend — but NEVER above a leading recorded `xpath=`. The
    exact location stays rank 0 (2026-08-13 directive); a heal is an added fallback,
    not a demotion of the anchor the recording is built on."""
    lead = [s for s in old_sels[:1] if s.startswith("xpath=")]
    rest = old_sels[len(lead):]
    seen: set[str] = set()
    return [s for s in lead + new_sels + rest
            if not (s in seen or seen.add(s))][:cap]


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
        merged = merge_promoted_selectors(new_sels, old_sels, _MAX_SELECTORS)
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
