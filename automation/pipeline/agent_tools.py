"""Custom agent tools the prompts rely on, registered on a browser-use ``Tools`` registry.

The system/expander prompts (see prompts.py) instruct the agent to call custom actions that are
NOT part of browser-use's built-in set:

  * skip_step(reason)          — escape hatch: abandon the current objective, keep going.
  * fail_and_stop(reason)      — escape hatch: terminate the whole run as a failure.
  * capped_scroll(down, pages) — discovery scroll capped at 0.5 pages per call.
  * find_by_text(text)         — find interactive elements by label in a FRESH snapshot,
                                 returning their current click indexes (optionally clicking).
  * extract_data(text, label)  — capture a piece of on-page data by a REPLAYABLE locator:
                                 the value is returned to the agent now AND recorded so a
                                 compiled `extract` step re-reads it fresh on every replay
                                 (aux-tab subtasks fetching live data depend on this).
  * list_actions(near_text)    — list clickable controls near a heading/row, decoding the
                                 nameless icon buttons (Fluent/SVG) browser-use renders blank.
  * verify_save_registered()   — ground truth for saves: did a create-write actually hit the
                                 server this run? (probe wired by the Runner per run).
  * detect_layout_issues()     — heuristic layout/overflow scan of the current page.
  * run_accessibility_scan()   — WCAG scan via vendored axe-core (assets/axe.min.js).

`build_tools()` returns a `Tools` instance with these registered (plus every built-in EXCEPT
`evaluate` — see below, since `Tools()` starts from the default registry). The Runner passes
it to `Agent(tools=...)`.

The built-in `input` action is REPLACED (same name, same params) by a variant that presses
Enter after typing: this app's search/filter boxes only apply on Enter, and agents regularly
typed a query without submitting it. Dropdown/combobox filter inputs (react-select) are
exempt — Enter there selects whatever option is focused (see the SEARCH BOXES prompt rule).
The result's `metadata.auto_enter` flag says whether Enter was pressed (restored into saved
recordings by runner.restore_result_metadata), and compile_recording mirrors it with a
`press` step so replays match the live run.

Two behaviours here exist to keep authored runs COMPILABLE into replay scripts
(script_compile.py only translates click/input/navigate/wait/send_keys):
  * `evaluate` (built-in JS execution) is excluded from the registry. The agent used it to
    force React field values when normal `input` looked like it "didn't register" (banned by
    the NO JS FORM FILL prompt rule anyway); those JS writes are unrecordable, so a recording
    that relied on them compiled to a script missing those fields. Removing the action forces
    the agent to fill via `input`, which the compiler and replay both understand.
  * `find_by_text(click_first=True)` records the element it clicked into the ActionResult's
    `metadata` (as a DOMInteractedElement dict, the same shape a built-in click records under
    `state.interacted_element`). A custom action has no `index`, so browser-use never captures
    its target — without this, every navigation click made via find_by_text was dropped from
    the compiled script, breaking replay. script_compile reads this metadata back.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from browser_use import Tools
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.events import (ClickElementEvent, SelectDropdownOptionEvent,
                                        SendKeysEvent, TypeTextEvent)
from browser_use.dom.views import DOMInteractedElement
from browser_use.tools.views import InputTextAction, SelectDropdownOptionAction

from automation.pipeline.script_compile import (
    RAW_FIND_JS as _RAW_FIND_JS,
    RAW_TEXT_FIND_JS as _RAW_TEXT_FIND_JS,
    REVEAL_CSS_JS as _REVEAL_CSS_JS,
    _RS_FILTER_ID,
    value_took as _value_took,
)

logger = logging.getLogger("framework.tools")

# Vendored axe-core, loaded once and injected into the page on demand.
_AXE_PATH = Path(__file__).resolve().parent.parent / "assets" / "axe.min.js"

# Live ground-truth probe for verify_save_registered. The Runner sets a closure per run
# (returning the first successful create-write record, or None) and clears it afterwards;
# built here as a holder so this module never has to import the runner.
_SAVE_PROBE: Any = None


def set_save_probe(fn) -> None:
    """Install the per-run save probe (Runner only). `fn() -> dict | None`."""
    global _SAVE_PROBE
    _SAVE_PROBE = fn


def clear_save_probe() -> None:
    global _SAVE_PROBE
    _SAVE_PROBE = None


# Icon-font class tokens that carry an icon's meaning (Fluent `ms-Icon--Mail`, FontAwesome
# `fa-envelope`, generic `icon-send`). The captured group is the semantic part.
_ICON_CLASS_RE = re.compile(r"(?:ms-Icon--|fa-|icon-|glyphicon-)([A-Za-z][A-Za-z0-9]+)")


def _descendant_icon_hints(node: Any, depth: int = 4) -> str:
    """Semantic hints for an otherwise-nameless icon control, harvested from its DESCENDANTS.

    browser-use surfaces a node's OWN title/aria-label but never its children's, so a Fluent
    icon button (`<button class="ms-Button--icon"><i data-icon-name="Send"/></button>`) and
    its peers all reach the agent as bare `<button/>` — indistinguishable. We walk the subtree
    and collect the child hints that name the icon: `data-icon-name`, child `title`/
    `aria-label`, SVG `<title>` text, `<use href="#icon-...">` fragments, and icon-font class
    tokens. Returns a space-joined, de-duplicated string ("" when nothing was found)."""
    found: list[str] = []

    def walk(n: Any, d: int) -> None:
        if n is None or d < 0:
            return
        attrs = getattr(n, "attributes", None) or {}
        for attr in ("data-icon-name", "title", "aria-label", "alt"):
            val = attrs.get(attr)
            if val:
                found.append(str(val))
        href = attrs.get("href") or attrs.get("xlink:href") or ""
        if "#" in str(href):
            found.append(str(href).rsplit("#", 1)[1])
        for token in _ICON_CLASS_RE.findall(attrs.get("class") or ""):
            found.append(token)
        # SVG <title> text is a child text node under a <title> element.
        if (getattr(n, "node_name", "") or "").lower() == "title":
            txt = getattr(n, "node_value", "") or ""
            if txt.strip():
                found.append(txt)
        for child in (getattr(n, "children", None) or []):
            walk(child, d - 1)

    try:
        for child in (getattr(node, "children", None) or []):
            walk(child, depth)
    except Exception:  # noqa: BLE001 - a naming aid must never crash a lookup
        return ""
    seen: set[str] = set()
    uniq: list[str] = []
    for hint in found:
        h = " ".join(str(hint).split())
        if h and h.lower() not in seen:
            seen.add(h.lower())
            uniq.append(h)
    return " ".join(uniq)


async def _eval_js(browser_session: BrowserSession | None, expression: str, *, await_promise: bool = False):
    """Run `expression` in the page over CDP and return its by-value result (or raise).

    Mirrors how browser-use's own `evaluate` action talks to the page.
    """
    if browser_session is None:
        raise RuntimeError("BrowserSession not injected. Ensure the tool function signature includes the 'browser_session: BrowserSession' type hint.")
    cdp_session = await browser_session.get_or_create_cdp_session()
    result = await cdp_session.cdp_client.send.Runtime.evaluate(
        params={"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
        session_id=cdp_session.session_id,
    )
    if result.get("exceptionDetails"):
        details = result["exceptionDetails"]
        msg = details.get("exception", {}).get("description") or details.get("text") or str(details)
        raise RuntimeError(msg)
    return result.get("result", {}).get("value")


# Raw-DOM locate (+ optional click) for controls that browser-use's interactive snapshot
# OMITS. Some Fluent widgets render functional buttons at 0x0 inside a virtualized
# ScrollablePane (e.g. the "Send NPS survey request" icon): they have a real title and a live
# click handler, but zero layout, so they never enter the selector_map and find_by_text's
# normal path can't see them. This queries the live DOM directly (viewport/size-independent),
# matches ALL tokens across text + title/aria-label/name + descendant data-icon-name/title,
# and clicks the best match via the element's own handler (works on a 0x0 node).
# The raw-DOM find+click algorithm lives in script_compile.RAW_FIND_JS and is SHARED with
# the replay engine: a `find_click` step replays exactly what this tool did at record time,
# so hover-revealed/0-size controls behave identically at author and replay time.


def _norm_phrase(s: str) -> str:
    """find_by_text's token grammar collapsed back to a phrase — the one normalization for
    comparing a clicked element's name against the query that found it (and the same
    normalization RAW_FIND_JS ranks candidates with)."""
    return " ".join(t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t)


def _matching_nodes(state: Any, tokens: list[str]) -> list[tuple[int, Any, str]]:
    """The token match over the interactive snapshot: (index, node, label) for every
    element whose visible text / attributes / descendant icon hints contain ALL tokens.
    Shared by find_by_text and extract_data so both tools locate elements identically."""
    matches: list[tuple[int, Any, str]] = []
    for idx, node in sorted(state.dom_state.selector_map.items()):
        try:
            label = " ".join(node.get_all_children_text(max_depth=5).split())[:300]
            haystack = label.lower()
            for attr in ("aria-label", "title", "placeholder", "value", "alt", "name", "id"):
                val = (node.attributes or {}).get(attr) or ""
                if val:
                    haystack += " " + val.lower()
                    if not label and attr not in ("id", "name"):
                        label = val
            # Icon buttons carry their meaning in a child glyph browser-use drops; fold
            # the child hints in so a nameless <button/> becomes matchable/visible.
            hints = _descendant_icon_hints(node)
            if hints:
                haystack += " " + hints.lower()
                if not label:
                    label = hints
            if all(t in haystack for t in tokens):
                matches.append((idx, node, label))
        except Exception:  # noqa: BLE001 - skip malformed nodes, keep scanning
            continue
    return matches


def _hidden_click_receipt(query: str, clicked_name: str) -> str:
    """Receipt for a raw-DOM hidden-path click. It must be UNMISTAKABLY "the click already
    happened": a model that reads it as a find-result clicks a second time — observed live:
    the follow-up click closed the panel the first had just opened, then the agent hunted
    the vanished icon for 18 steps. And when the clicked NAME is neither the query nor a
    prefix-extension of it, the generic "re-call once" advice looped the agent into
    re-clicking the same wrong control (observed: 'Reviews' -> 'Add reviews', three times),
    so that case warns and forbids the re-call instead."""
    base = (f"find_by_text('{query}'): ✅ ALREADY CLICKED '{clicked_name}' for you "
            f"(a 0-size/hover-revealed control outside the interactive snapshot, clicked "
            f"via its own handler). Do NOT click it again — a second click can close what "
            f"the first just opened. ")
    name, phrase = _norm_phrase(clicked_name), _norm_phrase(query)
    if name == phrase or name.startswith(phrase + " "):
        return base + (f"NEXT: wait ~2 seconds, then check whether the expected "
                       f"panel/content appeared. Only if it truly did not, re-call "
                       f"find_by_text('{query}', click_first=true) once.")
    return base + (f"⚠ NAME MISMATCH: '{clicked_name}' is not '{query}' — this may be the "
                   f"WRONG control (nothing better-named exists in the DOM). NEXT: check "
                   f"what changed on the page. If it is NOT what you wanted, close/undo it "
                   f"and do NOT re-call find_by_text('{query}') — it would click this same "
                   f"control again; reach your target differently (scroll to the section, "
                   f"or use a more specific label).")


# Page notifications (toasts / message bars) are how the app reports the OUTCOME of an action
# — "saved", "validation failed", "permission denied", a 500. They are transient: they fade
# in a few seconds, so by the time the agent finishes its LLM step and looks, they are gone.
# This installs a MutationObserver ONCE that buffers every notification as it appears, then
# returns (and clears) the ones seen since the last read. Pattern-based, not tied to any one
# app's markup: ARIA alert/status roles plus the common toast/message-bar class conventions.
_NOTIF_JS = r"""
(function () {
  try {
    if (!window.__ao_notifs) window.__ao_notifs = [];
    if (!window.__ao_notif_obs) {
      var RX = /(toast|notification|message-?bar|snackbar|growl|flash|banner|alert)/i;
      var isNotif = function (n) {
        if (!n || n.nodeType !== 1) return false;
        var role = (n.getAttribute && n.getAttribute('role')) || '';
        if (role === 'alert' || role === 'status') return true;
        var sig = ((n.className && n.className.toString ? n.className.toString() : '') + ' ' +
                   (n.id || '')).toLowerCase();
        return RX.test(sig);
      };
      var record = function (n) {
        var t = ((n.innerText || n.textContent || '').trim()).replace(/\s+/g, ' ');
        if (!t || t.length > 300) return;
        var last = window.__ao_notifs[window.__ao_notifs.length - 1];
        if (!last || last !== t) window.__ao_notifs.push(t);
      };
      var scan = function (n) {
        if (isNotif(n)) record(n);
        else if (n.querySelectorAll)
          n.querySelectorAll('[role=alert],[role=status]').forEach(record);
      };
      window.__ao_notif_obs = new MutationObserver(function (muts) {
        muts.forEach(function (m) {
          (m.addedNodes || []).forEach(scan);
        });
      });
      window.__ao_notif_obs.observe(document.documentElement, { childList: true, subtree: true });
    }
    var out = window.__ao_notifs.slice();
    window.__ao_notifs = [];
    return out;
  } catch (e) { return []; }
})()
"""


async def read_new_notifications(browser_session: BrowserSession | None) -> list[str]:
    """Install the page-notification observer (idempotent) and return the toast/message-bar
    texts seen since the last call. Best-effort: returns [] on any failure so it can never
    break a step. Called once per agent step by the Runner to surface transient notifications
    the agent would otherwise miss."""
    if browser_session is None:
        return []
    try:
        result = await _eval_js(browser_session, _NOTIF_JS)
    except Exception as exc:  # noqa: BLE001 - surfacing notices must never crash a run
        logger.debug("read_new_notifications failed: %s", exc)
        return []
    if not isinstance(result, list):
        return []
    # Clean + de-dup within the batch, order preserved. Icon fonts (Fluent, FontAwesome)
    # render glyphs as Unicode private-use codepoints (U+E000..U+F8FF) inside the toast
    # text; strip them so the agent reads words. An icon-only notification collapses to "".
    seen: set[str] = set()
    out: list[str] = []
    for item in result:
        cleaned = "".join(c for c in str(item) if not ("\ue000" <= c <= "\uf8ff"))
        t = " ".join(cleaned.split())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


async def ensure_reveal_css(browser_session: BrowserSession | None) -> None:
    """Install the reveal stylesheet (script_compile.REVEAL_CSS_JS, idempotent) into the
    agent's CURRENT document. login.py's context init script covers the main context from
    birth; this per-step pass heals what it can't reach — a tab browser-use creates via CDP
    outside that context, or a document whose <head> was rebuilt. Best-effort: never raises.
    The caller gates on Config.reveal_hidden_controls."""
    if browser_session is None:
        return
    try:
        await _eval_js(browser_session, _REVEAL_CSS_JS)
    except Exception as exc:  # noqa: BLE001 - styling must never break a step
        logger.debug("ensure_reveal_css skipped: %s", exc)


# --- Layout heuristic (runs entirely in the page) --------------------------------------------
# Flags the layout bugs a QA pass cares about: the page scrolling sideways, visible elements
# spilling past the right edge of the viewport, and clickable controls rendered at (near) zero
# size. Returns a compact JSON summary so the agent's context stays small.
_LAYOUT_JS = r"""
(function () {
  try {
    var issues = [];
    var vw = window.innerWidth, vh = window.innerHeight;
    var docW = document.documentElement.scrollWidth;
    if (docW > vw + 2) {
      issues.push({type: 'horizontal-overflow',
        detail: 'page scrolls sideways: scrollWidth ' + docW + ' > viewport ' + vw});
    }
    function visible(el, r) {
      var s = getComputedStyle(el);
      if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
      return r.width > 0 || r.height > 0;
    }
    var all = document.body ? document.body.querySelectorAll('*') : [];
    var overflow = [], tiny = [], seenO = 0;
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      var r = el.getBoundingClientRect();
      if (!visible(el, r)) continue;
      // Element spilling past the right/left edge of the viewport.
      if ((r.right > vw + 2 || r.left < -2) && r.width < vw && r.width > 4) {
        if (overflow.length < 5) overflow.push(_desc(el, r));
        seenO++;
      }
      // Interactive control rendered at (near) zero size — a click target you can't hit.
      var tag = el.tagName.toLowerCase();
      var clickable = tag === 'button' || tag === 'a' ||
        (tag === 'input' && el.type !== 'hidden') || el.getAttribute('role') === 'button';
      if (clickable && (r.width < 2 || r.height < 2) && getComputedStyle(el).display !== 'none') {
        if (tiny.length < 5) tiny.push(_desc(el, r));
      }
    }
    if (seenO) issues.push({type: 'element-overflow',
      detail: seenO + ' element(s) extend past the viewport edge', examples: overflow});
    if (tiny.length) issues.push({type: 'zero-size-control',
      detail: tiny.length + '+ clickable element(s) rendered at ~0 size', examples: tiny});
    return JSON.stringify({issues: issues});
    function _desc(el, r) {
      var id = el.id ? '#' + el.id : '';
      var cls = (typeof el.className === 'string' && el.className)
        ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
      return el.tagName.toLowerCase() + id + cls +
        ' @[' + Math.round(r.left) + ',' + Math.round(r.top) + ' ' +
        Math.round(r.width) + 'x' + Math.round(r.height) + ']';
    }
  } catch (e) {
    return JSON.stringify({error: String(e)});
  }
})()
"""


def _captured_element(node: Any, label: str) -> dict[str, Any] | None:
    """The clicked element's identity, DOMInteractedElement-shaped, for the recording.

    The full loader dereferences xpath/stable-hash internals that off-screen or zero-size
    nodes — find_by_text's specialty — may lack, and a capture failure used to mean the
    click compiled only to weak query-text fallback selectors (observed live: the "View
    all" icon replayed against an invisible twin). So on loader failure, capture the
    identity MANUALLY: the fields compile actually anchors on (tag, attributes, ax_name —
    falling back to the matched label — and xpath when reachable).
    """
    try:
        return DOMInteractedElement.load_from_enhanced_dom_tree(node).to_dict()
    except Exception as exc:  # noqa: BLE001 - fall through to the manual capture
        logger.debug("find_by_text: full element capture failed (%s); using manual identity",
                     exc)
    try:
        ax = getattr(getattr(node, "ax_node", None), "name", None)
        element: dict[str, Any] = {
            "node_name": getattr(node, "node_name", "") or "",
            "attributes": dict(getattr(node, "attributes", None) or {}),
            "ax_name": str(ax or label or "").strip(),
            "backend_node_id": getattr(node, "backend_node_id", None),
        }
        try:
            xpath = node.xpath
        except Exception:  # noqa: BLE001 - xpath traversal is exactly what fails off-screen
            xpath = None
        if xpath:
            element["x_path"] = xpath
        return element
    except Exception as exc:  # noqa: BLE001 - recording the target stays best-effort
        logger.debug("find_by_text: manual element capture failed too: %s", exc)
        return None


# A dropdown/combobox filter input: Enter there commits the focused option instead of
# submitting a search, so the auto-Enter `input` replacement must not fire it.
def _is_dropdown_filter(node: Any) -> bool:
    attrs = getattr(node, "attributes", None) or {}
    if _RS_FILTER_ID.match(attrs.get("id") or ""):
        return True
    if (attrs.get("role") or "").strip().lower() == "combobox":
        return True
    return (attrs.get("aria-autocomplete") or "").strip().lower() not in ("", "none")


# ------------------------------- stubborn-field fills -------------------------------
# browser-use clears a field by assigning `this.value = ""` from JS. React's value tracker
# wraps the value property on the ELEMENT INSTANCE, so that assignment updates the tracker's
# last-known value along with the DOM: the `input` event dispatched right after looks like a
# no-op, React skips the change, and its state still holds the OLD value — which it paints
# back over whatever we type next. The field then "won't clear", or the old figure returns
# the moment the row re-renders (the pay-forecast amount cells fail exactly this way).
#
# So we clear the way a user does, with real key events no framework can miss: focus,
# select-all, Delete, then End + Backspace-per-character for fields that refuse a selection.
# Select-all goes out as the `selectAll` EDITING COMMAND rather than a Ctrl+A/Cmd+A chord so
# it lands the same on every platform. browser-use already types char-by-char with real key
# events, so with the JS clear out of the way the whole fill is keyboard-only.
#
# Every fill is then READ BACK and repaired if the field didn't take: this is the capacity
# the agent was missing, and it is why no prompt should ever have to explain "use backspace".

# Clear+retype rounds attempted when a fill reads back wrong before we report the mismatch.
_FILL_REPAIR_ROUNDS = 2


async def _field_handle(browser_session: BrowserSession | None, node: Any):
    """(cdp_session, object_id) for `node`'s live DOM element, or None if it can't be
    resolved — every caller degrades to browser-use's own behaviour on None."""
    backend_id = getattr(node, "backend_node_id", None)
    if browser_session is None or backend_id is None:
        return None
    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
        resolved = await cdp_session.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": backend_id}, session_id=cdp_session.session_id)
        object_id = ((resolved or {}).get("object") or {}).get("objectId")
    except Exception as exc:  # noqa: BLE001 - best-effort; the normal fill path still runs
        logger.debug("could not resolve field object id: %s", exc)
        return None
    return (cdp_session, object_id) if object_id else None


async def _call_on_field(handle, declaration: str, args: list | None = None):
    """Run `declaration` with the field as `this` and return its by-value result."""
    cdp_session, object_id = handle
    result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
        params={"objectId": object_id, "functionDeclaration": declaration,
                "arguments": [{"value": a} for a in (args or [])], "returnByValue": True},
        session_id=cdp_session.session_id)
    return (result.get("result") or {}).get("value")


async def _field_value(handle) -> str | None:
    """The field's current value (textContent for contenteditable), or None when it can't
    be read — a re-rendered node leaves a stale object id, and reporting that as "" would
    look exactly like a field that refused our text. Callers skip verification on None."""
    try:
        value = await _call_on_field(
            handle, "function(){ return this.value !== undefined ? this.value "
                    ": (this.textContent || ''); }")
    except Exception as exc:  # noqa: BLE001 - a readback failure must not fail the fill
        logger.debug("field readback failed (node likely re-rendered): %s", exc)
        return None
    return "" if value is None else str(value)


async def _select_state(handle) -> tuple[str, str] | None:
    """The <select>'s current (value, selected-option label), or None when it can't be read
    (stale object id after a re-render, or the element is not a <select>) — callers treat
    None as "cannot verify", never as a mismatch."""
    try:
        state = await _call_on_field(
            handle, "function(){ if (this.tagName !== 'SELECT') return null; "
                    "var o = this.selectedOptions && this.selectedOptions[0]; "
                    "return { value: this.value == null ? '' : String(this.value), "
                    "label: (o && o.text) || '' }; }")
    except Exception as exc:  # noqa: BLE001 - verification is best-effort
        logger.debug("select readback failed (node likely re-rendered): %s", exc)
        return None
    if not isinstance(state, dict):
        return None
    return (str(state.get("value") or ""), str(state.get("label") or ""))


# ------------------------------- custom-combobox picks -------------------------------
# The app's dropdowns are react-select comboboxes, not native <select>s: an
# <input role=combobox id=react-select-N-input> that browser-use renders NAMELESS (the
# visible "Monthly"/"Select employee" label lives in a sibling div), a menu that opens on
# mousedown, and option divs that select on mousedown. Observed live (payroll run
# 20260803_112915): with no tool owning that transaction the agent hand-rolled it across
# batched actions — typed filter text made the input itself match its own find_by_text
# query, a Save click batched into the same step closed the menu, and the run died after
# 17 steps of dropdown thrashing. select_dropdown's non-native branch below owns the whole
# open → list → pick → verify sequence in ONE action, and on a miss reports the options
# that ACTUALLY exist so the agent can course-correct instead of hunting phantom text.

_CB_STAMP = "data-ao-cb-root"

# Run ON the agent-indexed node (this = element): resolve the combobox the agent means.
# Accepts the combobox <input> itself, anything INSIDE the widget (placeholder/value div,
# container), or a small wrapper around it — but refuses an ancestor holding SEVERAL
# comboboxes: guessing between adjacent widgets is exactly the wrong-target bug this tool
# exists to kill (the employee name typed into the Monthly frequency dropdown).
_CB_RESOLVE_JS = """
function () {
  var SEL = 'input[role=combobox], input[id^="react-select"][id$="-input"]';
  var isCb = function (e) {
    if (!e || e.tagName !== 'INPUT') return false;
    return (e.getAttribute('role') || '').toLowerCase() === 'combobox' ||
           /^react-select-.+-input$/.test(e.id || '');
  };
  var input = null, root = null;
  if (isCb(this)) {
    input = this;
    root = input.parentElement || input;
    for (var i = 0; i < 4; i++) {
      var p = root.parentElement;
      if (!p || p.querySelectorAll(SEL).length !== 1) break;
      root = p;
    }
  } else {
    var el = this;
    for (var hops = 0; el && hops < 8; hops++, el = el.parentElement) {
      if (!el.querySelectorAll) continue;
      var found = el.querySelectorAll(SEL);
      if (found.length === 1) { input = found[0]; root = el; break; }
      if (found.length > 1) return { error: 'ambiguous', count: found.length };
    }
  }
  if (!input) return { error: 'none' };
  if (!input.id) input.id = 'ao-cb-' + (++window.__ao_cb_seq || (window.__ao_cb_seq = 1));
  document.querySelectorAll('[STAMP]').forEach(function (n) { n.removeAttribute('STAMP'); });
  root.setAttribute('STAMP', '1');
  return { input_id: input.id };
}
""".replace("STAMP", _CB_STAMP)

# Document-level ops on the resolved combobox, addressed by the input's id.
#   open    — focus the input and fire a real pointer+mouse sequence on it (react-select
#             opens on the control's mousedown; a bare .focus() or .click() does nothing).
#   options — the menu's CURRENT options: react-select instance-prefixed ids first, then
#             the input's aria-controls/owns listbox, then any [role=option]. Options with
#             no id get one stamped so a later 'pick' can address them.
#   pick    — dispatch the pointer+mouse sequence on ONE option (by stamped id). react-select
#             selects on the option's mousedown — this is the click el.click() never was.
#   state   — menu-open flag + the widget root's visible text (the read-back source).
_CB_OPS_JS = """
(function () {
  var ID = %(id)s, OP = %(op)s, WANTED = %(wanted)s;
  var input = document.getElementById(ID);
  if (!input) return { error: 'input-gone' };
  var fire = function (el, kinds) {
    kinds.forEach(function (t) {
      try {
        var ev = (t.indexOf('pointer') === 0 && window.PointerEvent)
          ? new PointerEvent(t, { bubbles: true, cancelable: true, view: window })
          : new MouseEvent(t, { bubbles: true, cancelable: true, view: window });
        el.dispatchEvent(ev);
      } catch (e) {}
    });
  };
  var SEQ = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'];
  if (OP === 'open') {
    try { input.focus(); } catch (e) {}
    fire(input, SEQ);
    return { ok: true };
  }
  var optionNodes = function () {
    var m = (input.id || '').match(/^(react-select-\\d+)-input$/);
    var nodes = [];
    if (m) {
      nodes = document.querySelectorAll('[id^="' + m[1] + '-option"]');
      if (nodes.length) return Array.prototype.slice.call(nodes);
    }
    var owns = input.getAttribute('aria-controls') || input.getAttribute('aria-owns');
    var box = owns && document.getElementById(owns);
    if (box) {
      nodes = box.querySelectorAll('[role=option], [id*="-option"]');
      if (nodes.length) return Array.prototype.slice.call(nodes);
    }
    return Array.prototype.slice.call(document.querySelectorAll('[role=option]'));
  };
  var opts = optionNodes().filter(function (o) { return (o.innerText || '').trim(); });
  opts.forEach(function (o, i) { if (!o.id) o.id = ID + '-ao-opt-' + i; });
  var listing = opts.map(function (o) {
    return { id: o.id, text: (o.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 120) };
  });
  if (OP === 'options') return { options: listing };
  if (OP === 'pick') {
    var target = null;
    for (var i = 0; i < opts.length; i++) {
      if (opts[i].id === WANTED) { target = opts[i]; break; }
    }
    if (!target) return { error: 'option-gone', options: listing };
    try { target.scrollIntoView({ block: 'nearest' }); } catch (e) {}
    fire(target, SEQ);
    var attrs = {};
    ['id', 'role', 'class', 'aria-label'].forEach(function (a) {
      var v = target.getAttribute(a);
      if (v) attrs[a] = v;
    });
    return { clicked: true, tag: target.tagName.toLowerCase(), attrs: attrs };
  }
  if (OP === 'state') {
    var root = document.querySelector('[STAMP]');
    var display = root ? (root.innerText || '').replace(/\\s+/g, ' ').trim() : '';
    return { menu_open: listing.length > 0, display: display.slice(0, 200) };
  }
  return { error: 'bad-op' };
})()
""".replace("STAMP", _CB_STAMP)


def _choose_option(options: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    """The option the agent means: exact normalized-text match first; else the UNIQUE
    option containing the target as a word-aligned phrase (react-select option cards pad
    the label with detail lines). Ambiguity returns None — the caller lists the options
    rather than guessing (guessing is how the wrong employee got picked in live runs)."""
    want = _norm_phrase(target)
    if not want:
        return None
    for opt in options:
        if _norm_phrase(str(opt.get("text") or "")) == want:
            return opt
    partial = [o for o in options
               if f" {want} " in f" {_norm_phrase(str(o.get('text') or ''))} "]
    return partial[0] if len(partial) == 1 else None


def _cb_option_lines(options: list[dict[str, Any]], limit: int = 10) -> str:
    shown = ", ".join(f"'{str(o.get('text') or '')[:60]}'" for o in options[:limit])
    more = f" (+{len(options) - limit} more)" if len(options) > limit else ""
    return shown + more


async def _cb_op(browser_session, op: str, input_id: str, wanted: str | None = None):
    expr = _CB_OPS_JS % {"id": json.dumps(input_id), "op": json.dumps(op),
                         "wanted": json.dumps(wanted)}
    return await _eval_js(browser_session, expr)


async def _cb_poll_options(browser_session, input_id: str,
                           timeout: float) -> list[dict[str, Any]]:
    """The menu's options, polling until they render (react-select mounts the menu a beat
    after the open mousedown; server-backed lists take longer). Empty list on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            got = await _cb_op(browser_session, "options", input_id)
        except Exception as exc:  # noqa: BLE001 - keep polling; the page may be re-rendering
            logger.debug("combobox options poll failed: %s", exc)
            got = None
        opts = (got or {}).get("options") if isinstance(got, dict) else None
        if opts:
            return opts
        if asyncio.get_event_loop().time() >= deadline:
            return []
        await asyncio.sleep(0.25)


async def _combobox_select(browser_session, params: SelectDropdownOptionAction,
                           handle) -> ActionResult:
    """select_dropdown's non-native branch: one action that opens the custom combobox,
    reads what the menu ACTUALLY lists, picks the matching option with the pointer+mouse
    sequence the widget listens for, and read-back-verifies the widget now shows it.
    Same fail-honest contract as the native path — and a miss reports the real option
    texts, which is the receipt that redirects a wrong-semantics agent."""
    target = (params.text or "").strip()
    info = None
    if handle is not None:
        try:
            info = await _call_on_field(handle, _CB_RESOLVE_JS)
        except Exception as exc:  # noqa: BLE001 - resolution failure reads as "no combobox"
            logger.debug("combobox resolve failed: %s", exc)
    if not isinstance(info, dict) or not info.get("input_id"):
        if isinstance(info, dict) and info.get("error") == "ambiguous":
            return ActionResult(error=(
                f"select_dropdown at index {params.index}: that element contains "
                f"{info.get('count')} different comboboxes — cannot know which one you "
                "mean. Locate the dropdown by its VISIBLE placeholder/current value with "
                "find_by_text (e.g. 'Select employee', 'Monthly') and call select_dropdown "
                "on THAT index."))
        return ActionResult(error=(
            f"select_dropdown at index {params.index}: the element is not a native "
            "<select> and no combobox input (role=combobox / react-select) exists at or "
            "around it. Pass the index of the dropdown's input, its placeholder/current-"
            "value text, or its container."))
    input_id = str(info["input_id"])

    try:
        await _cb_op(browser_session, "open", input_id)
    except Exception as exc:  # noqa: BLE001 - report; the agent recovers via its receipt
        return ActionResult(error=(
            f"select_dropdown '{target}' at index {params.index}: could not open the "
            f"combobox ({exc})."))
    options = await _cb_poll_options(browser_session, input_id, timeout=4.0)

    chosen = _choose_option(options, target)
    filtered = False
    if chosen is None:
        # Not among the visible options (or none rendered): type the text into the filter
        # input — CDP insertText fires the trusted input events React's filter needs —
        # and give the narrowed/loaded list one more look.
        try:
            await _eval_js(browser_session,
                           f"document.getElementById({json.dumps(input_id)}).focus()")
            cdp_session = await browser_session.get_or_create_cdp_session()
            await cdp_session.cdp_client.send.Input.insertText(
                params={"text": target}, session_id=cdp_session.session_id)
            filtered = True
        except Exception as exc:  # noqa: BLE001 - a read-only combobox refuses typing
            logger.debug("combobox filter typing failed: %s", exc)
        refreshed = await _cb_poll_options(browser_session, input_id, timeout=2.0)
        if refreshed:
            options = refreshed
        chosen = _choose_option(options, target)
    if chosen is None:
        listed = _cb_option_lines(options)
        if options:
            return ActionResult(error=(
                f"select_dropdown '{target}' at index {params.index}: no such option. "
                f"The dropdown ACTUALLY lists: {listed}. These are all that exist — "
                f"re-read the task and pick one of these exact texts with "
                f"select_dropdown(index={params.index}, text='<option>'). Do NOT hunt "
                f"the page for '{target}'."))
        return ActionResult(error=(
            f"select_dropdown '{target}' at index {params.index}: the combobox opened "
            "but no options rendered within the wait"
            + (" (even after typing to filter)" if filtered else "")
            + ". The list may load from the server — wait a moment and call "
              "select_dropdown again with the same arguments."))

    picked = None
    try:
        picked = await _cb_op(browser_session, "pick", input_id, wanted=str(chosen["id"]))
    except Exception as exc:  # noqa: BLE001 - the read-back below is the real verdict
        logger.debug("combobox pick dispatch failed: %s", exc)
    if not (isinstance(picked, dict) and picked.get("clicked")):
        last_seen = picked.get("options") if isinstance(picked, dict) else None
        listed = _cb_option_lines(last_seen or options)
        return ActionResult(error=(
            f"select_dropdown '{target}' at index {params.index}: the option "
            f"'{chosen.get('text')}' vanished before it could be clicked (menu re-render). "
            f"Options last seen: {listed}. Call select_dropdown again with the same "
            "arguments."))

    # Read-back: the pick took when the menu is closed again AND the widget now shows the
    # option. Poll briefly — the value lands after React commits.
    display, took = "", False
    deadline = asyncio.get_event_loop().time() + 2.0
    want = _norm_phrase(str(chosen.get("text") or target))
    while True:
        try:
            state = await _cb_op(browser_session, "state", input_id)
        except Exception:  # noqa: BLE001 - a re-render mid-poll is fine, try again
            state = None
        if isinstance(state, dict):
            display = str(state.get("display") or "")
            if not state.get("menu_open") and \
                    f" {want} " in f" {_norm_phrase(display)} ":
                took = True
                break
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.25)

    # Record the OPTION's identity so the recording compiles to a replayable by-label
    # click (script_compile routes non-native select_dropdown picks through the same
    # react-select synthesis as recorded option clicks).
    meta = {"interacted_element": {
        "node_name": str(picked.get("tag") or "div").upper(),
        "attributes": dict(picked.get("attrs") or {}),
        "ax_name": str(chosen.get("text") or target).strip(),
    }}
    if took:
        msg = (f"Selected '{chosen.get('text')}' in the combobox at index {params.index} — "
               f"it now shows '{display}'. Do NOT set it again.")
        logger.info("🔽 %s", msg)
        return ActionResult(extracted_content=msg, include_in_memory=True,
                            long_term_memory=msg, metadata=meta)
    return ActionResult(error=(
        f"select_dropdown '{target}' at index {params.index}: clicked the option "
        f"'{chosen.get('text')}' but the combobox does not show it (reads: '{display}'). "
        "Do not report it as set — re-check the field and call select_dropdown again if "
        "it still shows the old value."))


async def _press(handle, key: str, code: str, vk: int, *,
                 commands: list[str] | None = None, repeat: int = 1) -> None:
    """Dispatch `repeat` real keyDown/keyUp pairs at the focused element."""
    cdp_session, _ = handle
    down: dict[str, Any] = {"type": "keyDown", "key": key, "code": code,
                            "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
    if commands:
        down["commands"] = commands
    up = {"type": "keyUp", "key": key, "code": code,
          "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
    for _ in range(repeat):
        await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
            params=down, session_id=cdp_session.session_id)
        await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
            params=up, session_id=cdp_session.session_id)


async def _keyboard_clear(handle) -> bool:
    """Empty the field with real keystrokes. True once a readback shows it empty."""
    try:
        await _call_on_field(handle, "function(){ this.focus(); return true; }")
        current = await _field_value(handle)
        if current is None:
            return False  # unreadable: let browser-use's own clear have its turn
        if not current:
            return True
        # Select-all as an editing command (platform-independent), then Delete.
        await _press(handle, "a", "KeyA", 65, commands=["selectAll"])
        await _press(handle, "Delete", "Delete", 46)
        after_delete = await _field_value(handle)
        if not after_delete:
            return after_delete is not None
        # Fields that drop the selection: walk back from the end, one Backspace per
        # character (+ slack for anything the page re-inserted while we typed).
        logger.debug("select-all clear left %r; falling back to Backspace", after_delete)
        await _press(handle, "End", "End", 35)
        await _press(handle, "Backspace", "Backspace", 8, repeat=len(after_delete) + 4)
        return await _field_value(handle) == ""
    except Exception as exc:  # noqa: BLE001 - caller falls back to the JS clear
        logger.debug("keyboard clear failed: %s", exc)
        return False


def build_tools() -> Tools:
    """Return a browser-use `Tools` registry with our custom actions added, `evaluate`
    removed (JS form-fills are unrecordable — see module docstring), and the built-in
    `input` overridden by the auto-Enter variant."""
    tools = Tools(exclude_actions=["evaluate"])

    # Same name/param model as the built-in: same-name registration OVERRIDES it (excluding
    # "input" would drop this replacement too), and the recorder still captures the
    # interacted element via the index param.
    @tools.action(
        'Input text into element by index, then press Enter automatically to submit/apply it '
        '(Enter is suppressed for dropdown/combobox filters — click the option you want '
        'instead). Clears existing text by default; pass text="" to clear only, or '
        "clear=False to append.",
        param_model=InputTextAction,
    )
    async def input(params: InputTextAction, browser_session=None) -> ActionResult:
        node = await browser_session.get_element_by_index(params.index)
        if node is None:
            msg = (f"Element index {params.index} not available - page may have changed. "
                   "Try refreshing browser state.")
            logger.warning("⚠️ %s", msg)
            return ActionResult(extracted_content=msg)
        dropdown = _is_dropdown_filter(node)
        # Clear with real keystrokes ourselves when we can reach the element, and hand
        # browser-use clear=False so its JS `value = ""` (invisible to React — see the
        # stubborn-field notes above) never runs. `params.clear` is left untouched: the
        # compiler reads it to decide whether the replayed fill clears too.
        handle = await _field_handle(browser_session, node)
        kbd_cleared = bool(handle) and params.clear and await _keyboard_clear(handle)
        try:
            event = browser_session.event_bus.dispatch(TypeTextEvent(
                node=node, text=params.text, clear=params.clear and not kbd_cleared))
            await event
            input_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)
            # Read back BEFORE Enter and retry the whole clear+type when the field kept
            # another value: a repaired fill here is the difference between the run
            # continuing and the agent burning steps refreshing the page. It must be before
            # Enter — a search box that empties itself on submit would otherwise read back
            # as a field that refused the text.
            repaired, verified = False, None
            if handle and params.text:
                for attempt in range(_FILL_REPAIR_ROUNDS):
                    verified = await _field_value(handle)
                    if verified is None or _value_took(params.text, verified):
                        break  # took, or unreadable — nothing to repair against
                    logger.warning("field kept %r after typing %r; clearing with keystrokes "
                                   "and retyping (attempt %d/%d)", verified, params.text,
                                   attempt + 1, _FILL_REPAIR_ROUNDS)
                    await _keyboard_clear(handle)
                    retry = browser_session.event_bus.dispatch(
                        TypeTextEvent(node=node, text=params.text, clear=False))
                    await retry
                    await retry.event_result(raise_if_any=True, raise_if_none=False)
                    repaired = True
                else:
                    verified = await _field_value(handle)  # readback after the last repair
            if not dropdown:
                enter = browser_session.event_bus.dispatch(SendKeysEvent(keys="Enter"))
                await enter
                await enter.event_result(raise_if_any=True, raise_if_none=False)
        except Exception as exc:  # noqa: BLE001 - report; the agent recovers via its receipt
            logger.error("input failed at index %s: %s", params.index, exc)
            return ActionResult(error=f"Failed to type text into element {params.index}: {exc}")
        meta = dict(input_metadata) if isinstance(input_metadata, dict) else {}
        meta.pop("actual_value", None)  # stale once we repaired; `verified` supersedes it
        # The compiler's signal to mirror the Enter as a replay `press` step; a structured
        # flag, not the receipt text, so rewording the message can't change replays.
        meta["auto_enter"] = not dropdown
        if dropdown:
            msg = (f"Typed '{params.text}' (dropdown filter — Enter suppressed; "
                   "click the option you want)")
        else:
            msg = f"Typed '{params.text}' and pressed Enter"
        if verified is not None and not _value_took(params.text, verified):
            # Two causes, one receipt: the text went into a NEIGHBOURING control (the field
            # you aimed at never changed), or this field refuses the value. Escape+relocate
            # fixes the first and costs little on the second; the reload is the last resort.
            msg += (f". WARNING: the field still reads '{verified}', not '{params.text}' — the "
                    "value did NOT take, even after clearing it with keystrokes. Do NOT report "
                    "this field as set. Press Escape to close any dropdown the typing opened, "
                    "re-locate the field with find_by_text, and retype it. If it STILL refuses, "
                    "reload the page, navigate back to this field, and redo the change.")
        elif repaired:
            msg += " (the field first kept its old value; cleared and retyped, now correct)"
        logger.debug(msg)
        return ActionResult(extracted_content=msg, long_term_memory=msg, metadata=meta)

    # Same-name override of the built-in select_dropdown (same mechanism as `input` above).
    # The built-in awaits the picker's self-report UNBOUNDED and trusts it: on an ad-heavy
    # page the post-selection settle can outlive the event timeout, the receipt comes back
    # empty, and the agent re-sets an option that already took (observed live: the country
    # <select> was set three times). Here the element itself is READ BACK after every
    # dispatch and the receipt states what the select now shows — the same verify-after
    # rule the `input` fill follows. Same param model, so recorded history keeps the exact
    # built-in shape and the compiler's native-select path needs no changes.
    @tools.action(
        'Pick an option in ANY dropdown by its visible text: native <select> elements AND '
        'custom comboboxes (react-select etc.). For a custom combobox, pass the index of '
        'its input, its placeholder/current-value text element, or its container — the '
        'tool opens the menu, picks the matching option, and verifies it took, all in one '
        'action. If your text is not among the options, the error lists what the dropdown '
        'ACTUALLY offers.',
        param_model=SelectDropdownOptionAction,
    )
    async def select_dropdown(params: SelectDropdownOptionAction,
                              browser_session=None) -> ActionResult:
        node = await browser_session.get_element_by_index(params.index)
        if node is None:
            msg = (f"Element index {params.index} not available - page may have changed. "
                   "Try refreshing browser state.")
            logger.warning("⚠️ %s", msg)
            return ActionResult(extracted_content=msg)
        target = (params.text or "").strip()
        # Resolve the element BEFORE dispatching: a post-selection re-render leaves a stale
        # object id, which must read as "cannot verify", not as a refused option.
        handle = await _field_handle(browser_session, node)
        if str(getattr(node, "tag_name", "") or "").lower() != "select":
            # Custom combobox (react-select & co.) — the built-in SelectDropdownOptionEvent
            # only understands native <select>s, which is why this tool used to dead-end
            # here and agents hand-rolled dropdown picks across batched steps.
            return await _combobox_select(browser_session, params, handle)

        def _took(state: tuple[str, str] | None) -> bool:
            if state is None:
                return False
            want = " ".join(target.split()).lower()
            # Label first, value as fallback — the same ladder replay's select uses.
            return any(" ".join(s.split()).lower() == want for s in state if s)

        failure: str | None = None
        try:
            event = browser_session.event_bus.dispatch(
                SelectDropdownOptionEvent(node=node, text=params.text))
            await event
            data = await event.event_result(timeout=10.0, raise_if_any=True,
                                            raise_if_none=False)
            if not (isinstance(data, dict) and data.get("success") == "true"):
                failure = (data.get("error") if isinstance(data, dict) else None) \
                    or "the picker returned no confirmation"
        except Exception as exc:  # noqa: BLE001 - the read-back below is the real verdict
            failure = str(exc) or exc.__class__.__name__

        state = await _select_state(handle) if handle else None
        shows = (state[1] or state[0]) if state else ""
        if _took(state):
            if failure is None:
                msg = (f"Selected dropdown option '{params.text}' at index {params.index} — "
                       f"the select now reads '{shows}'.")
            else:
                msg = (f"Selected dropdown option '{params.text}' at index {params.index}. "
                       f"The picker's own confirmation failed ({failure}), but read-back "
                       f"confirms the select now reads '{shows}' — the option IS set; do "
                       "NOT set it again.")
            logger.info("🔽 %s", msg)
            return ActionResult(extracted_content=msg, include_in_memory=True,
                                long_term_memory=msg)
        if failure is None and state is None:
            # The picker claims success and the element is unreadable — no grounds to
            # overrule the claim; keep the built-in's receipt.
            msg = f"Selected dropdown option '{params.text}' at index {params.index}"
            logger.info("🔽 %s", msg)
            return ActionResult(extracted_content=msg, include_in_memory=True,
                                long_term_memory=msg)
        if state is None:
            return ActionResult(error=(
                f"select_dropdown '{params.text}' at index {params.index} failed: {failure}"))
        reason = f" ({failure})" if failure else ""
        return ActionResult(error=(
            f"select_dropdown '{params.text}' at index {params.index} did NOT take{reason}: "
            f"the select still reads '{shows}'. Do not report it as set — call "
            f"dropdown_options(index={params.index}) to see the exact option texts and pick "
            "again with one of them."))

    @tools.action(
        "Abandon the CURRENT objective/step and continue the run. Use when a step failed but "
        "the rest of the workflow does NOT depend on it. Pass a short reason."
    )
    async def skip_step(reason: str) -> ActionResult:
        logger.info("⤼ skip_step: %s", reason)
        note = f"Skipped step: {reason}"
        return ActionResult(
            extracted_content=note,
            long_term_memory=note,
            include_in_memory=True,
        )

    @tools.action(
        "Terminate the ENTIRE run as a failure. Use when a required objective cannot be met and "
        "the next phase or the done condition depends on it. Pass a short reason."
    )
    async def fail_and_stop(reason: str) -> ActionResult:
        logger.info("■ fail_and_stop: %s", reason)
        return ActionResult(
            is_done=True,
            success=False,
            error=reason,
            extracted_content=f"Run failed and stopped: {reason}",
            long_term_memory=f"Run failed and stopped: {reason}",
            include_in_memory=True,
        )

    @tools.action(
        "Scroll for discovery, capped at 0.5 pages per call (values above 0.5 are clamped). "
        "down=True scrolls down, False scrolls up; pages is the fraction of a viewport to move "
        "(use ~0.2 to hunt for an unknown element). Scrolls the main window, falling back to the "
        "largest scrollable container if the window itself does not move."
    )
    # NOTE: `browser_session` is intentionally UNANNOTATED and LAST. browser-use injects it by
    # NAME; adding a `: BrowserSession` hint makes its registry compare our imported class against
    # its own and raise "conflicts with special argument injected". Do not annotate it.
    async def capped_scroll(down: bool = True, pages: float = 0.2, browser_session=None) -> ActionResult:
        capped = max(0.0, min(float(pages), 0.5))
        js = (
            "(function(){var dir=%s?1:-1;var px=Math.round(%f*window.innerHeight)*dir;"
            "var y0=window.scrollY;window.scrollBy(0,px);"
            "if(window.scrollY!==y0)return 'window';"
            "var els=[].slice.call(document.querySelectorAll('*')).filter(function(e){"
            "var s=getComputedStyle(e);return /(auto|scroll)/.test(s.overflowY)&&"
            "e.scrollHeight>e.clientHeight+4;});"
            "els.sort(function(a,b){return b.clientHeight*b.clientWidth-a.clientHeight*a.clientWidth;});"
            "if(els.length){els[0].scrollBy(0,px);return 'container';}return 'none';})()"
            % ("true" if down else "false", capped)
        )
        try:
            where = await _eval_js(browser_session, js)
        except Exception as exc:  # noqa: BLE001 - a scroll must never crash the run
            logger.warning("capped_scroll failed: %s", exc)
            return ActionResult(error=f"capped_scroll failed: {exc}")
        direction = "down" if down else "up"
        moved = "nothing scrolled (already at edge)" if where == "none" else f"scrolled the {where}"
        msg = f"capped_scroll {direction} {capped} page(s): {moved}"
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    @tools.action(
        "Find interactive elements matching `text`, searched in a FRESH page snapshot. Matches "
        "when every word of `text` appears in the element's visible text, aria-label, title, "
        "placeholder, value, name, or id (case-insensitive; punctuation ignored, so '+ Invoice' "
        "matches a button labeled 'Invoice' or id 'btnInvoice'). Returns every match with its "
        "CURRENT click index — use click(index) with that index immediately. Use this to locate "
        "a specific button/link/tab/menu item by its label, especially after an 'Element index "
        "N not available' failure. Pass click_first=True to also click it when exactly one "
        "element matches."
    )
    async def find_by_text(text: str, click_first: bool = False, browser_session=None) -> ActionResult:  # injected by name; do not annotate
        query = (text or "").strip()
        if not query:
            return ActionResult(error="find_by_text: text must be non-empty")
        if browser_session is None:
            return ActionResult(error="find_by_text: BrowserSession not injected")
        try:
            # Fresh (non-cached) snapshot: the indexes we return are exactly the ones
            # click(index) resolves against right now.
            state = await browser_session.get_browser_state_summary(include_screenshot=False)
        except Exception as exc:  # noqa: BLE001 - a lookup must never crash the run
            logger.warning("find_by_text failed to snapshot the page: %s", exc)
            return ActionResult(error=f"find_by_text: could not read page state: {exc}")

        # Token match, not literal substring: "+ Invoice" must find a button whose accessible
        # text is just "Invoice" (the "+" is an icon) or whose id is "btnInvoice".
        tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
        if not tokens:
            return ActionResult(error=f"find_by_text: no searchable text in {query!r}")
        matches = _matching_nodes(state, tokens)

        def _line(idx: int, node: Any, label: str) -> str:
            attrs = node.attributes or {}
            parts = [f"index={idx} <{node.tag_name}> text='{label[:80]}'"]
            for attr in ("id", "aria-label"):
                if attrs.get(attr):
                    parts.append(f"{attr}='{attrs[attr][:60]}'")
            return " ".join(parts)

        page_url = getattr(state, "url", "") or ""
        if not matches:
            # Fallback: the target may exist in the DOM but be EXCLUDED from the interactive
            # snapshot (a 0x0 button in a Fluent virtualized ScrollablePane, an off-screen
            # control). Query the live DOM directly and, when click_first, click it via its
            # own handler — the only way to reach a functional 0x0 element.
            raw = None
            try:
                expr = _RAW_FIND_JS % (json.dumps(tokens), "true" if click_first else "false")
                raw = await _eval_js(browser_session, expr)
            except Exception as exc:  # noqa: BLE001 - fallback is best-effort
                logger.debug("find_by_text raw-DOM fallback failed: %s", exc)
            if raw and not raw.get("error") and raw.get("count"):
                if raw.get("clicked"):
                    # Record the clicked element's identity + the fact that it was clicked
                    # while INVISIBLE (hidden_click): replay's visibility-gated resolver
                    # can never reach a hover-revealed/0-size control, so compile marks the
                    # step hidden_ok and replay dispatches the click the same way.
                    meta = None
                    el = raw.get("element") or {}
                    if el.get("tag"):
                        meta = {"interacted_element": {
                            "node_name": str(el.get("tag") or ""),
                            "attributes": dict(el.get("attrs") or {}),
                            "ax_name": str(raw.get("name") or "").strip(),
                            "hidden_click": True,
                        }}
                    # Receipt semantics (why it must scream "already clicked", and the
                    # wrong-control warning on a name mismatch) live in _hidden_click_receipt.
                    msg = _hidden_click_receipt(query, str(raw.get("name") or "").strip())
                    logger.info("🔎 %s", msg)
                    return ActionResult(extracted_content=msg, long_term_memory=msg,
                                        include_in_memory=True, metadata=meta)
                if click_first and raw.get("refused"):
                    # The raw path found only an INVISIBLE element whose name is not the
                    # query — clicking it would be the wrong-control no-op that looped a
                    # live run for 6 steps. Nothing was clicked; say what actually exists.
                    msg = (f"find_by_text('{query}'): NOT clicked. The only match is a "
                           f"0-size/hidden element named '{str(raw.get('name') or '').strip()}', "
                           f"which is NOT '{query}' — almost certainly the wrong control. "
                           f"No clickable control named '{query}' exists on this page right "
                           f"now. If you expected a dropdown option: open the dropdown and "
                           f"choose from the options it ACTUALLY lists instead of this text.")
                    logger.info("🔎 %s", msg)
                    return ActionResult(extracted_content=msg, long_term_memory=msg,
                                        include_in_memory=True, metadata={"no_click": True})
                names = ", ".join(f"'{n}'" for n in (raw.get("names") or []) if n)
                msg = (f"find_by_text('{query}'): {raw['count']} match(es) exist in the DOM but "
                       f"are NOT clickable via index (0-size/virtualized): {names}. Re-call "
                       f"find_by_text('{query}', click_first=true) to click the best match directly.")
                logger.info("🔎 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True, metadata={"no_click": True})
            msg = (
                f"find_by_text('{query}'): 0 matches on the CURRENT page ({page_url}). "
                "The element is not in this page's DOM. FIRST check: is this the page you think "
                "you are on? If a previous click navigated you away (e.g. back to a list page), "
                "recover with go_back or re-open the right section. Otherwise scroll with "
                "capped_scroll or apply the ELEMENT NOT FOUND POLICY; do not repeat this exact query."
            )
            logger.info("🔎 %s", msg)
            # no_click: this was a PROBE that touched nothing — without the stamp, compile
            # treats a metadata-less click_first result as a dropped-metadata click and
            # emits a semantic find_click (observed live: a closed-panel check committed
            # a find_click('save') that then failed every replay on the healthy page).
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True, metadata={"no_click": True})

        # Exact-label preference: 'Inputs' matches both the link AND its parent <li> container;
        # when exactly one candidate's own label/aria-label/id equals the query, that is the
        # intended target — click it despite the partial matches.
        if click_first and len(matches) > 1:
            def _norm(s: str) -> str:
                return " ".join(re.split(r"[^a-z0-9]+", (s or "").lower())).strip()
            wanted = _norm(query)
            exact = [m for m in matches
                     if _norm(m[2]) == wanted
                     or any(_norm((m[1].attributes or {}).get(a) or "") == wanted
                            for a in ("aria-label", "id", "name"))]
            if len(exact) == 1:
                matches = exact

        if click_first and len(matches) == 1:
            idx, node, label = matches[0]
            tag = str(getattr(node, "tag_name", "") or "").lower()
            attrs_map = getattr(node, "attributes", None) or {}
            # Native <select>s (and file inputs) can never be clicked — browser-use refuses
            # the click, and the refusal comes back as a RETURNED {'validation_error': ...},
            # not an exception, so a blind dispatch reads as success (observed live: two
            # phantom "clicked" receipts on the country <select> convinced the agent its
            # already-applied selection kept failing). Refuse up front, naming the action
            # that actually works, and stamp no_click so nothing compiles from this.
            if tag == "select":
                msg = (f"find_by_text('{query}'): found the single match "
                       f"{_line(idx, node, label)} but did NOT click it — it is a native "
                       f"<select>, which cannot be clicked. Use dropdown_options(index={idx}) "
                       f"to list its options, then select_dropdown(index={idx}, "
                       f"text='<option>') to pick one.")
                logger.info("🔎 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True, metadata={"no_click": True})
            if tag == "input" and str(attrs_map.get("type") or "").lower() == "file":
                msg = (f"find_by_text('{query}'): found the single match "
                       f"{_line(idx, node, label)} but did NOT click it — it is a file "
                       f"input; use upload_file on index {idx} instead.")
                logger.info("🔎 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True, metadata={"no_click": True})
            # Self-match: when the single match is a text-entry field whose IDENTITY
            # (aria-label/title/placeholder/name/id) does not carry the query, the only
            # reason it matched is the text sitting IN it — almost always what the agent
            # itself just typed (observed live: the employee name typed into the wrong
            # react-select matched only itself, and the "clicked the single match" receipt
            # convinced the agent a dropdown option had been selected). Clicking it is a
            # no-op; refuse instead. A field found by its placeholder/label stays clickable
            # — that is the legitimate way to focus/open a combobox.
            if tag == "textarea" or (
                tag == "input"
                and str(attrs_map.get("type") or "text").lower()
                not in ("button", "submit", "reset", "checkbox", "radio", "image", "file")
            ):
                identity = " ".join(
                    str(attrs_map.get(a) or "")
                    for a in ("aria-label", "title", "placeholder", "name", "id")
                )
                identity = (identity + " " + (_descendant_icon_hints(node) or "")).lower()
                if not all(t in identity for t in tokens):
                    msg = (f"find_by_text('{query}'): found the single match "
                           f"{_line(idx, node, label)} but did NOT click it — the only "
                           f"'{query}' on this page is the text INSIDE that input field "
                           f"(did you just type it there?), and clicking it selects "
                           f"nothing. If you expected a dropdown option named '{query}', "
                           f"the open menu does not list it ('No options' means the list "
                           f"is empty) — you may have typed into the WRONG field. Clear "
                           f"this field, open the intended control (find_by_text on its "
                           f"placeholder/label text, click_first=true), and pick from the "
                           f"options it ACTUALLY shows. Only click(index={idx}) if this "
                           f"field itself is truly your target.")
                    logger.info("🔎 %s", msg)
                    return ActionResult(extracted_content=msg, long_term_memory=msg,
                                        include_in_memory=True, metadata={"no_click": True})
            try:
                event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                res = await event.event_result(raise_if_any=True, raise_if_none=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("find_by_text click failed: %s", exc)
                return ActionResult(error=f"find_by_text: found '{label[:80]}' but click failed: {exc}")
            # browser-use also refuses clicks by RETURNING {'validation_error': ...} (its own
            # click action checks for exactly this) — a refusal dict is NOT a click.
            if isinstance(res, dict) and res.get("validation_error"):
                msg = (f"find_by_text('{query}'): found the single match "
                       f"{_line(idx, node, label)} but the click was REFUSED: "
                       f"{res['validation_error']} Nothing was clicked.")
                logger.info("🔎 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True, metadata={"no_click": True})
            # Record WHAT we clicked so script_compile can turn this custom action into a real
            # click step (a custom action carries no index, so browser-use captures no
            # interacted_element for it). Same DOMInteractedElement shape a built-in click records.
            captured = _captured_element(node, label)
            meta = {"interacted_element": captured} if captured else None
            msg = f"find_by_text('{query}'): clicked the single match {_line(idx, node, label)}"
            logger.info("🔎 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True, metadata=meta)

        shown = matches[:25]
        lines = [_line(idx, node, label) for idx, node, label in shown]
        tail = "" if len(matches) <= 25 else f"\n...and {len(matches) - 25} more — narrow your text."
        guidance = (
            f"Ambiguous: {len(matches)} candidates. Do NOT call find_by_text('{query}') again — "
            "pick the right index from the list above and click(index) NOW."
            if click_first and len(matches) > 1
            else "Click your target with click(index) NOW — indexes are fresh but go stale on re-render."
        )
        content = (
            f"find_by_text('{query}'): {len(matches)} match(es):\n" + "\n".join(lines) + tail
            + f"\n{guidance}"
        )
        memory = (
            f"find_by_text('{query}'): {len(matches)} match(es); "
            + "; ".join(f"index={idx} <{node.tag_name}> '{label[:40]}'" for idx, node, label in shown[:5])
        )
        logger.info("🔎 find_by_text('%s'): %d match(es)", query, len(matches))
        # no_click: a candidate LISTING — the agent clicks by index next; compiling this
        # result as a find_click would bake a phantom duplicate click into the recording.
        return ActionResult(extracted_content=content, long_term_memory=memory,
                            include_in_memory=True, metadata={"no_click": True})

    @tools.action(
        "Capture on-page data so this step can report it AND future replays can re-read it "
        "fresh without an LLM. Locates the element whose visible text/labels match every "
        "word of `text` (same matching as find_by_text) and records its current text under "
        "`label` (short snake_case role name, e.g. generated_identity, account_balance). "
        "Read-only — clicks nothing. Prefer ONE call on the block/card that shows the "
        "facts (the block's whole text is the value; later steps parse it). Do NOT use it "
        "for values your own instructions specify or that you typed/picked yourself — "
        "state those in your done message instead."
    )
    async def extract_data(text: str, label: str, browser_session=None) -> ActionResult:  # injected by name; do not annotate
        query = (text or "").strip()
        slug = re.sub(r"[^a-z0-9_]+", "_", (label or "").strip().lower()).strip("_") or "value"
        if not query:
            return ActionResult(error="extract_data: text must be non-empty")
        if browser_session is None:
            return ActionResult(error="extract_data: BrowserSession not injected")
        try:
            state = await browser_session.get_browser_state_summary(include_screenshot=False)
        except Exception as exc:  # noqa: BLE001 - a lookup must never crash the run
            logger.warning("extract_data failed to snapshot the page: %s", exc)
            return ActionResult(error=f"extract_data: could not read page state: {exc}")
        tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
        if not tokens:
            return ActionResult(error=f"extract_data: no searchable text in {query!r}")

        value, element = "", None
        expanded = False
        matches = _matching_nodes(state, tokens)
        # A form control's snapshot text is NOT its value: a <select>'s children text is
        # every option label concatenated ("Random Male Female" for the gender chooser —
        # the observed live failure), an <input>'s is empty. Defer control matches to the
        # raw-DOM finders below, whose select branch reads the SELECTED option.
        matches = [m for m in matches
                   if str(getattr(m[1], "node_name", "") or "").lower()
                   not in ("select", "input", "textarea")]

        async def _raw_lookup(exprs: tuple[str, ...]):
            for expr in exprs:
                try:
                    found = await _eval_js(browser_session, expr)
                except Exception as exc:  # noqa: BLE001 - fallback is best-effort
                    logger.debug("extract_data raw-DOM fallback failed: %s", exc)
                    found = None
                if found and not found.get("error") and found.get("count"):
                    return found
            return None

        def _raw_capture(raw: dict) -> tuple[str, dict | None]:
            raw_value = " ".join(str(raw.get("name") or "").split())[:1000]
            raw_element = None
            el = raw.get("element") or {}
            if el.get("tag"):
                raw_element = {
                    "node_name": str(el.get("tag") or ""),
                    "attributes": dict(el.get("attrs") or {}),
                    "ax_name": " ".join(str(raw.get("name") or "").split()),
                }
                if el.get("xpath"):
                    # Positional anchor — the ONLY selector that survives on pages
                    # whose value (and therefore its text= selector) changes each run.
                    raw_element["x_path"] = str(el["xpath"])
            return raw_value, raw_element

        if matches:
            # Exact-label preference, as in find_by_text: with several candidates, the one
            # whose own label/aria-label/id equals the query is the intended target.
            chosen = matches
            if len(matches) > 1:
                wanted = _norm_phrase(query)
                exact = [m for m in matches
                         if _norm_phrase(m[2]) == wanted
                         or any(_norm_phrase((m[1].attributes or {}).get(a) or "") == wanted
                                for a in ("aria-label", "id", "name"))]
                if exact:
                    chosen = exact
            _idx, node, node_label = chosen[0]
            text_value = " ".join(node.get_all_children_text(max_depth=5).split())
            value = (text_value or node_label or "").strip()[:1000]
            element = _captured_element(node, node_label)
            if value and _norm_phrase(value) == _norm_phrase(query):
                # Zero information gained: the capture IS the query (a bare name in its
                # own heading — the caller wanted the card AROUND it). The static-text
                # finder expands exactly such matches to the enclosing block; replace the
                # capture only when it really grew, so an honest tight value never becomes
                # a miss.
                raw = await _raw_lookup((_RAW_TEXT_FIND_JS % json.dumps(tokens),))
                if raw and raw.get("expanded"):
                    value, element = _raw_capture(raw)
                    expanded = True
        else:
            # The value may live outside the interactive snapshot (plain text is not an
            # interactive element). Query the raw DOM directly, click disabled: first the
            # control-shaped finder, then the static-text finder — a value in a bare
            # <h3>/<div> (the fakenamegenerator identity block) is invisible to both the
            # snapshot and RAW_FIND_JS's control selector.
            raw = await _raw_lookup((_RAW_FIND_JS % (json.dumps(tokens), "false"),
                                     _RAW_TEXT_FIND_JS % json.dumps(tokens)))
            if raw:
                value, element = _raw_capture(raw)
                expanded = bool(raw.get("expanded"))

        if not value:
            # No metadata on a miss: a valueless extract must compile to NOTHING, not to a
            # step that would fail every replay.
            msg = (f"extract_data('{query}'): nothing matched on the CURRENT page, or the "
                   f"match has no visible text. The value was NOT captured. Scroll it into "
                   f"view or re-call extract_data with words that appear in or right next "
                   f"to the value.")
            logger.info("📋 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True)

        msg = f"extract_data('{query}'): {slug} = '{value}'"
        if expanded:
            msg += (" (the exact match was only the query text itself; captured its "
                    "surrounding block)")
        logger.info("📋 %s", msg)
        return ActionResult(
            extracted_content=msg, long_term_memory=msg, include_in_memory=True,
            metadata={"extract": {"label": slug, "value": value, "query": query,
                                  "interacted_element": element}},
        )

    @tools.action(
        "Ground truth for saves: check whether the record you tried to save actually reached the "
        "server in this run (a successful create-write request was observed on the network). Call "
        "this after clicking the final Save on a create form. CONFIRMED means the save is real; "
        "NOT REGISTERED means the Save did not go through — fix the form's validation errors and "
        "save again. Read-only — changes nothing."
    )
    async def verify_save_registered() -> ActionResult:
        if _SAVE_PROBE is None:
            msg = "Save verification is not available for this run; verify via the UI instead."
            return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)
        try:
            write = _SAVE_PROBE()
        except Exception as exc:  # noqa: BLE001 - a probe failure must never crash the run
            logger.warning("verify_save_registered probe failed: %s", exc)
            return ActionResult(error=f"verify_save_registered failed: {exc}")
        if write:
            url_tail = str(write.get("url", ""))[-80:]
            msg = (f"CONFIRMED: the save reached the server "
                   f"({write.get('method')} ...{url_tail}, agent step {write.get('step')}). Proceed.")
        else:
            msg = ("NOT REGISTERED: no create-write has hit the server — the Save did NOT go "
                   "through. The form almost certainly shows validation errors (required fields, "
                   "invalid values, missing item selection). Find the error messages on the form, "
                   "fix those exact fields, and save again. Do NOT report success until "
                   "this tool returns CONFIRMED.")
        logger.info("🧾 %s", msg.split(".")[0])
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    @tools.action(
        "Ground truth for downloads: list the files this browser session has downloaded. A "
        "click that starts a download often reports a TIMEOUT even though it worked — call "
        "this instead of clicking again. CONFIRMED + the expected filename means the "
        "download is real; NONE means no download has happened yet. Read-only."
    )
    async def verify_download(browser_session=None) -> ActionResult:  # injected by name; do not annotate
        try:
            files = list(getattr(browser_session, "downloaded_files", None) or [])
        except Exception as exc:  # noqa: BLE001 - a probe failure must never crash the run
            return ActionResult(error=f"verify_download failed: {exc}")
        if not files:
            msg = ("verify_download: NONE — no file download has been observed in this "
                   "session. The download control has not been successfully triggered yet.")
        else:
            names = ", ".join(Path(p).name for p in files[-5:])
            msg = (f"verify_download: CONFIRMED — {len(files)} file(s) downloaded this "
                   f"session (latest: {names}). If this covers the file your step needed, "
                   f"the download succeeded: do NOT click the download control again.")
        logger.info("📥 %s", msg.split(" If ")[0])
        return ActionResult(extracted_content=msg, long_term_memory=msg,
                            include_in_memory=True)

    @tools.action(
        "Scan the CURRENT page for layout bugs: sideways page overflow, visible elements spilling "
        "past the viewport edge, and clickable controls rendered at ~zero size. Returns the issues "
        "found (or reports none). Read-only — changes nothing."
    )
    async def detect_layout_issues(browser_session=None) -> ActionResult:  # injected by name; do not annotate
        try:
            raw = await _eval_js(browser_session, _LAYOUT_JS)
            data = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("detect_layout_issues failed: %s", exc)
            return ActionResult(error=f"detect_layout_issues failed: {exc}")
        if data.get("error"):
            return ActionResult(error=f"detect_layout_issues error: {data['error']}")
        issues = data.get("issues", [])
        if not issues:
            msg = "Layout scan: no issues detected."
        else:
            lines = [f"- {i['type']}: {i['detail']}" for i in issues]
            msg = "Layout scan found %d issue type(s):\n%s" % (len(issues), "\n".join(lines))
        logger.info("🧭 %s", msg.replace("\n", " | "))
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    @tools.action(
        "Run an accessibility (WCAG) scan of the CURRENT page with axe-core and return a summary "
        "of violations by rule (id, impact, affected node count). Read-only — changes nothing."
    )
    async def run_accessibility_scan(browser_session=None) -> ActionResult:  # injected by name; do not annotate
        try:
            axe_src = _AXE_PATH.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return ActionResult(error=f"could not load bundled axe-core: {exc}")
        try:
            # Inject axe only once per page context, then run it.
            already = await _eval_js(browser_session, "typeof window.axe !== 'undefined'")
            if not already:
                await _eval_js(browser_session, axe_src)
            run_js = (
                "(async function(){try{var r=await axe.run(document,{resultTypes:['violations']});"
                "return JSON.stringify({violations:r.violations.map(function(v){"
                "return {id:v.id,impact:v.impact,help:v.help,nodes:v.nodes.length};})});"
                "}catch(e){return JSON.stringify({error:String(e)});}})()"
            )
            raw = await _eval_js(browser_session, run_js, await_promise=True)
            data = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_accessibility_scan failed: %s", exc)
            return ActionResult(error=f"run_accessibility_scan failed: {exc}")
        if data.get("error"):
            return ActionResult(error=f"accessibility scan error: {data['error']}")
        violations = data.get("violations", [])
        if not violations:
            msg = "Accessibility scan (axe-core): no violations detected."
        else:
            violations.sort(key=lambda v: (v.get("nodes") or 0), reverse=True)
            lines = [
                f"- {v['id']} ({v.get('impact') or 'n/a'}): {v.get('nodes')} node(s) — {v.get('help')}"
                for v in violations[:15]
            ]
            extra = "" if len(violations) <= 15 else f"\n  ...and {len(violations) - 15} more rule(s)"
            msg = "Accessibility scan (axe-core) found %d rule violation(s):\n%s%s" % (
                len(violations), "\n".join(lines), extra,
            )
        logger.info("♿ %s", msg.split("\n")[0])
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    @tools.action(
        "List the clickable controls near a heading/row, INCLUDING unlabeled icon buttons whose "
        "meaning is hidden in a child icon (which normally show up as a nameless '<button/>'). "
        "Each result carries its decoded icon name and its click index. Use this when you need an "
        "icon whose tooltip/label is not findable by text (e.g. a 'send survey', 'edit', or "
        "'download' icon in a toolbar or table row) — call list_actions with the nearest visible "
        "heading or row text, read the decoded names, then click the matching index. Read-only."
    )
    async def list_actions(near_text: str = "", browser_session=None) -> ActionResult:  # injected by name; do not annotate
        if browser_session is None:
            return ActionResult(error="list_actions: BrowserSession not injected")
        try:
            state = await browser_session.get_browser_state_summary(include_screenshot=False)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(error=f"list_actions: could not read page state: {exc}")

        def _decoded(node: Any) -> str:
            attrs = node.attributes or {}
            own = " ".join((node.get_all_children_text(max_depth=5) or "").split())[:120]
            if not own:
                for attr in ("aria-label", "title", "name", "placeholder", "value"):
                    if attrs.get(attr):
                        own = str(attrs[attr])
                        break
            hints = _descendant_icon_hints(node)
            if hints and hints.lower() not in own.lower():
                return f"{own} [{hints}]".strip() if own else hints
            return own

        items = sorted(state.dom_state.selector_map.items())
        # Anchor to the heading/row text so the list is scoped to the right region. Indexes in
        # the selector map track document order, so a window around the landmark's index keeps
        # the controls that belong to that section.
        anchor = None
        near = (near_text or "").strip().lower()
        near_tokens = [t for t in re.split(r"[^a-z0-9]+", near) if t]
        if near_tokens:
            for idx, node in items:
                try:
                    hay = " ".join((node.get_all_children_text(max_depth=5) or "").split()).lower()
                    if all(t in hay for t in near_tokens):
                        anchor = idx
                        break
                except Exception:  # noqa: BLE001
                    continue

        window = 40
        rows: list[str] = []
        for idx, node in items:
            if anchor is not None and abs(idx - anchor) > window:
                continue
            try:
                name = _decoded(node)
            except Exception:  # noqa: BLE001
                continue
            if not name:
                continue
            tag = getattr(node, "tag_name", None) or (getattr(node, "node_name", "") or "").lower()
            rows.append(f"index={idx} <{tag}> '{name[:70]}'")
            if len(rows) >= 30:
                break

        where = f" near '{near_text}'" if near_text else ""
        if not rows:
            msg = (f"list_actions{where}: no named controls found"
                   + (" — is this the right page/section? try a different nearby heading."
                      if near_tokens else "."))
            logger.info("🧭 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)
        msg = (f"Clickable controls{where} (decoded icon names in [brackets]):\n"
               + "\n".join(rows)
               + "\nClick your target with click(index) — indexes are fresh but go stale on re-render.")
        logger.info("🧭 list_actions%s: %d control(s)", where, len(rows))
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    return tools
