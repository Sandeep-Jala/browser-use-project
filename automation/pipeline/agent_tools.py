"""Custom agent tools the prompts rely on, registered on a browser-use ``Tools`` registry.

The system/expander prompts (see prompts.py) instruct the agent to call custom actions that are
NOT part of browser-use's built-in set:

  * skip_step(reason)          — escape hatch: abandon the current objective, keep going.
  * fail_and_stop(reason)      — escape hatch: terminate the whole run as a failure.
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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from browser_use import Tools
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.events import (ClickElementEvent, SelectDropdownOptionEvent,
                                        SendKeysEvent, TypeTextEvent)
from browser_use.dom.views import DOMInteractedElement
from browser_use.tools.views import InputTextAction, SelectDropdownOptionAction

from automation.pipeline.script_compile import (
    DIALOG_COUNT_JS as _DIALOG_COUNT_JS,
    DIALOG_STAMP_JS as _DIALOG_STAMP_JS,
    DIALOG_STAMPED_OPEN_JS as _DIALOG_STAMPED_OPEN_JS,
    FIELD_REFIND_JS as _FIELD_REFIND_JS,
    RAW_FIND_JS as _RAW_FIND_JS,
    RAW_TEXT_FIND_JS as _RAW_TEXT_FIND_JS,
    GROUP_CLEAR_JS as _GROUP_CLEAR_JS,
    GROUP_VALUES_JS as _GROUP_VALUES_JS,
    PASTE_EVENT_JS as _PASTE_EVENT_JS,
    _PASTE_MODIFIER,
    SCROLL_CONTAINERS_JS as _SCROLL_CONTAINERS_JS,
    SCROLL_TOPS_JS as _SCROLL_TOPS_JS,
    paste_group_empty as _paste_group_empty,
    paste_took as _paste_took,
    normalize_block_text,
    CALLOUT_OPEN_FN_JS as _CALLOUT_OPEN_FN_JS,
    CALLOUT_SCROLL_PIN_JS as _CALLOUT_SCROLL_PIN_JS,
    REVEAL_CSS_JS as _REVEAL_CSS_JS,
    _RS_FILTER_ID,
    value_took as _value_took,
)

logger = logging.getLogger("framework.tools")

# Vendored axe-core, loaded once and injected into the page on demand.
_AXE_PATH = Path(__file__).resolve().parent.parent / "assets" / "axe.min.js"

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


# How far up the parent chain a wrapper duplicate can sit. A control wrapped in its own
# padding/anchor/label divs is a handful of hops; anything deeper is a real container
# that happens to hold nothing but this control, and collapsing it buys nothing.
_NEST_WALK_DEPTH = 12


def _node_key(node: Any) -> Any:
    """Identity for comparing a snapshot node against a parent-chain node. The DOM ids
    are the same objects in both, so `id()` suffices; backend_node_id is preferred when
    present because it survives a re-wrapped node."""
    backend = getattr(node, "backend_node_id", None)
    return backend if backend is not None else id(node)


def _collapse_nested_duplicates(
        matches: list[tuple[int, Any, str]]) -> list[tuple[int, Any, str]]:
    """Drop WRAPPER duplicates: when one candidate is an ancestor of another and both
    carry the same normalized text, they are ONE control counted twice, not a choice.

    Run 20260824_155123 seg 2: the app's "Get OTP" is a <div> holding a <div> with the
    same text, so find_by_text('Get OTP', click_first=true) saw 2 candidates every time
    and refused to click — the OTP was never fetched. Equal LABELS is what makes the
    collapse safe: _matching_nodes labels an ancestor with all of its descendants' text,
    so a row/section holding the control plus anything else reads differently and is
    left alone.

    Keeps the DEEPEST node of each chain: a click there bubbles up to every ancestor's
    handler, while a click on the wrapper can miss a handler that lives on the inner
    control (the <li>-around-<a> case the exact-label preference below was written for).
    """
    by_key = {_node_key(node): label for _, node, label in matches}
    # Every candidate that some OTHER candidate sits inside, with the same text: those
    # are the wrappers, and the one nobody wraps is the innermost control.
    wrappers: set[Any] = set()
    for _idx, node, label in matches:
        wanted = _norm_phrase(label)
        parent = getattr(node, "parent_node", None)
        for _ in range(_NEST_WALK_DEPTH):
            if parent is None:
                break
            key = _node_key(parent)
            if key in by_key and _norm_phrase(by_key[key]) == wanted:
                wrappers.add(key)
            parent = getattr(parent, "parent_node", None)
    kept = [m for m in matches if _node_key(m[1]) not in wrappers]
    return kept or matches


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


async def ensure_callout_scroll_pin(browser_session: BrowserSession | None) -> None:
    """Install the callout scroll pin (script_compile.CALLOUT_SCROLL_PIN_JS, idempotent)
    into the agent's CURRENT document.

    A Fluent Callout dismisses itself when anything OUTSIDE it scrolls, and the scroll that
    kills it is usually not ours: a control sitting PARTLY outside the viewport bounds is
    still visible and clickable, so the click has to scroll it into view to reach it, and
    the popup that click opened dies on that movement (user, 2026-08-24). The pin reverts
    such a scroll and swallows the event so the popup never learns of it.

    Same shape as ensure_reveal_css: the init scripts in login.py/HybridSession.open cover
    documents born inside those contexts, and this per-step pass heals what they cannot
    reach — a tab browser-use creates via CDP outside them. Ungated and best-effort: a
    popup that vanishes loses the value being typed, so this is correctness, but it must
    never be able to break a step either."""
    if browser_session is None:
        return
    try:
        await _eval_js(browser_session, _CALLOUT_SCROLL_PIN_JS)
    except Exception as exc:  # noqa: BLE001 - the init scripts are the primary path
        logger.debug("ensure_callout_scroll_pin skipped: %s", exc)


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


def _control_name(node: Any, index: int | None = None) -> str:
    """A human name for a control, for receipts.

    The accessible name lives at `node.ax_node.name` — NOT `node.ax_name`, which does not
    exist on a live EnhancedDOMTreeNode (see _captured_element below, which reads it
    correctly). repeat_click read the wrong attribute and so reported "Clicked element 1063
    5 times" for a button whose captured metadata said "Save & Next"; a receipt that cannot
    name what it acted on is exactly what lets an agent lose track of what it has already
    done, and run 20260901_122209 then clicked Save & Next sixteen times for a slice asking
    five."""
    ax = getattr(getattr(node, "ax_node", None), "name", None)
    if str(ax or "").strip():
        return str(ax).strip()
    attrs = getattr(node, "attributes", None) or {}
    for attr in ("aria-label", "title", "name", "id"):
        if str(attrs.get(attr) or "").strip():
            return str(attrs[attr]).strip()
    text = str(getattr(node, "node_value", "") or "").strip()
    return text or (f"element {index}" if index is not None else "the control")


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


# Scroll rounds find_by_text spends opening up panels/dialog lists whose rows are
# rendered only near their own scroll position (Fluent virtualized ScrollablePane).
# Cap on find_by_text's container-scroll fallback rounds. 6 ran out MID-LIST once the
# Add Data Request employee list grew past ~60 rows (every run appends an employee; run
# 20260817_110501 exited partway down ~80 rows and honestly reported "no match"). The
# loop breaks the moment nothing moves (moved==0), so a high cap costs nothing on short
# lists — it only ever runs to the actual end of the content.
_PANEL_SCROLL_ROUNDS = 20

# The beat a scroll needs before the DOM is worth reading again. Setting scrollTop fires the
# app's scroll handler, which sets React state, which MOUNTS the newly revealed rows — on a
# later frame. Read in the same tick and the query answers about the previous render window.
#
# Run 20260903_110957_989862 subtask 6: the noted employee WAS in the Data Request list, and
# find_by_text's hunt swept the whole thing without seeing them, because it re-queried
# immediately after every scroll. The step size was never the problem (0.8 of the container
# leaves 20% overlap; nothing is skipped visually) — the READ was.
#
# 400ms is not a new number: script_compile's _scroll_containers, _scroll_tops and
# _wheel_scroll all wait exactly this, and the scroll_panels tool had its own inline copy.
# The rule was in the codebase four times and the one path that hunts for a NAME in a
# virtualized list still missed it, which is why scrolling and settling are now a single
# helper that both callers go through.
_SCROLL_SETTLE_S = 0.4


async def _scroll_and_settle(browser_session, fraction: float = 0.8) -> int:
    """Advance every scrollable container, then WAIT for the rows to mount. Returns how many
    scrollers moved — 0 means nothing left to scroll, which is how a sweep knows to stop.

    Best-effort like every other scroll path: an unscrollable/unreachable page returns 0 and
    ends the sweep rather than failing the lookup that called it."""
    try:
        moved = int(await _eval_js(
            browser_session, _SCROLL_CONTAINERS_JS % json.dumps(float(fraction))) or 0)
    except Exception as exc:  # noqa: BLE001 - a scroll must never crash the run
        logger.debug("container scroll failed: %s", exc)
        return 0
    await asyncio.sleep(_SCROLL_SETTLE_S)
    return moved


async def _scroll_tops_and_settle(browser_session) -> int:
    """Reset page + containers to the top, then wait for the rows to mount. The hunt
    companion to _scroll_and_settle, and it needs the same beat for the same reason: a sweep
    that starts by resetting and reads immediately reads the PRE-reset window."""
    try:
        moved = int(await _eval_js(browser_session, _SCROLL_TOPS_JS) or 0)
    except Exception as exc:  # noqa: BLE001 - a scroll must never crash the run
        logger.debug("scroll-to-top failed: %s", exc)
        return 0
    await asyncio.sleep(_SCROLL_SETTLE_S)
    return moved


# A combobox whose popup is a dialog is a date/time picker (Fluent DatePicker), not an
# option filter: typed text IS its value and reads back verifiably.
def _is_date_picker(node: Any) -> bool:
    attrs = getattr(node, "attributes", None) or {}
    return (attrs.get("aria-haspopup") or "").strip().lower() == "dialog"


# A dropdown/combobox filter input: Enter there commits the focused option instead of
# submitting a search, so the auto-Enter `input` replacement must not fire it.
def _is_dropdown_filter(node: Any) -> bool:
    attrs = getattr(node, "attributes", None) or {}
    # Date pickers must never get the filter refusal (run 20260807_093003: the DOB
    # refusal forced calendar navigation to 1982 and killed the run) — but they keep
    # their own Enter suppression via _is_date_picker (see the fill path).
    if _is_date_picker(node):
        return False
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


# What copy_text last captured. paste_text with no `text` delivers this — the pair the
# agent reaches for when a value shown on one page must be entered on another.
_CLIPBOARD: dict[str, str] = {"value": "", "label": ""}

async def _paste_reading(handle) -> dict[str, Any] | None:
    """The target's own value plus its input GROUP's values, or None when unreadable.
    See script_compile.GROUP_VALUES_JS for why the group matters."""
    try:
        return await _call_on_field(
            handle, "function(){ return (%s)(this); }" % _GROUP_VALUES_JS)
    except Exception as exc:  # noqa: BLE001 - an unreadable widget is not a failed paste
        logger.debug("paste read-back failed: %s", exc)
        return None


async def _clear_paste_group(handle) -> None:
    """Empty the target and its input group between rungs. A rung that lands PART of the
    value poisons the next one: measured in chromium, Chrome's own paste command puts a
    truncated "5" into a maxlength=1 box and distributes nothing, after which typing
    appends to a full box. See script_compile.GROUP_CLEAR_JS for why a plain
    `node.value = ''` will not do."""
    try:
        await _call_on_field(handle, "function(){ return (%s)(this); }" % _GROUP_CLEAR_JS)
    except Exception as exc:  # noqa: BLE001 - the read-back still judges what happened
        logger.debug("paste group clear failed: %s", exc)


async def _clipboard_write(browser_session, text: str) -> bool:
    """Put `text` on the real clipboard (best-effort). browser-use's profile already grants
    clipboardReadWrite; the write still rejects when the document is not focused, which is
    why every caller treats False as a degraded-but-fine outcome."""
    try:
        return bool(await _eval_js(
            browser_session,
            "navigator.clipboard.writeText(%s).then(function(){return true;},"
            "function(){return false;})" % json.dumps(text),
            await_promise=True))
    except Exception as exc:  # noqa: BLE001 - clipboard is a convenience, never a gate
        logger.debug("clipboard write failed: %s", exc)
        return False


async def _native_paste(handle) -> bool:
    """Rung 2: ask Chrome to execute its own paste editing command, which produces a
    genuine isTrusted paste event from the real clipboard — for widgets that ignore the
    synthetic one. Requires the element to hold focus (rung 1 focused it)."""
    cdp_session, _object_id = handle
    try:
        for event_type in ("keyDown", "keyUp"):
            params: dict[str, Any] = {
                "type": event_type, "key": "v", "code": "KeyV",
                "windowsVirtualKeyCode": 86, "modifiers": _PASTE_MODIFIER,
            }
            if event_type == "keyDown":
                params["commands"] = ["paste"]
            await cdp_session.cdp_client.send.Input.dispatchKeyEvent(
                params=params, session_id=cdp_session.session_id)
        return True
    except Exception as exc:  # noqa: BLE001 - fall through to the keystroke rung
        logger.debug("native paste command failed: %s", exc)
        return False


async def _field_connected(handle) -> bool | None:
    """Whether the element is still attached to a live document. False for a node an
    earlier action re-rendered away: its CDP object id keeps resolving, so fills and
    read-backs against it look normal while the USER-visible field never changes. None
    when the element can't be asked — callers then trust the normal path."""
    try:
        return bool(await _call_on_field(handle, "function(){ return this.isConnected; }"))
    except Exception as exc:  # noqa: BLE001 - an unprobeable node is not proof of staleness
        logger.debug("connectivity probe failed: %s", exc)
        return None


async def _in_layer_popup(handle) -> bool:
    """Is the field inside a transient Fluent layer/callout/dialog? Enter there risks
    dismiss/reset side effects — observed run 20260813_123549: auto-Enter re-rendered
    the Salary-to-take-home callout and the batched Calculate click dispatched onto a
    detached node. The popup's own confirm button is the commit. False when
    unprobeable: the normal Enter path stays the default, and the dropdown/date-picker
    rules run before this one."""
    try:
        return bool(await _call_on_field(
            handle,
            "function(){ return !!(this.closest && this.closest("
            "'.ms-Layer, #fluent-default-layer-host, [role=\"dialog\"]')); }"))
    except Exception as exc:  # noqa: BLE001 - unprobeable is not proof of a popup
        logger.debug("layer-popup probe failed: %s", exc)
        return False


# Document-level twin of _in_layer_popup, narrowed to the DISMISS-ON-SCROLL family:
# Fluent Callouts (and the ContextualMenus/pickers built on them) close themselves when
# anything outside scrolls. Panels and Modals do NOT — and must stay out of this probe:
# the Add Data Request side PANEL owns the employee list whose rows only container
# scrolling reveals (runs 20260814_105247 / 20260817_110501 / 20260817_133135), so a
# guard that counted panels would kill the hunt those runs exist to protect.
#
# The predicate itself comes from script_compile.CALLOUT_OPEN_FN_JS, shared with the scroll
# pin (the protection) and RAW_FIND_JS (which skips its own scrollIntoView). This probe
# drives the ADVICE half — refusing page-moving tools and gating the text hunt, so the agent
# is told a scroll is pointless instead of silently getting a no-op. Advice and protection
# only stay consistent while they ask the same question, and three hand-copied loops did not.
_CALLOUT_OPEN_JS = """
(function () {
  try {
    return { open: (__CALLOUT_OPEN_FN__)() ? 1 : 0 };
  } catch (e) { return { error: String(e) }; }
})()
""".replace("__CALLOUT_OPEN_FN__", _CALLOUT_OPEN_FN_JS)


async def _callout_open(browser_session) -> bool:
    """Is a dismiss-on-scroll popup on screen right now? False when unprobeable — the
    scroll hunt stays the default rather than being suppressed by a broken probe."""
    try:
        got = await _eval_js(browser_session, _CALLOUT_OPEN_JS)
    except Exception as exc:  # noqa: BLE001 - a probe must never fail the lookup
        logger.debug("callout probe failed: %s", exc)
        return False
    return bool(isinstance(got, dict) and not got.get("error") and got.get("open"))


# How many times each control has been clicked in THIS segment, and the number of repeats
# the segment's own wording declares ("exactly 5 more clicks" -> 5, parsed by
# script_compile.repeat_hint_from_wording, the same value the compiler already uses).
#
# Why a ledger at all: repeat_click counts WITHIN a call, but nothing stopped the agent
# calling it again. Run 20260901_122209 subtask 15 asked for "exactly 5 more clicks"; the
# agent clicked once by hand, then called repeat_click(times=5) THREE times — each eval
# saying the previous repeat had succeeded, then re-forming the same goal — for 16 clicks and
# 26 payroll writes where the task wanted 11. The prompt already says to call it ONCE
# (prompts.py) and that was not enough, so the budget is enforced rather than advised.
#
# NOTE this reads a NUMBER THE USER WROTE, which is not the wording inference banned on
# 2026-08-28: that rule forbids deciding a subtask's KIND or cacheability from prose. A
# declared count is a declaration.
_CLICK_LEDGER: dict[str, int] = {}
_REPEAT_BUDGET: int | None = None
# Controls repeat_click has actually run a repetition on THIS segment. The budget is one
# segment-wide number while the ledger is keyed per control, so it may only be enforced
# against a control a repetition has already established it is about — capping every other
# button in the segment at the same N would be an over-reach.
_REPEAT_TARGETS: set[str] = set()


def set_repeat_budget(budget: int | None) -> None:
    """Declare this segment's repeat count (runner-owned, once per segment)."""
    global _REPEAT_BUDGET
    _REPEAT_BUDGET = int(budget) if budget else None


def _ledger_key(node: Any) -> str:
    """Stable-ish identity for a control across re-renders. The id attribute is the best
    key this app offers (btnSave survives the row swapping behind it); the accessible name
    is the fallback, and both beat a backend node id, which changes every render."""
    attrs = getattr(node, "attributes", None) or {}
    for attr in ("id", "data-automationid", "aria-label", "name"):
        if str(attrs.get(attr) or "").strip():
            return f"{attr}={str(attrs[attr]).strip()}"
    return f"name={_control_name(node)}"


def _note_clicks(node: Any, n: int = 1) -> int:
    """Add `n` clicks for `node` to this segment's ledger and return its new total."""
    try:
        key = _ledger_key(node)
    except Exception:  # noqa: BLE001 - accounting must never break a click
        return 0
    _CLICK_LEDGER[key] = _CLICK_LEDGER.get(key, 0) + int(n)
    return _CLICK_LEDGER[key]


def _refuse_if_over_budget(node: Any, label: str) -> "ActionResult | None":
    """Refuse a PLAIN click that would exceed the segment's declared repeat count, or None
    to let it through. Only ever fires on a control repeat_click has already repeated (see
    _REPEAT_TARGETS).

    Why plain clicks need this at all: repeat_click's own guard is the one that stopped run
    20260901_122209, but it only covers repeat_click. Run 20260902_105732 subtask 3 went the
    other way round — repeat_click(5) landed all five, browser-use silently DROPPED the
    `done` the agent had batched behind it (agent/service.py: "Done action is allowed only
    as a single action"), and on the unexplained extra turn the agent read the correct end
    state as a failure and clicked Save & Next five more times BY HAND. Ten employees were
    paid for a five-employee slice, and the segment cached repeat_click(9). Obeying the
    letter of "do NOT call repeat_click again" while redoing the work with `click` is the
    hole this closes.

    ERROR channel + no_click, like every other refusal here: the error stops the rest of a
    batched step, and no_click keeps the phantom out of the recording."""
    if _REPEAT_BUDGET is None:
        return None
    try:
        key = _ledger_key(node)
    except Exception:  # noqa: BLE001 - a guard must never break a click
        return None
    if key not in _REPEAT_TARGETS:
        return None
    already = _CLICK_LEDGER.get(key, 0)
    if already < _REPEAT_BUDGET:
        return None
    msg = (f"click REFUSED — did NOT click: {label} has already been clicked {already} "
           f"time(s) in this step, which is the {_REPEAT_BUDGET} this step asks for. The "
           f"repetition is COMPLETE and those clicks all landed. Do NOT click it again by "
           f"hand — verify the end state on the page and call done.")
    logger.info("⛔ %s", msg)
    return ActionResult(error=msg, metadata={"no_click": True})


# repeat_click's cadence (the live twin of skills/api.py's constants). Between clicks the
# control must be clickable AGAIN before the next one, so a slow employee load delays the
# cadence instead of eating a click. The hard cap bounds the until-it-stops mode: reaching
# it means the control was STILL advancing, which is reported as a failure rather than as
# a finished list.
_REPEAT_SETTLE_S = 0.4
_REPEAT_READY_CAP_S = 10.0
_REPEAT_POLL_S = 0.25
_REPEAT_HARD_CAP = 200

# Is the just-clicked control ready to be clicked again? Runs ON the node, so it survives a
# re-render that keeps the same element (React reuses the button and swaps the row behind
# it) and reports honestly when the node goes away — which is how the end of a list looks.
_REPEAT_READY_JS = """function () {
  if (!this.isConnected) return {ready: false, why: 'the control left the page'};
  if (this.disabled || this.getAttribute('aria-disabled') === 'true')
    return {ready: false, why: 'the control became disabled'};
  var r = this.getBoundingClientRect();
  if (!(r.width > 0 && r.height > 0))
    return {ready: false, why: 'the control is no longer visible'};
  try {
    var cs = getComputedStyle(this);
    if (cs && (cs.visibility === 'hidden' || cs.display === 'none'))
      return {ready: false, why: 'the control was hidden'};
  } catch (e) {}
  return {ready: true, why: ''};
}"""


# The control's OWN name, read in the page. Visible text first: that is what the task and
# the user call the control by ("click Next", "then click submit"), and it is what changes
# when an app swaps a button's job on the last row. PUA glyphs are stripped because Fluent
# renders icons as literal text nodes inside the control (see _CAND_ROW_NAME_JS).
_CONTROL_NAME_JS = """function () {
  var t = function (s) { return (s || '').replace(/[\\uE000-\\uF8FF]/g, ' ')
                                         .replace(/\\s+/g, ' ').trim(); };
  var text = t(this.innerText || this.textContent);
  if (text) return text;
  var g = this.getAttribute ? this.getAttribute.bind(this) : function () { return ''; };
  return t(g('aria-label')) || t(g('title')) || t(this.value) || '';
}"""


async def _repeat_live_name(browser_session, node) -> str:
    """One in-page read of the control's name, or "" when it cannot be read.

    Deliberately unpolled and fail-open: this feeds a STOP decision, and a name that
    cannot be read is not evidence of anything.
    """
    try:
        handle = await _field_handle(browser_session, node)
        if not handle:
            return ""
        return str(await _call_on_field(handle, _CONTROL_NAME_JS) or "").strip()
    except Exception as exc:  # noqa: BLE001 - unreadable name never stops a healthy loop
        logger.debug("repeat_click: could not read the control's name (%s)", exc)
        return ""


# How many candidates get a row probe. Each is one CDP round trip, and a listing shows at
# most 25 — past that the agent is being told to narrow its text, not to read more rows.
_ROW_PROBE_CAP = 30


def _pua_strip(text: Any) -> str:
    """Readable text: Private Use Area glyphs dropped, whitespace collapsed. Case KEPT.

    Fluent renders its icons as literal PUA TEXT NODES (the 2026-08-21 pencil trap, where a
    has-text guard disabled the very reader written for that cell), so any text read off this
    app's DOM can carry them. One definition, used wherever such text is shown or compared."""
    return re.sub(r"\s+", " ", re.sub(r"[\uE000-\uF8FF]", " ", str(text or ""))).strip()


def _pua_norm(text: Any) -> str:
    """_pua_strip, casefolded — the comparable form."""
    return _pua_strip(text).casefold()


def _repeat_norm_name(name: str) -> str:
    return _pua_norm(name)


def _row_matches(row_text: Any, tokens: list[str]) -> bool:
    """Does a candidate's row carry every token of `near_text`?

    Token containment, the same rule `text` itself uses \u2014 so 'feb 27' and 'Feb-27' agree and
    punctuation never decides a row. Empty/unreadable row text is False, never True: see
    find_by_text's near_text branch for why this fails CLOSED."""
    if not tokens:
        return False
    hay = _pua_norm(row_text)
    return bool(hay) and all(t in hay for t in tokens)


async def _row_labels(browser_session, matches: list) -> dict[int, str]:
    """{index: row text} for candidates, best-effort and concurrent.

    Reuses _row_context_label \u2014 the probe built so an anonymous checkbox click could name
    the employee row it ticked. A candidate whose row cannot be read is simply absent:
    callers treat it as garnish (the listing) or as a non-match (near_text scoping)."""
    async def one(idx: int, node: Any) -> tuple[int, Any]:
        return idx, await _row_context_label(browser_session, node)

    out: dict[int, str] = {}
    try:
        results = await asyncio.gather(
            *(one(idx, node) for idx, node, _ in matches[:_ROW_PROBE_CAP]),
            return_exceptions=True)
    except Exception as exc:  # noqa: BLE001 - row identity must never break a lookup
        logger.debug("row-label probe failed: %s", exc)
        return out
    for res in results:
        if isinstance(res, tuple) and res[1]:
            text = _pua_strip(res[1])
            if text:
                out[res[0]] = text
    return out


def _repeat_relabelled(before: str, after: str) -> bool:
    """True when the repeated control has become a DIFFERENT control.

    _REPEAT_READY_JS checks connected/disabled/visible and never the name — by design,
    so the loop survives the re-render that swaps the row behind a reused button. That
    same tolerance is what let run 20260901_165748 click Submit as iteration 12 of a
    Next loop: the app relabels the one button on the last employee. 11 Next + 1 Submit
    was recorded as `repeat_until_done('next')` — one step, no submit in it anywhere.

    Both names must be readable before this may stop anything: a blank is not evidence,
    and truncating a healthy run is worse than one swallowed click.
    """
    a, b = _repeat_norm_name(before), _repeat_norm_name(after)
    return bool(a and b and a != b)


async def _repeat_ready(browser_session, node) -> tuple[bool, str]:
    """(ready, why-not) for the control repeat_click just clicked, polled up to
    _REPEAT_READY_CAP_S. Unprobeable counts as NOT ready with the reason said plainly: a
    repeat that cannot verify its own target must stop and report, never keep clicking."""
    deadline = time.monotonic() + _REPEAT_READY_CAP_S
    why = "the control could not be probed"
    while time.monotonic() < deadline:
        try:
            handle = await _field_handle(browser_session, node)
            if handle:
                got = await _call_on_field(handle, _REPEAT_READY_JS)
                if isinstance(got, dict):
                    if got.get("ready"):
                        return True, ""
                    why = str(got.get("why") or why)
        except Exception as exc:  # noqa: BLE001 - a stale node reads as not-ready
            why = f"the control could not be probed ({exc})"
        await asyncio.sleep(_REPEAT_POLL_S)
    return False, why


# Keys that scroll the PAGE. Escape/Tab/Enter/arrows/Space are deliberately absent: they
# are how the agent closes a popup, moves between its fields and walks a combobox INSIDE
# it, and inside a text box they type or move the caret. Refusing those would trap the
# agent in the popup the refusal itself tells it to close.
_PAGE_SCROLL_KEYS = frozenset({"pagedown", "pageup"})


async def _refuse_if_callout(browser_session, action: str):
    """Refuse a page-MOVING action while a dismiss-on-scroll popup is open; None when the
    action may proceed.

    A Fluent Callout closes itself the moment anything outside it scrolls (the user's own
    words, 2026-08-18: "if you click on it and then scroll or do something, the popup
    vanishes"). find_by_text's hunt was gated on 08-17; this is the same gate for every
    remaining way the page can move. It reuses `_callout_open` unchanged, so Panels and
    Modals — the Add Data Request employee list, whose rows ONLY container scrolling
    reveals — never trip it, and an unprobeable page still scrolls.

    ERROR channel on purpose (the `input` dropdown refusal's rule): multi_act stops the
    rest of a batched step, so a queued click cannot fire against a popup the refused
    scroll would have closed. The message must name the way OUT or the agent deadlocks.
    """
    if not await _callout_open(browser_session):
        return None
    msg = (f"REFUSED — did NOT {action}: a POPUP is open on this page and it DISMISSES "
           "ITSELF when anything outside it scrolls. A popup owns the screen, so nothing "
           "it holds is behind a scroll: its fields are already in the page state — read "
           "them by index and act on them there. If you genuinely need the page BEHIND "
           "it, close the popup FIRST (press Escape, or click its own Cancel/close "
           "button), then scroll. Do NOT re-click the control that opened the popup "
           "either — that dismisses it too.")
    logger.info("⛔ %s", msg)
    return ActionResult(error=msg, metadata={"no_scroll": True})


_FIELD_LABEL_JS = """function(){
  var t = function(s){ return String(s || '').replace(/\\s+/g, ' ').trim(); };
  var el = this, label = '';
  var ids = el.getAttribute ? t(el.getAttribute('aria-labelledby')) : '';
  if (ids) label = t(ids.split(' ').map(function(id){
    var n = document.getElementById(id); return n ? t(n.textContent) : ''; }).join(' '));
  if (!label && el.labels && el.labels.length) label = t(el.labels[0].textContent);
  if (!label && el.closest) { var l = el.closest('label'); if (l) label = t(l.textContent); }
  var value = '';
  var box = el.closest ? el.closest('[class*="container"], [role="combobox"]') : null;
  if (box) {
    var v = box.querySelector(
      '[class*="single-value"], [class*="singleValue"], [class*="placeholder"]');
    value = t(v ? v.textContent : '') || t(box.textContent);
  }
  return {label: label.slice(0, 80), value: value.slice(0, 60)};
}"""


async def _dropdown_descriptor(browser_session, node) -> str | None:
    """Short identity for a combobox a fill just got refused on — "'Tax year' — currently
    showing '2026/27'". The refusal's remedy re-advertises the SAME index+text, which is
    right when only the verb was wrong; when the TARGET was wrong, nothing in the message
    let the agent notice (run 20260817_093555: three refusals kept pointing the agent
    back at the tax-year box while the task wanted a grid row — react-select inputs
    carry no label-ish attributes, so the agent's own find_elements sweeps could not
    tell the three comboboxes apart either). Best-effort: label-ish attributes first,
    then a live probe for the associated <label>/aria-labelledby text and the widget's
    visible value; None degrades to the bare message."""
    attrs = getattr(node, "attributes", None) or {}
    label = next((str(attrs[a]).strip() for a in ("aria-label", "title", "placeholder")
                  if str(attrs.get(a) or "").strip()), None)
    value = None
    try:
        handle = await _field_handle(browser_session, node)
        if handle:
            got = await _call_on_field(handle, _FIELD_LABEL_JS)
            if isinstance(got, dict):
                label = label or (str(got.get("label") or "").strip() or None)
                value = str(got.get("value") or "").strip() or None
    except Exception as exc:  # noqa: BLE001 - identity is garnish; the refusal must fire
        logger.debug("dropdown descriptor probe failed: %s", exc)
    if label and value and label != value:
        return f"'{label}' — currently showing '{value}'"
    if label:
        return f"'{label}'"
    return f"currently showing '{value}'" if value else None


# Identity ladder for re-finding a detached field's live twin; first attr the stale node
# actually carries wins. Class is deliberately absent (framework hashes churn per render).
_REFIND_IDENTITY_ATTRS = ("id", "name", "placeholder", "aria-label")


async def _refind_fill(browser_session, node, params: InputTextAction) -> ActionResult:
    """Land a fill whose target index went STALE mid-step on the live twin instead.

    The observed loop (run 20260805_131827_339055): dropdown picks batched before the
    fill re-mounted the modal, the fill's node detached, keystrokes went to whatever held
    focus, and the dead-node read-back produced a false "did NOT take" warning that drove
    ten duplicate saves. Here the CURRENT element with the same tag + identity attribute
    is found (composed-tree walk), focused, and filled via CDP insertText, with the same
    value_took read-back — or the action refuses honestly with the no_fill stamp. The
    recorded interacted_element keeps the stale node's attrs, which are the twin's too,
    so compiled replays are unaffected."""
    attrs = getattr(node, "attributes", None) or {}
    tag = str(getattr(node, "node_name", "") or "").lower()
    ident = next(((a, str(attrs.get(a)).strip()) for a in _REFIND_IDENTITY_ATTRS
                  if str(attrs.get(a) or "").strip()), None)
    stale = (f"STALE INDEX — element {params.index} was re-rendered away by an earlier "
             "action in this step and is no longer in the document; nothing was typed "
             "(keystrokes to a dead element land in whatever holds focus). Re-read the "
             "page and retype at the FRESH index — do not batch fills into the same step "
             "as dropdown picks.")
    if not ident or not tag:
        # All refusals below ride the error channel: multi_act stops the remaining
        # queued actions of this step on it, so nothing (a Save, an Enter, another
        # fill) executes on top of a fill that never landed.
        return ActionResult(error=stale, metadata={"no_fill": True})
    attr, value = ident

    def _expr(op: str) -> str:
        return _FIELD_REFIND_JS % {
            "tag": json.dumps(tag), "attr": json.dumps(attr), "value": json.dumps(value),
            "op": json.dumps(op), "clear": json.dumps(bool(params.clear))}

    try:
        found = await _eval_js(browser_session, _expr("focus"))
    except Exception as exc:  # noqa: BLE001 - refusal is the safe degradation
        logger.debug("stale-fill re-find failed: %s", exc)
        found = None
    if not (isinstance(found, dict) and found.get("count") == 1):
        n = found.get("count") if isinstance(found, dict) else None
        msg = stale if not n else \
            stale + f" ({n} candidates share {attr}='{value}' — cannot pick one safely.)"
        logger.info("⛔ %s", msg)
        return ActionResult(error=msg, metadata={"no_fill": True})
    label = str(found.get("label") or value)
    if (params.text or "").strip() and \
            _is_dropdown_filter(SimpleNamespace(attributes=dict(found.get("attrs") or {}))):
        # The refusal must hold through re-resolution too, or staleness becomes a side
        # door into exactly the typed-filter no-op the refusal exists to kill.
        msg = (f"REFUSED — did NOT type '{params.text}': element {params.index} "
               f"re-rendered into a dropdown/combobox filter ('{label}'), and typed "
               "filter text selects NOTHING. Re-read the page and call "
               f"select_dropdown(index=<fresh index>, text='{params.text}') instead.")
        logger.info("⛔ %s", msg)
        return ActionResult(error=msg, metadata={"no_fill": True})
    try:
        cdp_session = await browser_session.get_or_create_cdp_session()
        await cdp_session.cdp_client.send.Input.insertText(
            params={"text": params.text}, session_id=cdp_session.session_id)
        got = await _eval_js(browser_session, _expr("read"))
    except Exception as exc:  # noqa: BLE001 - report; the agent recovers via its receipt
        msg = stale + f" (typing into the live '{label}' field failed: {exc})"
        return ActionResult(error=msg, metadata={"no_fill": True})
    verified = None
    if isinstance(got, dict) and got.get("value") is not None:
        verified = str(got["value"])
    if verified is None or not _value_took(params.text, verified):
        shows = "(unreadable)" if verified is None else f"'{verified}'"
        msg = (f"element {params.index} had re-rendered (stale index); re-typed "
               f"'{params.text}' into the live '{label}' field but it now reads {shows} "
               "— the value did NOT take. Do NOT report this field as set; re-read the "
               "page and retype at the fresh index.")
        logger.warning("⚠️ %s", msg)
        return ActionResult(error=msg, metadata={"no_fill": True})
    if _is_date_picker(SimpleNamespace(attributes=dict(found.get("attrs") or {}))):
        # Same date-picker rule as the main fill path: no Enter, blur commits.
        meta = {"auto_enter": False}
    else:
        meta = {"auto_enter": True}
        try:
            enter = browser_session.event_bus.dispatch(SendKeysEvent(keys="Enter"))
            await enter
            await enter.event_result(raise_if_any=True, raise_if_none=False)
        except Exception as exc:  # noqa: BLE001 - the fill already landed and verified
            logger.debug("Enter after re-found fill failed: %s", exc)
            meta = {"auto_enter": False}
    msg = (f"element {params.index} had re-rendered (stale index) — typed '{params.text}' "
           f"into the live '{label}' field instead; it now reads '{verified}'.")
    logger.info("🩹 %s", msg)
    return ActionResult(extracted_content=msg, long_term_memory=msg,
                        include_in_memory=True, metadata=meta)


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

# Resolve a combobox by the LABEL a human reads beside it, so the agent never has to
# supply an index. This exists because select_dropdown was the ONE tool that always worked
# on the Send Email "From" field (4/4 in run 20260827_012631) and the ONLY tool that
# demanded an index — every other tool takes text. The agent therefore reached for
# text-shaped tools first, burned steps on a panel that was not open yet, and only landed
# on select_dropdown after failing. Two earlier attempts at helping it FIND the box (a
# positional identity in list_actions; a borrowed name in RAW_FIND_JS) were never called
# once in three live runs, because both ended in "now go call select_dropdown with an
# index" — the extra hop the agent skips.
#
# react-select gives its input no aria-label / aria-labelledby / name / placeholder, and
# the visible label is a separate node — often a BARE TEXT NODE sharing a block with the
# widget's current value ("From Me  Include signature" in the live DOM). So the match runs
# over preceding siblings including text nodes, collecting the two nearest: the innermost
# is the VALUE ("Me") and the label ("From") is further out. Placeholder: %s = JSON tokens.
_CB_RESOLVE_BY_TEXT_JS = r"""
(function () {
  var TOKENS = %s;
  var CTRL_SEL = 'input[role=combobox], input[id^="react-select"][id$="-input"], select';
  var t = function (s) { return String(s || '').replace(/\s+/g, ' ').trim(); };
  var labelOf = function (el) {
    var ids = el.getAttribute('aria-labelledby');
    if (ids) {
      var j = t(ids.split(/\s+/).map(function (id) {
        var n = document.getElementById(id); return n ? t(n.textContent) : ''; }).join(' '));
      if (j) return j;
    }
    if (el.labels && el.labels.length) { var lt = t(el.labels[0].textContent); if (lt) return lt; }
    var w = el.closest ? el.closest('label') : null;
    if (w) { var wt = t(w.textContent); if (wt) return wt; }
    for (var a = 0; a < ['aria-label', 'title', 'placeholder'].length; a++) {
      var v = t(el.getAttribute(['aria-label', 'title', 'placeholder'][a]));
      if (v) return v;
    }
    // Nothing wired up: the two nearest preceding texts, innermost first. previousSibling
    // (not previousElementSibling) because the label is frequently a bare text node.
    var found = [], node = el;
    for (var hop = 0; hop < 8 && node && found.length < 2; hop++) {
      for (var s = node.previousSibling; s && found.length < 2; s = s.previousSibling) {
        // A preceding sibling holding its OWN dropdown is the previous FIELD, not this
        // one's label. Without this the walk climbed out of its row and 'Tax year' also
        // matched the Period box sitting under it — two anonymous comboboxes, one label.
        if (s.nodeType === 1 && s.querySelector && s.querySelector(CTRL_SEL)) continue;
        var st = s.nodeType === 3 ? t(s.data)
               : s.nodeType === 1 ? t(s.innerText || s.textContent) : '';
        if (st) found.push(st.slice(0, 80));
      }
      node = node.parentNode;
      if (node && node.nodeType !== 1) break;
    }
    return found.join(' ');
  };
  var boxes = [], seen = {};
  var all = document.querySelectorAll(CTRL_SEL);
  for (var i = 0; i < all.length; i++) {
    var el = all[i], r = el.getBoundingClientRect();
    // A control the user cannot see is not the one they named. react-select's input is
    // narrow but never zero-area; a display:none twin in a collapsed panel is.
    if (!r.width && !r.height) continue;
    var label = labelOf(el);
    if (!el.id) el.id = 'ao-cb-' + (++window.__ao_cb_seq || (window.__ao_cb_seq = 1));
    if (seen[el.id]) continue;
    seen[el.id] = 1;
    boxes.push({ id: el.id, label: label, native: el.tagName === 'SELECT' });
  }
  var hits = boxes.filter(function (b) {
    var l = b.label.toLowerCase();
    return TOKENS.every(function (tk) { return l.indexOf(tk) !== -1; });
  });
  if (hits.length === 1) {
    // Mark the widget root the 'state' read-back reads from — same walk _CB_RESOLVE_JS
    // does on the index path: climb while the subtree still holds exactly OUR control.
    var input = document.getElementById(hits[0].id);
    var root = input && (input.parentElement || input);
    for (var h = 0; root && h < 4; h++) {
      var par = root.parentElement;
      if (!par || par.querySelectorAll(CTRL_SEL).length !== 1) break;
      root = par;
    }
    var prev = document.querySelectorAll('[STAMP]');
    for (var k = 0; k < prev.length; k++) prev[k].removeAttribute('STAMP');
    if (root) root.setAttribute('STAMP', '1');
  }
  return { hits: hits, labels: boxes.map(function (b) { return b.label; }).filter(Boolean) };
})()
""".replace("STAMP", _CB_STAMP)


# Document-level ops on the resolved combobox, addressed by the input's id.
#   open    — focus the input and fire a real pointer+mouse sequence on it (react-select
#             opens on the control's mousedown; a bare .focus() or .click() does nothing).
#   options — the menu's CURRENT options: react-select instance-prefixed ids first, then
#             the input's aria-controls/owns listbox, then any [role=option]. Options with
#             no id get one stamped so a later 'pick' can address them.
#   pick    — dispatch the pointer+mouse sequence on ONE option (by stamped id). react-select
#             selects on the option's mousedown — this is the click el.click() never was.
#   escape  — keydown/keyup Escape on the input: close the menu and clear the typed filter
#             (react-select resets inputValue on Escape) for the zero-options recovery cycle.
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
    try { input.focus({preventScroll: true}); } catch (e) {}
    fire(input, SEQ);
    return { ok: true };
  }
  if (OP === 'escape') {
    // react-select's own Escape handling: closes the menu and clears the typed filter
    // (a stuck filter is why a dead combobox keeps showing 'No options' for any text).
    try { input.focus({preventScroll: true}); } catch (e) {}
    ['keydown', 'keyup'].forEach(function (t) {
      try {
        input.dispatchEvent(new KeyboardEvent(t, { key: 'Escape', code: 'Escape',
                                                   bubbles: true, cancelable: true }));
      } catch (e) {}
    });
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
    if len(partial) == 1:
        return partial[0]
    if partial:
        return None
    # Separator-squashed tier: the task's fragment and the option spell the same thing
    # with different separators ('no-reply' vs 'noreply@actingoffice.com' — run
    # 20260817_124339 seg 7 missed the From option both ways). Unique containment only,
    # and never for tiny squashes (a 2-3 char fragment would match half the menu).
    squash = re.sub(r"[^a-z0-9]+", "", (target or "").lower())
    if len(squash) >= 4:
        hits = [o for o in options
                if squash in re.sub(r"[^a-z0-9]+", "",
                                    str(o.get("text") or "").lower())]
        if len(hits) == 1:
            return hits[0]
    return None


def _cb_option_lines(options: list[dict[str, Any]], limit: int = 10) -> str:
    shown = ", ".join(f"'{str(o.get('text') or '')[:60]}'" for o in options[:limit])
    more = f" (+{len(options) - limit} more)" if len(options) > limit else ""
    return shown + more


# The combobox CONTROL's identity, for the recording. select_dropdown opens the widget
# INSIDE the tool, so a run that used it leaves no click on the box anywhere in its trace —
# and the compiled script then has nothing that opens the menu. script_compile synthesizes
# an opener from this stamp (see the select_dropdown branch); without it, replay's only
# recourse is api.select_option's "re-click the previous click" guess, which on the Send
# Email panel re-clicked the envelope icon that OPENS the panel and could never work
# (entry aa3a76b7c82dcf8b, run 20260827_112618). Shaped like DOMInteractedElement.to_dict()
# so compile anchors it through the same _selectors/_fingerprint path as any recorded
# click. The synthetic 'ao-cb-N' id the resolver stamps is stripped: it does not exist on
# the next run and must never reach a selector. Placeholder: %(id)s = JSON input id.
_CB_IDENTITY_JS = r"""
(function () {
  var el = document.getElementById(%(id)s);
  if (!el) return null;
  var t = function (s) { return String(s || '').replace(/\s+/g, ' ').trim(); };
  // Same positional-path shape script_compile records for every other element.
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
  var attrs = {};
  ['id', 'role', 'name', 'class', 'aria-label', 'title', 'placeholder',
   'data-testid', 'data-automationid'].forEach(function (a) {
    var v = el.getAttribute(a);
    if (v && !(a === 'id' && v.indexOf('ao-cb-') === 0)) attrs[a] = v;
  });
  var r = el.getBoundingClientRect();
  var out = { node_name: el.tagName, attributes: attrs,
              x_path: (el.getRootNode && el.getRootNode() !== document)
                        ? '' : xpathOf(el).replace(/^\//, '') };
  if (r && (r.width || r.height)) {
    out.bounds = { x: r.left, y: r.top, width: r.width, height: r.height };
  }
  return out;
})()
"""


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
    return await _combobox_pick(browser_session, str(info["input_id"]),
                                target, f"at index {params.index}")


_NATIVE_SELECT_BY_ID_JS = r"""
(function () {
  var el = document.getElementById(%(id)s), want = %(want)s;
  if (!el) return { error: 'gone' };
  var n = function (s) { return String(s || '').toLowerCase().replace(/\s+/g, ' ').trim(); };
  var w = n(want), opts = [], i;
  for (i = 0; i < el.options.length; i++) opts.push(el.options[i]);
  var hit = null;
  for (i = 0; i < opts.length && !hit; i++) {
    if (n(opts[i].text) === w || n(opts[i].value) === w) hit = opts[i];
  }
  for (i = 0; i < opts.length && !hit; i++) {
    if (n(opts[i].text).indexOf(w) !== -1) hit = opts[i];
  }
  if (!hit) return { error: 'no-option',
                     options: opts.slice(0, 30).map(function (o) { return o.text; }) };
  el.value = hit.value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  var sel = el.selectedOptions && el.selectedOptions[0];
  // Identity for the recording. compile keys the native-select step on
  // metadata.interacted_element (node_name SELECT + attributes) — without it the pick is
  // dropped and replay submits the form with its DEFAULTS. The resolver stamps a synthetic
  // 'ao-cb-N' id on any control that had none; that id does not exist on the next run, so
  // it must never reach a selector.
  var attrs = {};
  ['id', 'name', 'class', 'aria-label', 'title', 'data-testid'].forEach(function (a) {
    var v = el.getAttribute(a);
    if (v && !(a === 'id' && v.indexOf('ao-cb-') === 0)) attrs[a] = v;
  });
  return { ok: true, shows: (sel && sel.text) || el.value, attrs: attrs };
})()
"""


async def _native_select_by_id(browser_session, el_id: str, target: str,
                               where: str) -> ActionResult:
    """Set a native <select> located by label. Reports what the select SHOWS afterwards —
    never the request — so a refused pick cannot read as success."""
    try:
        got = await _eval_js(browser_session, _NATIVE_SELECT_BY_ID_JS % {
            "id": json.dumps(el_id), "want": json.dumps(target)})
    except Exception as exc:  # noqa: BLE001
        return ActionResult(error=f"select_dropdown '{target}' {where}: {exc}")
    if not isinstance(got, dict) or got.get("error"):
        if isinstance(got, dict) and got.get("error") == "no-option":
            listed = ", ".join(f"'{o}'" for o in (got.get("options") or []))
            return ActionResult(error=(
                f"select_dropdown '{target}' {where}: no such option. The dropdown "
                f"ACTUALLY lists: {listed}. Pick one of these exact texts."))
        return ActionResult(error=(
            f"select_dropdown '{target}' {where}: the dropdown went away before it "
            "could be set. Re-read the page and try again."))
    shows = str(got.get("shows") or "")
    meta = {"interacted_element": {"node_name": "SELECT",
                                   "attributes": dict(got.get("attrs") or {}),
                                   "ax_name": target}}
    msg = (f"Selected '{target}' in the dropdown {where} — it now shows '{shows}'. "
           "Do NOT set it again.")
    logger.info("🔽 %s", msg)
    return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg,
                        metadata=meta)


async def _select_dropdown_by_label(browser_session, near_text: str,
                                    target: str) -> ActionResult:
    """select_dropdown addressed by the LABEL beside the control instead of by index.

    Refuses honestly rather than guessing: nothing matched lists the labels that DO exist
    on the page (the same redirect the option-miss receipt gives), and two matches name
    both so the agent can say which — a silent .first here would set the wrong field, and
    this page has three comboboxes that carry no distinguishing attributes at all."""
    # Same token grammar find_by_text uses, so a label matches here exactly as it would
    # there — one grammar across the toolbox, no per-tool surprises.
    tokens = [t for t in re.split(r"[^a-z0-9]+", near_text.lower()) if t]
    if not tokens:
        return ActionResult(error="select_dropdown: near_text is empty.")
    try:
        got = await _eval_js(browser_session, _CB_RESOLVE_BY_TEXT_JS % json.dumps(tokens))
    except Exception as exc:  # noqa: BLE001 - a resolver that cannot run must say so
        return ActionResult(error=f"select_dropdown near '{near_text}': could not read the "
                                  f"page to locate the dropdown ({exc}).")
    hits = (got or {}).get("hits") or []
    if not hits:
        listed = ", ".join(f"'{lbl}'" for lbl in (got or {}).get("labels") or []) or "none"
        return ActionResult(error=(
            f"select_dropdown near '{near_text}': no dropdown on this page sits beside "
            f"that text. The dropdowns actually on the page are labelled: {listed}. Pick "
            "one of those labels, or check you are on the right page/panel."))
    if len(hits) > 1:
        listed = ", ".join(f"'{h.get('label')}'" for h in hits)
        return ActionResult(error=(
            f"select_dropdown near '{near_text}': {len(hits)} dropdowns match that text "
            f"({listed}) — cannot know which you mean. Use the fuller label that tells "
            "them apart, or pass index=<the combobox input's index>."))
    hit = hits[0]
    where = f"near '{near_text}'"
    if hit.get("native"):
        # A native <select> beside that label: drive it through the same read-back path the
        # index form uses, by handing the id straight to the DOM.
        return await _native_select_by_id(browser_session, str(hit["id"]), target, where)
    # The label the agent addressed it by rides along: it is the only name this control
    # has, and the compiled opener needs it for the label-scoped selector rung.
    return await _combobox_pick(browser_session, str(hit["id"]), target, where,
                                label=str(hit.get("label") or near_text))


async def _cb_identity(browser_session, input_id: str,
                       label: str = "") -> dict[str, Any] | None:
    """The combobox control as a recordable element, or None when it cannot be read.
    Best-effort: a pick that worked must never fail over its own bookkeeping."""
    try:
        got = await _eval_js(browser_session,
                             _CB_IDENTITY_JS % {"id": json.dumps(input_id)})
    except Exception as exc:  # noqa: BLE001 - identity is bookkeeping; the pick stands
        logger.debug("combobox identity probe failed: %s", exc)
        return None
    if not isinstance(got, dict) or not got.get("node_name"):
        return None
    name = " ".join(str(label or "").split())
    if name:
        # The label the agent addressed the box BY ("From :"). _selectors offers the
        # label-scoped rung to a control no attribute can name, which is exactly what a
        # react-select input is (volatile id, no aria-label, no placeholder).
        got["ax_name"] = name
    return got


async def _combobox_pick(browser_session, input_id: str, target: str,
                         where: str, label: str = "") -> ActionResult:
    """Open the resolved combobox, pick `target`, verify it took. Everything here is keyed
    on the input's DOM id, so it serves BOTH addressing modes — by index (the element the
    agent already has) and by label (`near_text`). `where` is how the receipts name the
    target back to the agent ("at index 7093" / "near 'From'") and is the ONLY difference
    between the two paths: the pick machinery itself is measured-good and stays untouched."""
    try:
        await _cb_op(browser_session, "open", input_id)
    except Exception as exc:  # noqa: BLE001 - report; the agent recovers via its receipt
        return ActionResult(error=(
            f"select_dropdown '{target}' {where}: could not open the "
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
                           f"document.getElementById({json.dumps(input_id)}).focus({{preventScroll: true}})")
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
    if chosen is None and not options:
        # ZERO options even after typing: one recovery cycle before concluding anything.
        # A reloaded SPA page can mount the combobox before its option source binds — the
        # filter then shows 'No options' for ANY text and retyping never recovers it
        # (run 20260807_110530: 'Daniel Bruce' → 'No options' on every retry while the
        # employee-list GET kept returning 200). Escape clears the stuck filter, a fresh
        # open shows the UNFILTERED truth about the source.
        try:
            await _cb_op(browser_session, "escape", input_id)
            await _cb_op(browser_session, "open", input_id)
        except Exception as exc:  # noqa: BLE001 - the receipt below still tells the truth
            logger.debug("combobox zero-options recovery reopen failed: %s", exc)
        options = await _cb_poll_options(browser_session, input_id, timeout=4.0)
        if not options:
            # The source binds LATE after a page load (run 20260807_120553: identical
            # post-reload attempts found options at idx 183/348 and nothing at 264/429 —
            # a race, not a dead endpoint). One paused second attempt catches the late
            # bind without handing the agent a retry loop.
            await asyncio.sleep(2.0)
            try:
                await _cb_op(browser_session, "escape", input_id)
                await _cb_op(browser_session, "open", input_id)
            except Exception as exc:  # noqa: BLE001 - same degradation as above
                logger.debug("combobox second recovery reopen failed: %s", exc)
            options = await _cb_poll_options(browser_session, input_id, timeout=3.0)
        chosen = _choose_option(options, target)
        if options and chosen is None:
            try:
                await _eval_js(browser_session,
                               f"document.getElementById({json.dumps(input_id)}).focus({{preventScroll: true}})")
                cdp_session = await browser_session.get_or_create_cdp_session()
                await cdp_session.cdp_client.send.Input.insertText(
                    params={"text": target}, session_id=cdp_session.session_id)
                filtered = True
            except Exception as exc:  # noqa: BLE001 - fall through to the miss listing
                logger.debug("combobox recovery filter typing failed: %s", exc)
            refreshed = await _cb_poll_options(browser_session, input_id, timeout=2.0)
            if refreshed:
                options = refreshed
            chosen = _choose_option(options, target)
    if chosen is None:
        listed = _cb_option_lines(options)
        if options:
            return ActionResult(error=(
                f"select_dropdown '{target}' {where}: no such option. "
                f"The dropdown ACTUALLY lists: {listed}. These are all that exist — "
                f"re-read the task and pick one of these exact texts with "
                f"select_dropdown(index=<the combobox index>, text='<option>'). Do NOT hunt "
                f"the page for '{target}'."))
        return ActionResult(error=(
            f"select_dropdown '{target}' {where}: the combobox opened but "
            "its option list NEVER rendered — even after clearing the filter, reopening "
            "the menu, and waiting. The dropdown's data source did not load on this page "
            "view; retyping into it as-is will keep showing no options. Recover the way "
            "the task prescribes if it names a recovery (e.g. refresh the page and start "
            "over); otherwise reload the page yourself. After the reload, WAIT for the "
            "page to finish loading before touching the combobox, then re-run "
            "select_dropdown — do NOT type into any other field until this selection has "
            "succeeded and the record's data is on screen."))

    picked = None
    try:
        picked = await _cb_op(browser_session, "pick", input_id, wanted=str(chosen["id"]))
    except Exception as exc:  # noqa: BLE001 - the read-back below is the real verdict
        logger.debug("combobox pick dispatch failed: %s", exc)
    if not (isinstance(picked, dict) and picked.get("clicked")):
        last_seen = picked.get("options") if isinstance(picked, dict) else None
        listed = _cb_option_lines(last_seen or options)
        return ActionResult(error=(
            f"select_dropdown '{target}' {where}: the option "
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
    # ...and the OPENER's, because THIS tool opened the menu: nothing else in the trace
    # does, so compile has to synthesize that click (see _CB_IDENTITY_JS).
    opener = await _cb_identity(browser_session, input_id, label)
    if opener:
        meta["opener_element"] = opener
    if took:
        msg = (f"Selected '{chosen.get('text')}' in the combobox {where} — "
               f"it now shows '{display}'. Do NOT set it again.")
        logger.info("🔽 %s", msg)
        return ActionResult(extracted_content=msg, include_in_memory=True,
                            long_term_memory=msg, metadata=meta)
    return ActionResult(error=(
        f"select_dropdown '{target}' {where}: clicked the option "
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
        await _call_on_field(handle,
            # preventScroll: focusing a field the browser thinks is off-screen scrolls
            # the page to the caret, and that movement dismisses an open Callout. The
            # scroll pin would revert it, but not moving at all is cheaper and also
            # covers a document the pin failed to install into.
            "function(){ this.focus({preventScroll: true}); return true; }")
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


# ------------------------------- dialog-outcome clicks -------------------------------
# The duplicate-add loop (run 20260805_131827_339055): the modal's Save closes the dialog
# silently; with no receipt saying so, the agent judged every (successful) save a failure
# and re-added the same benefit ten times. Clicks on elements INSIDE a dialog now report
# what happened to the dialog — the closed/still-open distinction is a did-it-actually-
# save signal, and "STILL OPEN" also exposes silent client-side validation rejection.

_DIALOG_SETTLE_S = 0.8

# ------------------------------- network-outcome clicks -------------------------------
# The FPS redo (run 20260805_142523_388784): the May submission POSTed and the server
# answered `"isSubmitted": true` — while the agent, staring at a stale "No employees FPS
# submitted so far" list, decided it had failed and re-submitted the whole company
# against April (bounced: "already submitted"). Both verdicts sat in the live
# NetworkCollector unshown. The runner registers that collector here per segment; every
# click receipt then reports the WRITE requests the click fired — method, status, and a
# short server-verdict extract from the captured response body — after waiting (bounded)
# for them to settle. Condition-based by construction: a Save click's receipt arrives
# when the request finishes, not after a guessed sleep, and an in-dialog click that fired
# NO write says exactly that (the swallowed-save flag).

_LIVE_NETWORK: Any = None
_SEGMENT_T0: float = 0.0  # monotonic segment start (stamped with the collector)
_FAIL_BOUNCED = False     # fail_and_stop's contradiction bounce fired this segment
# Monotonic start of the most recent ACTING verb, so verify_save_registered can answer
# "did MY last action write?" instead of "did anything write this segment?". The old
# probe returned the segment's FIRST create-write, which saturates: across a 12-employee
# list, employee 1's save vouched for save #3 forever (run 20260824_165824 seg 4 —
# Aayan Dickson ended with two or three payments).
_LAST_ACTION_T0: float = 0.0
_WRITE_SNIFF_S = 1.0     # window (from click dispatch) for a triggered write to START
# The verifier can wait longer than a click receipt: it is called deliberately, once, by an
# agent that is ALREADY unsure, so a slow POST is worth waiting for — whereas widening
# _WRITE_SNIFF_S would tax every click in every run. This is what keeps a real-but-late
# write from reading as "no write" and inviting a duplicate re-entry.
_VERIFY_SNIFF_S = 6.0
_WRITE_SETTLE_S = 8.0    # cap on waiting for started writes to finish (matches the
                         # end-of-run in-flight-save poll)
_BODY_POLL_S = 1.0       # extra grace for the async body capture after settle


def set_live_network(collector: Any) -> None:
    """Register the run's live NetworkCollector for click receipts (runner-owned).
    Marks the segment start, which windows the fail_and_stop contradiction bounce and
    re-arms it (once per segment), and clears the copy buffer.

    The clipboard is segment-scoped for the same reason the rest of this state is: a
    `paste_text(index)` with no `text` falls back to whatever `copy_text` last captured,
    so a value left over from an EARLIER subtask would be pasted silently instead of the
    step failing honestly. An OTP is the case that matters — it is captured in one slice,
    consumed in the next, and is stale by the one after."""
    global _LIVE_NETWORK, _SEGMENT_T0, _FAIL_BOUNCED, _LAST_ACTION_T0
    _LIVE_NETWORK = collector
    _CLICK_LEDGER.clear()   # per-segment, like every other name reset here
    _REPEAT_TARGETS.clear()
    _SEGMENT_T0 = time.monotonic()
    _LAST_ACTION_T0 = _SEGMENT_T0    # no action yet: the whole segment is the window
    _FAIL_BOUNCED = False
    _CLIPBOARD.update(value="", label="")


def _stamp_action() -> float:
    """Mark now as the start of the current acting verb and return it.

    Every acting verb already needed this timestamp for its own network receipt; recording
    it module-side too is what lets verify_save_registered scope its answer to THIS action
    rather than to the whole segment."""
    global _LAST_ACTION_T0
    _LAST_ACTION_T0 = time.monotonic()
    note_interaction()
    return _LAST_ACTION_T0


def note_interaction() -> None:
    """Tell the live collector the segment is ACTING on the page, which closes the
    page-load window it opened at the last document load (NetworkCollector.note_interaction).

    Called from all three acting paths — this module's verbs via _stamp_action, tier-0
    replay via script_compile.run_steps, tier-1 replay via skills.api.SkillApi — because a
    path that never closed the window would leave every write after its first navigation
    marked as the app's boot traffic, and the write gate would stop judging that segment
    at all. Best-effort: no collector simply means there is nothing to attribute."""
    if _LIVE_NETWORK is None:
        return
    try:
        _LIVE_NETWORK.note_interaction()
    except Exception as exc:  # noqa: BLE001 - attribution must never break an action
        logger.debug("note_interaction failed: %s", exc)


def clear_live_network() -> None:
    global _LIVE_NETWORK, _FAIL_BOUNCED
    _LIVE_NETWORK = None
    _FAIL_BOUNCED = False


# Writes the app's infrastructure fires constantly (push auth, SignalR negotiate, token
# refresh, keep-alives) — never evidence that USER work landed, so the fail_and_stop
# bounce must ignore them (observed live: /auth/webpush POSTs in every segment).
_CLOSE_DONE_HINT = (
    "If closing that tab was the LAST action of your step, your step is COMPLETE — call "
    "done NOW, reporting what you accomplished before the close. A done batched behind a "
    "close is DROPPED: closing the focused tab detaches it, and the rest of that action "
    "batch is skipped. Do NOT re-derive your job from the page you landed on — it is a "
    "different page from the one you did the work in, and it may well show that work as "
    "not started."
)


def _close_receipt(inner: Any, tab_id: str) -> Any:
    """A successful close, carrying the completion cue into the NEXT step's memory.

    `close` is the one action that can destroy the page the agent is standing on, and
    browser-use skips whatever was queued behind it. A slice whose wording ends with
    "close this tab" therefore ALWAYS emits close+done and ALWAYS loses the done — the
    agent then wakes on a surviving tab with no record of finishing and starts over.
    Run 20260901_165748 subtask 8 is the measured case: 12 employees advanced and
    submitted in the portal, the done dropped, and the whole slice re-run through the
    app's own review panel (which honestly opens at employee 1 of 12).

    A close that ERRORED is returned untouched: it closed nothing, and telling the agent
    its step is complete there would be the same false completion with the sign flipped.
    """
    if getattr(inner, "error", None):
        return inner
    base = (getattr(inner, "extracted_content", None) or f"Closed tab #{tab_id}").strip()
    msg = f"{base} — {_CLOSE_DONE_HINT}"
    return ActionResult(
        extracted_content=msg,
        long_term_memory=msg,
        include_in_memory=True,
        metadata=getattr(inner, "metadata", None),
    )


def _state_text_witness(state: Any, tokens: list[str], query: str) -> str:
    """The query as a line of the page state the agent was SHOWN, or "".

    A second, independent witness for find_by_text's static-text branch. The JS probe
    runs in the main frame AFTER the scroll sweep, against a DOM that may have
    re-rendered underneath it; `state` was serialized one frame earlier and is the page
    the agent actually read. selector_map holds interactive nodes only — which is why
    the name lookup missed — but llm_representation() re-serializes the same tree
    INCLUDING its static text.

    Run 20260901_174417 subtask 2 is what this is for: 'Select sender' was the open
    panel's combobox placeholder, right there in the state message under 'From' (the
    same snapshot still said "Loading ..."), the JS probe returned 0 anyway, and the
    receipt then told the agent its open panel "is not in this page's DOM". It
    blind-clicked a nameless button and re-ran the same string through search_page.

    Best-effort by construction: llm_representation is semi-private in browser-use, so
    an absent or raising one degrades to exactly the previous behaviour.
    """
    try:
        rep = state.dom_state.llm_representation() or ""
    except Exception as exc:  # noqa: BLE001 - a witness that cannot testify says nothing
        logger.debug("find_by_text state-text witness unavailable: %s", exc)
        return ""
    low = rep.lower()
    if not rep or not all(t in low for t in tokens):
        return ""
    line = next((ln for ln in rep.splitlines()
                 if all(t in ln.lower() for t in tokens)), query)
    return " ".join(line.split())[:200]


_INFRA_WRITE_RE = re.compile(r"/auth/|negotiate|token|keepalive|telemetry", re.I)


def _segment_accepted_write() -> dict[str, Any] | None:
    """The LATEST accepted BUSINESS write of the current segment, or None: settled 2xx,
    non-negative body verdict, not infra traffic. Read from the live collector at call
    time so a write that settled after its click's receipt still counts.

    None when the segment's most recent business write was REFUSED. The bounce this
    feeds argues "your failure claim contradicts this segment's own receipt", and that
    only holds when the receipt is about the write the agent is actually reacting to.
    Run 20260901_163709 subtask 8 is what the latest-not-first rule is for: the agent
    advanced 11 employees (11 accepted writes), clicked Submit, and the server refused
    it 200-with-"Unable to sent email to client." Scanning FORWARD stepped over that
    refusal, produced one of the 11 routine advances, and told an agent that had
    reported the truth to carry on — it reopened the review panel in the app and
    re-entered all 11 employees through the agent-side screen. A refusal landing after
    the last acceptance is evidence the claim is TRUE; it must never be what refutes it.
    """
    if _LIVE_NETWORK is None:
        return None
    try:
        writes = _LIVE_NETWORK.writes_since(_SEGMENT_T0)
    except Exception:  # noqa: BLE001 - the bounce is best-effort, never a crash source
        return None
    for w in reversed(list(writes)):
        record = w.get("record") or {}
        if _INFRA_WRITE_RE.search(str(record.get("url") or "")):
            continue
        if record.get("failed") or not w.get("settled"):
            continue
        status = record.get("status")
        if not (isinstance(status, int) and 200 <= status < 300):
            continue
        verdict = _write_verdict(record)
        if verdict is not None and verdict[0]:
            # The segment's last word is a refusal. Nothing earlier can contradict a
            # failure claim made after it.
            return None
        return record
    return None


async def _fail_and_stop_result(reason: str) -> ActionResult:
    """fail_and_stop's body, with the contradiction bounce: a failure claim made in a
    segment whose OWN traffic carries an accepted business write is refused ONCE, citing
    the receipt (run 20260807_164137: the Save receipt said "the write above SUCCEEDED —
    do NOT redo" and the agent declared "new employee not created" one step later). The
    second call always goes through — honest failures (saved-but-wrong, later objectives
    unreachable) stay possible."""
    global _FAIL_BOUNCED
    if not _FAIL_BOUNCED:
        accepted = _segment_accepted_write()
        if accepted is not None:
            _FAIL_BOUNCED = True
            msg = (
                "fail_and_stop REFUSED (once): your failure claim contradicts this "
                f"segment's own receipt — {_format_write({'record': accepted})}. The "
                "record/change likely EXISTS and the app may simply have navigated "
                "after the save. Re-read the page and continue from what it actually "
                "shows; call fail_and_stop again ONLY if you have evidence that "
                "contradicts the receipt (the record is genuinely absent or wrong "
                "on re-check)."
            )
            logger.info("■ fail_and_stop bounced (claim was: %s)", reason[:120])
            return ActionResult(error=msg, metadata={"refused_stop": True})
    logger.info("■ fail_and_stop: %s", reason)
    return ActionResult(
        is_done=True,
        success=False,
        error=reason,
        extracted_content=f"Run failed and stopped: {reason}",
        long_term_memory=f"Run failed and stopped: {reason}",
        include_in_memory=True,
    )


def _write_verdict(record: dict[str, Any]) -> tuple[bool, str] | None:
    """(negative, text) distilled from a captured JSON write body, or None.

    Walks the body (bounded depth — the FPS success nests `isSubmitted` two levels down
    in result.submitDetail) for the first failure signal (`errors`, `message` beside a
    false `status`, false `success`) and, failing that, an affirmation. A 200 whose body
    says "already submitted" is a REFUSAL, and the receipt must say so."""
    body = record.get("body")
    if not body:
        return None
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None
    found: dict[str, Any] = {}

    def _scan(obj: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                lk = str(key).lower()
                if lk in ("errors", "message", "status", "success", "issubmitted") \
                        and lk not in found:
                    found[lk] = value
                _scan(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:10]:
                _scan(item, depth + 1)

    _scan(data, 0)

    def _first_error_message() -> str | None:
        errors = found.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                return str(first.get("message") or first)
            return str(first)
        return None

    message = found.get("message")
    err = _first_error_message()
    if err or (message and found.get("status") is False):
        return True, str(err or message)[:200]
    if found.get("success") is False or found.get("status") is False:
        return True, "the server returned a false status with no message"
    if found.get("issubmitted") is True:
        return False, "isSubmitted: true"
    if found.get("success") is True:
        return False, "success: true"
    if message:
        return False, f'server says: "{str(message)[:160]}"'
    return None


def _format_write(snapshot: dict[str, Any]) -> str:
    record = snapshot["record"]
    url = str(record.get("url") or "")
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    path = "/" + url.split("/", 1)[1] if "/" in url else url
    if len(path) > 70:
        path = "…" + path[-69:]
    method = record.get("method") or "POST"
    if record.get("failed"):
        return f"{method} {path} FAILED ({record.get('errorText') or 'network error'})"
    status = record.get("status")
    if status is None:
        return (f"{method} {path} is STILL IN FLIGHT (no response yet) — wait and "
                "re-check before repeating anything")
    verdict = _write_verdict(record)
    if verdict is None:
        return f"{method} {path} → {status}"
    negative, text = verdict
    if negative:
        return f'{method} {path} → {status} but the server REFUSED it: "{text}"'
    return f"{method} {path} → {status}; {text}"


def _writes_accepted(writes: list[dict[str, Any]]) -> bool:
    """True when the fired writes prove the action landed: at least one settled 2xx with
    a non-negative server verdict, and NONE failed, refused, or still in flight. This is
    the signal that must override the dialog pessimism — a receipt may not print
    'POST → 200' and 'likely did NOT go through' about the same click."""
    good = False
    for w in writes:
        record = w["record"]
        if record.get("failed"):
            return False
        if not w["settled"] or record.get("status") is None:
            return False  # still in flight — the receipt already says to wait
        status = record.get("status")
        if not (isinstance(status, int) and 200 <= status < 300):
            return False
        verdict = _write_verdict(record)
        if verdict is not None and verdict[0]:
            return False  # 2xx whose body is a refusal
        good = True
    return good


async def _verify_last_action_write() -> str:
    """CONFIRMED / REFUSED / UNCONFIRMED for the writes fired since the last acting verb.

    Three verdicts, not two, because the two-verdict vocabulary is what made a
    client-staged save indistinguishable from a failed one: "NOT REGISTERED — save again"
    on a save that had actually committed is a duplicate-add instruction. Run
    20260824_165824 seg 4 spent three Saves on Aayan Dickson that way.

    Shares `_await_writes` with the click receipt, differing only in the START window
    (_VERIFY_SNIFF_S, far longer than a receipt can afford). The shared helper is what
    guarantees the response BODY has landed before `_writes_accepted` rules: an earlier
    hand-rolled copy of the polling here skipped that phase, and a 2xx whose body says
    "already submitted" then read as CONFIRMED — the one verdict this tool must never get
    wrong, since the agent calls it precisely when it is unsure."""
    t0 = _LAST_ACTION_T0
    try:
        writes = await _await_writes(_LIVE_NETWORK, t0, _VERIFY_SNIFF_S)
        if not writes:
            return ("UNCONFIRMED: no write request fired since your last action. This is NOT "
                    "proof of failure — some saves in this app commit without any network "
                    "traffic. Look for the record on the page (the new row, the updated "
                    "total) and treat THAT as the answer. Do NOT re-enter the data just "
                    "because this tool did not say CONFIRMED: if the save did land, a second "
                    "entry creates a DUPLICATE record.")
        if _writes_accepted(writes):
            return (f"CONFIRMED: the server accepted your last action's write "
                    f"({'; '.join(_format_write(w) for w in writes[:2])}). It is saved — do "
                    f"NOT repeat it.")
        # Fired but not accepted: name what the server actually said, so a retry fixes the
        # cause instead of resubmitting the same body.
        detail = "; ".join(_format_write(w) for w in writes[:2])
        return (f"REFUSED: your last action DID reach the server and it did not accept the "
                f"write ({detail}). Read that verdict, fix what it names (validation errors "
                f"on the form, a duplicate the server already holds), and only then save "
                f"again. If it says the record already exists, it is SAVED — do not retry.")
    except Exception as exc:  # noqa: BLE001 - a verifier must never crash the run
        logger.debug("last-action write verification failed: %s", exc)
        return ("UNCONFIRMED: the write probe failed, so this tool cannot tell you anything "
                "either way. Verify the record on the page; do not re-enter data blindly.")


async def _await_writes(collector: Any, t0: float, sniff_s: float) -> list[dict[str, Any]]:
    """The write records fired since `t0`, waited out in three phases: for one to START
    (`sniff_s`), for the started ones to SETTLE (_WRITE_SETTLE_S), and for their JSON
    BODIES to land (_BODY_POLL_S).

    All three phases matter to the verdict, and the third is the easy one to forget:
    `_write_verdict` reads `record["body"]`, so without the body wait `_writes_accepted`
    calls a settled 2xx whose body is a REFUSAL accepted — the "FPS already submitted"
    shape. Every deadline is measured from `t0`, not from entry: a caller reached after an
    agent-step boundary has already spent that budget in real time, and re-spending it is
    pure dead wall clock.
    """
    writes = collector.writes_since(t0)
    while not writes and time.monotonic() - t0 < sniff_s:
        await asyncio.sleep(0.15)
        writes = collector.writes_since(t0)
    if not writes:
        return []
    while any(not w["settled"] for w in writes) \
            and time.monotonic() - t0 < _WRITE_SETTLE_S:
        await asyncio.sleep(0.2)
        writes = collector.writes_since(t0)
    body_deadline = t0 + _WRITE_SETTLE_S + _BODY_POLL_S
    while time.monotonic() < body_deadline and any(
            w["settled"] and "body" not in w["record"]
            and "json" in str((w["record"].get("response_headers") or {})
                              .get("content-type", "")).lower()
            for w in writes):
        await asyncio.sleep(0.1)
    return writes


async def _network_outcome(collector: Any, t0: float,
                           expect_write: bool) -> tuple[str, bool, bool]:
    """(receipt suffix, fired, accepted) for the write requests a click fired — suffix
    empty when none fired and none were expected. Waits, bounded, for started writes to
    settle and their bodies to land — the condition the agent used to approximate with
    blind sleeps. `accepted` is _writes_accepted over the settled records."""
    try:
        writes = await _await_writes(collector, t0, _WRITE_SNIFF_S)
        if not writes:
            # Observation, not verdict: dialog saves can legitimately fire no request
            # (client-staged, websocket, beacon — run 20260810_092500's expense dialog),
            # so asserting "nothing reached the server" here primed a duplicate-add loop.
            return ((" — no write request was observed after this click. Some apps "
                     "save without network traffic, so this alone does not prove "
                     "failure: if this was a save/submit, check the page for the "
                     "change and only redo it if it is genuinely absent."
                     if expect_write else ""), False, False)
        parts = [_format_write(w) for w in writes[:3]]
        more = f" (+{len(writes) - 3} more write requests)" if len(writes) > 3 else ""
        text = " — this click fired " + "; ".join(parts) + more + "."
        return text, True, _writes_accepted(writes)
    except Exception as exc:  # noqa: BLE001 - the receipt degrades, the click stands
        logger.debug("network outcome probe failed: %s", exc)
        return "", False, False


async def _dialog_state(browser_session, node=None) -> dict[str, Any] | None:
    """{'in_dialog': node sits inside an open dialog, 'open': visible dialog count,
    'stamped': that dialog carries the watch stamp}, or None when the page can't be
    probed — callers then skip the dialog receipt. Stamping the SPECIFIC dialog is what
    lets the post-click check survive chained panels (save closes its dialog, the next
    panel opens, the global count never drops)."""
    try:
        got = await _eval_js(browser_session, _DIALOG_COUNT_JS)
    except Exception as exc:  # noqa: BLE001 - probe is best-effort, never fails the click
        logger.debug("dialog count probe failed: %s", exc)
        return None
    if not isinstance(got, dict) or got.get("error") or "open" not in got:
        return None
    in_dialog, stamped = False, False
    if node is not None:
        handle = await _field_handle(browser_session, node)
        if handle:
            try:
                mark = await _call_on_field(handle, _DIALOG_STAMP_JS)
                stamped = bool(isinstance(mark, dict) and mark.get("stamped"))
                in_dialog = stamped
            except Exception as exc:  # noqa: BLE001 - unprobeable node reads as outside
                logger.debug("dialog stamp probe failed: %s", exc)
    return {"in_dialog": in_dialog, "open": int(got["open"]), "stamped": stamped}


# Post-click close poll: cadence, and the longer cap used when the click fired a write
# (the app closes the panel only after it has processed the response).
_DIALOG_POLL_S = 0.25
_DIALOG_WRITE_SETTLE_S = 3.0


async def _stamped_dialog_open(browser_session) -> dict[str, Any] | None:
    """{'present': stamped dialog mounted with layout, 'open': visible dialog count},
    or None = unprobeable."""
    try:
        got = await _eval_js(browser_session, _DIALOG_STAMPED_OPEN_JS)
    except Exception as exc:  # noqa: BLE001 - probe failure must not fail the click
        logger.debug("stamped dialog probe failed: %s", exc)
        return None
    if not isinstance(got, dict) or got.get("error") or "present" not in got:
        return None
    return {"present": bool(got["present"]), "open": int(got.get("open") or 0)}


async def _dialog_closed(browser_session, pre: dict[str, Any],
                         fired_write: bool) -> bool | None:
    """Did the dialog the clicked element lived in close? Stamped path polls THAT
    dialog (early exit on close; longer cap after a write, whose processing is what
    closes the panel). A MISSING stamp alone is not a close: React re-renders REPLACE
    the stamped node, so the stamp vanishes while the dialog stays up (run
    20260807_123259: two loan-toggle clicks and the final Save all read false-CLOSED
    that way, the receipts said ACCEPTED, and an unsaved employee sailed through) —
    closure needs the visible-dialog count to have dropped too. Unstamped falls back
    to the global count delta. None = could not tell — the receipt then stays silent
    rather than guessing."""
    if pre.get("stamped"):
        deadline = time.monotonic() + \
            (_DIALOG_WRITE_SETTLE_S if fired_write else _DIALOG_SETTLE_S)
        verdict: bool | None = None
        while True:
            got = await _stamped_dialog_open(browser_session)
            if got is not None:
                if not got["present"] and got["open"] < pre["open"]:
                    return True        # THE dialog left AND the page has one fewer
                verdict = False        # still present, or re-rendered stamp loss
            if time.monotonic() >= deadline:
                return verdict
            await asyncio.sleep(_DIALOG_POLL_S)
    await asyncio.sleep(_DIALOG_SETTLE_S)
    post = await _dialog_state(browser_session)
    if not isinstance(post, dict):
        return None
    return post["open"] < pre["open"]


_TOGGLE_ROLES = ("switch", "checkbox", "radio", "menuitemcheckbox", "menuitemradio")


def _is_toggle_control(node: Any) -> bool:
    """Toggle-family controls never fire the write their click is 'for', so the in-dialog
    save/submit doubt advisory on their receipts only primes a false 'form is broken'
    narrative (run 20260807_164137 step 13: a loan switch got 'likely did NOT go
    through'). Their receipts stay plain; a toggle that DOES fire a write still gets the
    normal network receipt."""
    if node is None:
        return False
    attrs = getattr(node, "attributes", None) or {}
    if str(attrs.get("role") or "").lower() in _TOGGLE_ROLES:
        return True
    tag = str(getattr(node, "tag_name", "") or "").lower()
    return tag == "input" and str(attrs.get("type") or "").lower() in ("checkbox", "radio")


def _with_write_outcome(meta: dict[str, Any] | None,
                        outcome: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge a write_outcome stamp into existing ActionResult metadata WITHOUT replacing
    siblings (the compiler reads metadata["interacted_element"]; a stamp must never
    clobber it). None-in-None-out keeps unstamped results byte-identical."""
    if outcome is None:
        return meta
    return {**(meta or {}), "write_outcome": outcome}


def _stamp_interacted(meta: dict[str, Any] | None, node: Any) -> dict[str, Any] | None:
    """Record the element a click acted on in the result metadata, WITHOUT replacing
    siblings — the compiler reads metadata["interacted_element"] when browser-use's own
    post-action snapshot carries none. An existing stamp always wins (find_by_text has
    already named the exact node it clicked), and a capture failure is a no-op: None-in
    stays None-out so unstamped results are byte-identical."""
    if node is None or (isinstance(meta, dict) and meta.get("interacted_element")):
        return meta
    captured = _captured_element(node, "")
    if not captured:
        return meta
    return {**(meta or {}), "interacted_element": captured}


def _with_row(meta: dict[str, Any] | None,
              row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Attach the clicked element's row identity to the element already stamped in
    `meta`. No row, or no element to attach it to -> meta comes back untouched, so an
    unstamped result stays byte-identical."""
    if not row or not isinstance(meta, dict):
        return meta
    element = meta.get("interacted_element")
    if not isinstance(element, dict) or element.get("row"):
        return meta
    return {**meta, "interacted_element": {**element, "row": row}}


async def _click_outcome_suffix(browser_session, t0: float,
                                pre: dict[str, Any] | None,
                                node: Any = None,
                                ) -> tuple[str, dict[str, Any] | None]:
    """The NETWORK OUTCOME (write requests fired, with server verdicts) + DIALOG OUTCOME
    receipt suffix for a click dispatched at t0 whose pre-click dialog state was `pre`,
    plus the structured write_outcome stamp ({"fired","accepted","t0"}, None when no
    write fired) that the gate's receipt roll-up reads instead of parsing prose.
    Shared by the `click` override and find_by_text's click branch — a Save clicked via
    find_by_text used to fire its POST with no receipt at all (run 20260807_095537 seg6:
    the unreceipted DataRequest create was re-clicked into a duplicate)."""
    in_dialog = bool(isinstance(pre, dict) and pre.get("in_dialog"))
    # Toggle-family clicks get no save/submit doubt language (see _is_toggle_control) —
    # neither the no-write flag nor the dialog-closure advisory.
    advisory = in_dialog and not _is_toggle_control(node)
    suffix, fired, accepted = "", False, False
    if _LIVE_NETWORK is not None:
        text, fired, accepted = await _network_outcome(_LIVE_NETWORK, t0,
                                                       expect_write=advisory)
        suffix += text
    if advisory:
        closed = await _dialog_closed(browser_session, pre, fired)
        if closed is True:
            if accepted:
                suffix += (" — the dialog CLOSED after this click and the server "
                           "ACCEPTED the write above: this save is DONE — do NOT "
                           "reopen the dialog or enter the data again.")
            else:
                # Closed with no (clearly accepted) write: staged client-side saves
                # legitimately look like this, but so does a silently discarded form
                # (run 20260807_123259: Add Employee closed with no create POST, the
                # old unconditional "ACCEPTED" advisory waved it through, and the
                # whole run chased an employee that never existed).
                if fired:
                    why = ", but the write above did not clearly succeed"
                elif _LIVE_NETWORK is not None:
                    why = ", but NO write request fired"
                else:
                    why = ""
                suffix += (" — the dialog CLOSED after this click" + why +
                           ". If this was a save/submit, VERIFY the record/change "
                           "now exists (look it up in the list or on the page) "
                           "before reporting this step done — and only re-enter "
                           "the data if it is genuinely absent.")
        elif closed is False and accepted:
            # The write receipt above is ground truth; the still-open panel is NOT
            # evidence of failure (it may be a follow-up panel the save opened).
            suffix += (" — the dialog is STILL OPEN after this click, but the write "
                       "above SUCCEEDED: do NOT click this again and do NOT redo the "
                       "save/send. The open panel may be a FOLLOW-UP dialog the action "
                       "opened (e.g. an email panel after a save) or may close on its "
                       "own — re-read the page and continue from what it actually "
                       "shows.")
        elif closed is False and not fired and _LIVE_NETWORK is not None:
            # Writes WERE watched and none fired: that is what a client-side commit
            # looks like (an in-page calculator/picker that recomputes and closes its
            # own popup a render later), not evidence of failure. Run 20260817_163352
            # seg 4: the Net-to-Gross Calculate had already applied £4,000.00 to the
            # Feb-27 row when this branch called it a probable save failure — the agent
            # redid the work, dismissed the popup by re-clicking its pencil, and typed
            # the amount into the grid cell behind it. The observation stays; the
            # failure verdict does not.
            suffix += (" — the dialog is STILL OPEN this soon after the click, and no "
                       "write fired. A client-side commit looks exactly like this and "
                       "its popup can close a render later, so this is NOT evidence of "
                       "failure: READ THE PAGE for the change this dialog makes and "
                       "decide from what it shows. Re-enter the data only if the page "
                       "still shows the old value — and never re-click the control that "
                       "opened this popup while it is open, which only dismisses it.")
        elif closed is False:
            suffix += (" — the dialog is STILL OPEN after this click. If this was a "
                       "save/submit it likely did NOT go through: look for validation "
                       "messages inside the dialog before doing anything else.")
    outcome = {"fired": True, "accepted": accepted, "t0": t0} if fired else None
    return suffix, outcome


# From an anonymous row checkbox, the ROW's visible text is the only identity a receipt
# can offer: find the containing row element and return its (whitespace-normalized,
# capped) text. Run 20260817_124339 seg 6: the ticked checkbox's receipt named nothing,
# the agent believed it had ticked the noted employee, and the request saved for a
# different one two rows off — a false pass on real, wrong data.
_ROW_LABEL_JS = (
    "function(){ var t = function(s){ return String(s || '')"
    ".replace(/\\s+/g, ' ').trim(); };"
    " var r = this.closest('[role=\"row\"], tr, li, [class*=\"List-cell\"]');"
    " return r ? t(r.textContent).slice(0, 80) : ''; }"
)


# The ROW a click happened in, as compilable identity: which container it is, and the
# texts of its cells. Compile turns those into row-scoped selectors so a click with no
# name of its own is located by the row it belongs to instead of by row POSITION.
#
# Why it must exist: the Data Request grid's external-link icon has no name, every row
# carries one with the identical title, and its href embeds the RECORD id — so compile's
# only usable anchor was `.../div[2]/div[9]/...`. Run 20260827_104331 created CDR072,
# clicked whatever link sat at that position, and wrote 13 UpdateCal POSTs into CDR054, a
# request from the day before. A positional path resolves confidently onto the wrong row
# and nothing downstream can tell.
#
# PUA glyphs are stripped (Fluent renders its icons as literal text nodes — the 2026-08-21
# pencil trap) and each cell is capped, because a cell text becomes a selector literal.
_ROW_CELLS_JS = (
    "function(){ var t = function(s){ return String(s || '')"
    ".replace(/[\\uE000-\\uF8FF]/g, '').replace(/\\s+/g, ' ').trim(); };"
    " var SCOPES = [['[role=\"row\"]', '[role=\"row\"]'], ['tr', 'tr'], ['li', 'li'],"
    "               ['[class*=\"List-cell\"]', '[class*=\"List-cell\"]']];"
    " var r = null, sel = '';"
    " for (var s = 0; s < SCOPES.length && !r; s++) {"
    "   r = this.closest(SCOPES[s][0]); if (r) sel = SCOPES[s][1]; }"
    " if (!r) return null;"
    " var out = [], push = function (v) { v = t(v);"
    "   if (v && out.indexOf(v) < 0 && v.length <= 60) out.push(v); };"
    " var cells = r.querySelectorAll("
    "   '[role=\"gridcell\"], [role=\"cell\"], td, [class*=\"Row-cell\"]');"
    " for (var i = 0; i < cells.length && out.length < 8; i++)"
    "   push(cells[i].innerText || cells[i].textContent);"
    " if (!out.length) push(r.innerText || r.textContent);"
    " return { scope: sel, cells: out }; }"
)


def _named_element(node: Any) -> bool:
    """Does this element carry a name of its own? A named click is already guarded by
    expect_text at replay, so it needs no row scope; a NAMELESS one has nothing but its
    position and is the class of click that lands on the wrong row."""
    attrs = getattr(node, "attributes", None) or {}
    ax = getattr(getattr(node, "ax_node", None), "name", None)
    return bool(str(ax or "").strip() or str(attrs.get("aria-label") or "").strip())


async def _row_cells(browser_session, node) -> dict[str, Any] | None:
    """The clicked element's row scope + cell texts, or None. Best-effort: identity is
    bookkeeping and must never cost the click."""
    try:
        handle = await _field_handle(browser_session, node)
        if not handle:
            return None
        got = await _call_on_field(handle, _ROW_CELLS_JS)
    except Exception as exc:  # noqa: BLE001 - same degradation as _row_context_label
        logger.debug("row-cells probe failed: %s", exc)
        return None
    if not isinstance(got, dict) or not got.get("cells"):
        return None
    return {"scope": str(got.get("scope") or ""),
            "cells": [str(c) for c in got["cells"] if str(c).strip()]}


def _is_anonymous_toggle(node: Any) -> bool:
    """A checkbox/radio-ish element with no name of its own — the one click whose
    receipt would otherwise verify nothing."""
    attrs = getattr(node, "attributes", None) or {}
    toggle = (attrs.get("role") or "").strip().lower() in ("checkbox", "switch") \
        or (attrs.get("type") or "").strip().lower() in ("checkbox", "radio")
    return toggle and not (attrs.get("aria-label") or "").strip()


def _nothing_clicked(msg: str, click_first: bool) -> ActionResult:
    """Receipt for a find_by_text call that clicked NOTHING.

    The channel is decided by INTENT, not by why the click did not happen. With
    `click_first` the agent asked for an action, so a miss is a refusal and must ride the
    ERROR channel: browser-use's multi_act stops a step's remaining queued actions on an
    error, and on the success channel it does not. That is the exact hole the ambiguous
    branch closed in run 20260824_155123 — and the one the 0-match branch still had on
    2026-08-27, when a batched `find_by_text('Close', click_first) -> click(index)` sailed
    past a miss: the Payroll Review panel was never closed, the follow-up click fired
    against an unchanged page, and the half-trace was committed as if the close had
    happened (run 20260827_104331, entry 07044b6a0dbf7988).

    Without `click_first` the call IS a probe — "is this text here?" is a legitimate
    question whose answer can be no — so it keeps the success channel. `no_click` is
    stamped either way: compile must never read one of these as a click.
    """
    if click_first:
        return ActionResult(error=msg, metadata={"no_click": True})
    return ActionResult(extracted_content=msg, long_term_memory=msg,
                        include_in_memory=True, metadata={"no_click": True})


async def _row_context_label(browser_session, node) -> str | None:
    """Best-effort row text for an anonymous toggle, read BEFORE the click (the row as
    the agent saw it — ticking may re-render it). None degrades to today's receipt."""
    try:
        handle = await _field_handle(browser_session, node)
        if not handle:
            return None
        label = await _call_on_field(handle, _ROW_LABEL_JS)
        return (str(label).strip() or None) if label else None
    except Exception as exc:  # noqa: BLE001 - identity is garnish; the click must run
        logger.debug("row-context probe failed: %s", exc)
        return None


_SCRIPT_URL_RE = re.compile(r"^\s*(javascript|data)\s*:", re.IGNORECASE)


def _script_url_refusal(url: Any) -> "ActionResult | None":
    """Refuse a `navigate` to a script/document URL, or None to let it through.

    `evaluate` is excluded from this registry on purpose — JS writes are unrecordable, so a
    recording that leans on them compiles to a script missing those actions (see this
    module's docstring). `navigate` was the loophole that handed it back, and the loophole
    costs the page.

    Run 20260903_102354_153542 subtask 4: unable to address one of twelve identical "Net to
    gross" pencils, the agent sent
    `navigate(url="javascript:()=>{…querySelectorAll('tr')…}")`. browser-use's
    SecurityWatchdog blocks that — but only after the NavigateToUrlEvent is dispatched, and
    the block lands the tab on **about:blank**. The Pay Forecast page was gone; twelve of
    the segment's nineteen steps went on recovering, and it then did the same thing again.
    Three earlier authorings of that identical slice took three steps each.

    So the refusal happens BEFORE dispatch: the page keeps its state, and the receipt names
    the tools that actually reach a control — a refusal that only says "no" just sends the
    agent looking for the next door. `data:` is refused on the same ground: it is not a page
    in this app, and navigating to it replaces the one the segment is working in.

    ERROR channel, like every other refusal here — multi_act stops a step's remaining queued
    actions on an error, and both live attempts had further actions queued behind them.
    """
    if not _SCRIPT_URL_RE.match(str(url or "")):
        return None
    msg = (
        f"navigate REFUSED: {str(url)[:80]!r} is a script/document URL, not a page. This "
        "framework removes the `evaluate` action deliberately — JavaScript actions cannot "
        "be recorded or replayed, so anything done that way is lost from the run — and "
        "`navigate` is not a way around that. Nothing was navigated: the page is unchanged "
        "and your work on it stands, so do NOT recover from this. To reach a control, use "
        "find_by_text('<its name>', click_first=True), adding near_text='<text in its row>' "
        "when several look identical, or list_actions(near_text='<nearby heading or row>') "
        "and click the index it reports."
    )
    logger.warning("⛔ %s", msg)
    return ActionResult(error=msg, metadata={"no_click": True})


async def _click_with_dialog_outcome(builtin_click, params, browser_session) -> ActionResult:
    """Delegate the click to the built-in unchanged, then append the network+dialog
    outcome suffix and stamp the structured write_outcome. Plain clicks that fired no
    writes, probe failures, and error results pass through untouched. Anonymous
    checkbox/radio clicks additionally name the ROW they sit in (see _ROW_LABEL_JS)."""
    node = None
    index = getattr(params, "index", None)
    if browser_session is not None and index:
        try:
            node = await browser_session.get_element_by_index(index)
        except Exception:  # noqa: BLE001 - the built-in will report the real lookup error
            node = None
    if node is not None:
        # A plain click counts toward the segment's repeat ledger: subtask 15 of run
        # 20260901_122209 clicked Save & Next once by hand BEFORE calling repeat_click, so a
        # budget that ignored manual clicks would still overshoot by one. Refused BEFORE it
        # is counted (a refusal clicked nothing, so it must not inflate the ledger), and only
        # on a control repeat_click already repeated — see _refuse_if_over_budget.
        refusal = _refuse_if_over_budget(node, _control_name(node, index))
        if refusal is not None:
            return refusal
        _note_clicks(node, 1)
    row_label, row = None, None
    if node is not None and _is_anonymous_toggle(node):
        row_label = await _row_context_label(browser_session, node)
    if node is not None and not _named_element(node):
        # Identity for the RECORDING (not the receipt): a click with no name of its own
        # compiles to a positional row path, which resolves onto the wrong row on the
        # next run's data (see _ROW_CELLS_JS).
        row = await _row_cells(browser_session, node)
    pre = await _dialog_state(browser_session, node) if node is not None else None
    t0 = _stamp_action()
    res = await builtin_click(params=params, browser_session=browser_session)
    if res is None or getattr(res, "error", None) \
            or not getattr(res, "extracted_content", None):
        return res
    suffix, outcome = await _click_outcome_suffix(browser_session, t0, pre, node)
    if row_label:
        # In FRONT of the network/dialog outcome: the row identity is what the agent
        # must check against the employee it MEANT to tick.
        suffix = f' — in row "{row_label}"' + suffix
    meta = getattr(res, "metadata", None)
    # Stamp WHAT we clicked. browser-use fills state.interacted_element from the snapshot
    # it takes AFTER the action, which is empty when the click switched tabs: the
    # external-link click that opens the OTP tab recorded [null, null, null], and
    # compile's `elif name == "click" and element:` then dropped the one load-bearing
    # click of that segment without a word (run 20260825_090938). `node` is the PRE-click
    # element, so the stamp always names what we actually acted on. Same shape (and same
    # reason) as find_by_text's stamp.
    merged = _stamp_interacted(meta, node)
    merged = _with_row(merged, row)
    merged = _with_write_outcome(merged, outcome)
    if not suffix and merged is meta:
        return res
    update: dict[str, Any] = {}
    if suffix:
        update["extracted_content"] = res.extracted_content + suffix
        if getattr(res, "long_term_memory", None):
            update["long_term_memory"] = res.long_term_memory + suffix
    if merged is not meta:
        # model_copy REPLACES the metadata field wholesale — merged carries the siblings.
        update["metadata"] = merged
    return res.model_copy(update=update)


def build_tools() -> Tools:
    """Return a browser-use `Tools` registry with our custom actions added, `evaluate`
    removed (JS form-fills are unrecordable — see module docstring), and the built-in
    `input` overridden by the auto-Enter variant."""
    tools = Tools(exclude_actions=["evaluate"])

    # Same-name override of `click` that DELEGATES to the built-in (same param model and
    # description, so recorded history and the compiler's click branch are untouched) and
    # adds the dialog-outcome receipt suffix. Captured BEFORE registering the wrapper —
    # same-name registration replaces the entry, not the captured function. Note:
    # browser-use re-registers `click` when set_coordinate_clicking flips (claude-* /
    # gemini-3-pro model names); the configured agent models (gpt-4.1-mini, llama-4)
    # never trigger that, but a model switch would silently drop this wrapper — and with
    # it the write_outcome stamps the gate's receipt roll-up reads (fail-safe: the
    # roll-up simply finds no stamps and goes inert).
    _builtin_click = tools.registry.registry.actions["click"]
    _builtin_click_fn = _builtin_click.function

    @tools.action(_builtin_click.description, param_model=_builtin_click.param_model)
    async def click(params, browser_session=None) -> ActionResult:
        return await _click_with_dialog_outcome(_builtin_click_fn, params, browser_session)

    # Same-name override of `navigate` that DELEGATES to the built-in, refusing only
    # script/document URLs — see _script_url_refusal for the run that made this necessary.
    # Description, param model and terminates_sequence are carried over verbatim: from the
    # agent's side this must stay the same action, or the override silently changes how
    # every ordinary navigation is prompted.
    _builtin_navigate = tools.registry.registry.actions["navigate"]
    _builtin_navigate_fn = _builtin_navigate.function

    @tools.action(_builtin_navigate.description,
                  param_model=_builtin_navigate.param_model,
                  terminates_sequence=_builtin_navigate.terminates_sequence)
    async def navigate(params, browser_session=None) -> ActionResult:
        refusal = _script_url_refusal(getattr(params, "url", None))
        if refusal is not None:
            return refusal
        return await _builtin_navigate_fn(params=params, browser_session=browser_session)

    # Same-name override of `close` that DELEGATES to the built-in, refusing only the one
    # close that cannot be recovered from: the LAST open tab. Closing it destroys the page
    # the workflow runs in — browser-use then spawns a fresh about:blank with no history,
    # so the next segment has nothing to go back to and no URL to return to (run
    # 20260827_091313 subtask 8: the agent correctly closed the aux client tab, then closed
    # the app's own /datarequests tab one step later on an invented memory of a third tab;
    # subtask 9 opened on about:blank, guessed a URL, and died on ERR_NAME_NOT_RESOLVED).
    # Only the terminal case is guarded: a segment told to "close this tab" still closes its
    # aux tab normally, and the tab-count DROP that _recording_closed_its_page reads to skip
    # the end-context pin is unchanged.
    _builtin_close = tools.registry.registry.actions["close"]
    _builtin_close_fn = _builtin_close.function

    @tools.action(_builtin_close.description, param_model=_builtin_close.param_model)
    async def close(params, browser_session=None) -> ActionResult:
        try:
            remaining = len(await browser_session.get_tabs())
        except Exception as exc:  # noqa: BLE001 - never break the close path over a diagnosis
            logger.debug("close guard: unreadable tab list (%s) - delegating", exc)
            remaining = 0                      # fail OPEN: 0 skips the guard below
        if remaining == 1:
            msg = (f"REFUSED — did NOT close tab #{params.tab_id}: it is the LAST open tab, "
                   "and closing it destroys the page this workflow is running in. The next "
                   "step would start on a blank page with no history and no way back to the "
                   "app. If your instruction to close a tab meant the extra tab this workflow "
                   "opened, that tab is already closed — nothing further to close. Call done "
                   "instead.")
            logger.info("⛔ %s", msg)
            # error channel: multi_act stops the remaining queued actions, so a `done`
            # batched behind this close cannot report a close that never happened.
            return ActionResult(error=msg, metadata={"no_close": True})
        return _close_receipt(
            await _builtin_close_fn(params=params, browser_session=browser_session),
            str(params.tab_id))

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
        _stamp_action()   # verify_save_registered windows on the LAST acting verb
        node = await browser_session.get_element_by_index(params.index)
        if node is None:
            msg = (f"Element index {params.index} not available - page may have changed. "
                   "Try refreshing browser state.")
            logger.warning("⚠️ %s", msg)
            return ActionResult(extracted_content=msg)
        dropdown = _is_dropdown_filter(node)
        if dropdown and (params.text or "").strip():
            # Typing into a dropdown/combobox filter selects NOTHING: the text only
            # narrows the menu and is discarded when the menu closes, and the widget's
            # value lives outside the input, so the read-back below can never verify a
            # pick (observed live: 'Assets transferred' typed, no option clicked, field
            # reported as set, Save silently rejected). Refuse up front and name the
            # action that picks AND verifies — the fill-shaped twin of find_by_text's
            # <select> refusal. text="" (a pure clear) stays allowed for stuck filters.
            # When the field's identity is readable, say WHAT it is: the remedy repeats
            # the same index+text, which re-validates a wrong TARGET unless the message
            # lets the agent notice it (see _dropdown_descriptor).
            ident = await _dropdown_descriptor(browser_session, node)
            remedy = (f"select_dropdown(index={params.index}, text='{params.text}') — "
                      "it opens the menu, clicks the matching option, and verifies the "
                      "value took, all in one action.")
            if ident:
                msg = (f"REFUSED — did NOT type '{params.text}': element {params.index} "
                       f"is a dropdown/combobox filter ({ident}), and typed filter text "
                       "selects NOTHING (it is discarded when the menu closes). If that "
                       "is not the field this step needs, you have the WRONG element — "
                       "locate the control the task names instead. Otherwise call "
                       + remedy)
            else:
                msg = (f"REFUSED — did NOT type '{params.text}': element {params.index} "
                       "is a dropdown/combobox filter, and typed filter text selects "
                       "NOTHING (it is discarded when the menu closes). Call " + remedy)
            logger.info("⛔ %s", msg)
            # error channel: multi_act stops the remaining queued actions on it — a
            # refused fill must not let an already-queued Send/Save fire against the
            # value that never landed (run 20260807_095537 seg7: the refused From-fill's
            # queued Send went out with the wrong sender).
            return ActionResult(error=msg, metadata={"no_fill": True})
        # Clear with real keystrokes ourselves when we can reach the element, and hand
        # browser-use clear=False so its JS `value = ""` (invisible to React — see the
        # stubborn-field notes above) never runs. `params.clear` is left untouched: the
        # compiler reads it to decide whether the replayed fill clears too.
        handle = await _field_handle(browser_session, node)
        # Stale-index guard: a node an earlier action in THIS step re-rendered away still
        # resolves over CDP, so the clear/type below would run against a detached element
        # — keystrokes land in whatever holds focus and the read-back reads the dead node
        # (the false "did NOT take" that drove the duplicate-add loop). Route to the live
        # twin before any keystroke goes out.
        if handle and params.text and await _field_connected(handle) is False:
            return await _refind_fill(browser_session, node, params)
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
            # Mid-step detach: the type/read-back ran against a node that has since LEFT
            # the document — the read-back is self-consistent but the live form never saw
            # the value (the false-CLEAN twin of the guard above). Land it on the live
            # field, and never press Enter into a random focus target.
            if handle and params.text and await _field_connected(handle) is False:
                return await _refind_fill(browser_session, node, params)
            # Date pickers keep the pre-08-05 sequence that always filled them cleanly
            # (4523b58 parity, user-requested): clear → type → read-back → NO Enter; the
            # date commits when focus moves on. With the calendar callout open, Enter is
            # handled by the picker itself (it can commit the callout's highlighted date
            # over the typed text) — and the read-back above runs BEFORE Enter, so such a
            # clobber would wear a clean receipt.
            date_picker = _is_date_picker(node)
            # Third suppression class beside the two above: a field inside a transient
            # layer/callout commits via the popup's own button, never via Enter.
            in_popup = (not dropdown and not date_picker and bool(handle)
                        and await _in_layer_popup(handle))
            if not dropdown and not date_picker and not in_popup:
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
        meta["auto_enter"] = not dropdown and not date_picker and not in_popup
        if dropdown:
            msg = (f"Typed '{params.text}' (dropdown filter — Enter suppressed; "
                   "click the option you want)")
        elif date_picker:
            msg = (f"Typed '{params.text}' (date picker — Enter suppressed; the date "
                   "commits when you move to the next field. Do NOT open the calendar.)")
        elif in_popup:
            msg = (f"Typed '{params.text}' (popup input — Enter suppressed; click the "
                   "popup's own confirm button)")
        else:
            msg = f"Typed '{params.text}' and pressed Enter"
        if verified is not None and not _value_took(params.text, verified):
            # The page REFUSED this value, so the action is a phantom — it reports what the
            # tool tried, not what the field holds. Stamped the way every other did-not-take
            # path stamps it (the stale-index refind above, paste_text, select_dropdown), so
            # the compiler drops it by metadata rather than by parsing this prose. Content
            # channel, not the error channel: the agent recovers from this on its own, and
            # receipt_rollup's refusal rule wants an error beside the stamp.
            # Without it, subtask 13 of the payroll e2e committed a fill the page had
            # rejected and re-authored itself every run (20260901_110833).
            meta["no_fill"] = True
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
        'custom comboboxes (react-select etc.). ADDRESS IT BY THE LABEL BESIDE IT — '
        'select_dropdown(near_text="From", text="no-reply") — which is the form to use '
        'whenever the task names the field by its label. A custom dropdown usually has NO '
        'name of its own (its label is a separate piece of text), so no text search can '
        'find the control itself; near_text matches on that neighbouring label for you and '
        'needs no index. Pass index instead ONLY when you already have the combobox '
        "input's index. Either way the tool opens the menu, picks the matching option and "
        'verifies it took, in one action. If your text is not among the options, the error '
        'lists what the dropdown ACTUALLY offers.',
    )
    async def select_dropdown(text: str, near_text: str = "", index: int = -1,
                              browser_session=None) -> ActionResult:  # injected by name
        _stamp_action()   # verify_save_registered windows on the LAST acting verb
        if browser_session is None:
            return ActionResult(error="select_dropdown: BrowserSession not injected")
        if near_text.strip():
            by_label = await _select_dropdown_by_label(browser_session, near_text, text)
            if not by_label.error or index < 0:
                return by_label
            logger.info("🔽 near_text '%s' did not resolve (%s); falling back to index %d",
                        near_text, by_label.error, index)
        if index < 0:
            return ActionResult(error=(
                "select_dropdown needs a target: pass near_text='<the label beside the "
                "dropdown>' (preferred — no index needed), or index=<the combobox "
                "input's index>."))
        params = SelectDropdownOptionAction(index=index, text=text)
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
        return await _fail_and_stop_result(reason)

    # The page-MOVING built-ins get the same callout guard as our own scroll tools
    # (find_text's entire job is "Scroll to text"). Same-name registration overrides
    # them; the originals are captured FIRST, exactly as for `click` above — and with
    # the same caveat: browser-use re-registers built-ins when set_coordinate_clicking
    # flips, which would silently drop these wrappers.
    _builtin_scroll = tools.registry.registry.actions["scroll"]
    _builtin_scroll_fn = _builtin_scroll.function

    @tools.action(_builtin_scroll.description, param_model=_builtin_scroll.param_model)
    async def scroll(params, browser_session=None) -> ActionResult:
        refused = await _refuse_if_callout(browser_session, "scroll the page")
        if refused is not None:
            return refused
        return await _builtin_scroll_fn(params=params, browser_session=browser_session)

    _builtin_find_text = tools.registry.registry.actions["find_text"]
    _builtin_find_text_fn = _builtin_find_text.function

    @tools.action(_builtin_find_text.description,
                  param_model=_builtin_find_text.param_model)
    async def find_text(params, browser_session=None) -> ActionResult:
        refused = await _refuse_if_callout(browser_session, "scroll to that text")
        if refused is not None:
            return refused
        return await _builtin_find_text_fn(params=params, browser_session=browser_session)

    _builtin_send_keys = tools.registry.registry.actions["send_keys"]
    _builtin_send_keys_fn = _builtin_send_keys.function

    @tools.action(_builtin_send_keys.description,
                  param_model=_builtin_send_keys.param_model)
    async def send_keys(params, browser_session=None) -> ActionResult:
        _stamp_action()   # Enter here commits forms, so this verb can fire the save
        keys = str(getattr(params, "keys", "") or "")
        if any(k.strip().lower() in _PAGE_SCROLL_KEYS
               for k in keys.replace("+", " ").split()):
            refused = await _refuse_if_callout(browser_session, f"press {keys}")
            if refused is not None:
                return refused
        return await _builtin_send_keys_fn(params=params, browser_session=browser_session)

    @tools.action(
        "Scroll EVERY scrollable container (side panels, dialog lists, popup grids) down "
        "by `pages` of its own height and report how many moved. Page-level scrolling "
        "moves the page BEHIND a fixed side panel, not the panel — use THIS when a "
        "panel or dialog owns its own scrollbar and its list must be scrolled to reveal "
        "rows further down (virtualized lists render rows only near their scroll "
        "position). Repeat until the target row is visible or it reports 0 moved — "
        "0 means every container is at its end and the list is fully revealed. "
        "Element indexes captured BEFORE this scroll are STALE afterwards: NEVER queue "
        "a click-by-index behind this action in the same step — after scrolling, "
        "re-read the page, confirm the target row's NAME is visible, then click it."
    )
    async def scroll_panels(pages: float = 0.8, browser_session=None) -> ActionResult:  # injected by name; do not annotate
        # Even container scrolling is refused while a callout is open: SCROLL_CONTAINERS_JS
        # moves EVERY scrollable container, including the ones outside the popup.
        refused = await _refuse_if_callout(browser_session, "scroll the containers")
        if refused is not None:
            return refused
        frac = max(0.2, min(float(pages), 1.0))
        # Virtualized rows render only after the scroll commits, so the helper scrolls AND
        # waits — the tool used to keep its own copy of that beat while find_by_text's hunt
        # had none, which is exactly how the rule went missing where it mattered most.
        moved = await _scroll_and_settle(browser_session, frac)
        if moved:
            msg = (f"scroll_panels: scrolled {moved} container(s) down {frac} page(s) — "
                   "element indexes from before this scroll are now STALE; re-read the "
                   "page and verify the target row's name before any click.")
        else:
            msg = ("scroll_panels: no container moved — every scrollable container is "
                   "already at its end, the list is fully revealed. If the target is "
                   "still not on screen it is NOT in this list.")
        logger.info("📜 %s", msg)
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

    @tools.action(
        "Copy the element at `index` to the clipboard and remember it under `label` (short "
        "snake_case role name, e.g. otp, reference_no). Use it when a value the page SHOWS "
        "has to be entered somewhere else, then deliver it with paste_text — the pair "
        "keeps the value out of your own retyping, and future replays re-read it fresh. "
        "Read-only: clicks nothing, changes nothing."
    )
    async def copy_text(index: int, label: str, browser_session=None) -> ActionResult:  # injected by name; do not annotate
        slug = re.sub(r"[^a-z0-9_]+", "_", (label or "").strip().lower()).strip("_") or "value"
        if browser_session is None:
            return ActionResult(error="copy_text: BrowserSession not injected")
        node = await browser_session.get_element_by_index(index)
        if node is None:
            return ActionResult(error=f"copy_text: element index {index} is not available "
                                      f"— re-read the page and use a current index.")
        handle = await _field_handle(browser_session, node)
        value = normalize_block_text(await _field_value(handle) or "") if handle else ""
        if not value:
            # No metadata on a miss, exactly as extract_data: a valueless copy must compile
            # to NOTHING rather than to a step that fails every replay.
            msg = (f"copy_text: element {index} has no readable text, so NOTHING was "
                   f"copied. Point at the element that SHOWS the value.")
            logger.info("📋 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True)
        _CLIPBOARD.update(value=value, label=slug)
        on_clipboard = await _clipboard_write(browser_session, value)
        msg = f"copy_text: {slug} = '{value[:200]}'"
        if not on_clipboard:
            # Honest, and harmless: paste_text delivers from the remembered value, and it
            # re-seeds the real clipboard itself when it needs the native rung.
            msg += " (the system clipboard refused the write; paste_text still has it)"
        logger.info("📋 %s", msg)
        return ActionResult(
            extracted_content=msg, long_term_memory=msg, include_in_memory=True,
            # The extract channel: the value joins run_values/values.json exactly as an
            # extract_data capture does, so a later segment's binding can resolve against it.
            metadata={"extract": {"label": slug, "value": value, "query": f"index {index}",
                                  "interacted_element": _captured_element(node, "")},
                      "interacted_element": _captured_element(node, ""),
                      "no_click": True},
        )

    @tools.action(
        "Paste a whole value into the element at `index`. Use it when ONE value is split "
        "across SEVERAL inputs — a 6-box one-time code, a PIN, date parts: click the first "
        "box, then paste the complete value here and the widget spreads it across the "
        "boxes. Pass `text` to paste that, or leave it empty to paste what copy_text last "
        "captured. For an ordinary single field keep using input."
    )
    async def paste_text(index: int, text: str = "", browser_session=None) -> ActionResult:  # injected by name; do not annotate
        _stamp_action()   # verify_save_registered windows on the LAST acting verb
        target = (text or "").strip() or _CLIPBOARD.get("value", "")
        if not target:
            return ActionResult(error="paste_text: nothing to paste — pass `text`, or call "
                                      "copy_text first to capture the value.")
        if browser_session is None:
            return ActionResult(error="paste_text: BrowserSession not injected")
        node = await browser_session.get_element_by_index(index)
        if node is None:
            return ActionResult(error=f"paste_text: element index {index} is not available "
                                      f"— re-read the page and use a current index.")
        if _is_dropdown_filter(node):
            # Same trap as a typed fill: text put into a combobox filter selects NOTHING.
            ident = await _dropdown_descriptor(browser_session, node)
            msg = (f"REFUSED — did NOT paste '{target}': element {index} is a "
                   f"dropdown/combobox filter{(' (' + ident + ')') if ident else ''}, and "
                   f"filter text selects nothing. Call select_dropdown(index={index}, "
                   f"text='{target}') instead.")
            logger.info("⛔ %s", msg)
            return ActionResult(error=msg, metadata={"no_fill": True})
        handle = await _field_handle(browser_session, node)
        if handle is None:
            return ActionResult(error=f"paste_text: could not reach element {index} in the "
                                      f"live DOM, so nothing was pasted.",
                                metadata={"no_fill": True})
        if await _field_connected(handle) is False:
            # The node an earlier action in THIS step re-rendered away: pasting into it
            # would read back clean while the visible widget never changed.
            return ActionResult(error=f"paste_text: element {index} has left the page (it "
                                      f"was re-rendered), so nothing was pasted. Re-read "
                                      f"the page and paste into the current element.",
                                metadata={"no_fill": True})
        # The rung ladder, mirroring script_compile._paste_into step for step so a replay
        # delivers the value exactly as the recording did. Each rung is judged by READING
        # THE WIDGET BACK — never by its own return value, since a handler that CONSUMED a
        # paste reports the same "prevented" as one that ignored it — and the group is
        # cleared between rungs so a partial landing cannot poison the next.
        landed, rung, shows = False, "", ""
        try:
            await _call_on_field(
                handle, "function(t){ return (%s)(this, t); }" % _PASTE_EVENT_JS, [target])
        except Exception as exc:  # noqa: BLE001 - a rung that throws is a rung that failed
            logger.debug("synthetic paste failed: %s", exc)
        await asyncio.sleep(0.15)
        landed, shows = _paste_took(target, await _paste_reading(handle))
        if landed:
            rung = "paste event"
        if not landed:
            # Rung 2: real per-character key events. Ahead of the browser paste command on
            # purpose — a widget that advances focus box-to-box as you type fills correctly
            # with no paste handler at all, typing cannot truncate the way a paste into a
            # maxlength=1 box does, and it needs neither a clipboard permission nor a
            # secure origin.
            await _clear_paste_group(handle)
            try:
                keys = browser_session.event_bus.dispatch(SendKeysEvent(keys=target))
                await keys
                await keys.event_result(raise_if_any=True, raise_if_none=False)
            except Exception as exc:  # noqa: BLE001 - reported by the read-back below
                logger.debug("keystroke paste failed: %s", exc)
            await asyncio.sleep(0.15)
            landed, shows = _paste_took(target, await _paste_reading(handle))
            if landed:
                rung = "keystrokes"
        if not landed:
            # Rung 3: Chrome's own paste command — a genuine isTrusted event from the real
            # clipboard, for a widget that distributes on paste and rejects both of the
            # above. Measured to fill a six-box isTrusted-only widget that neither other
            # rung could.
            await _clear_paste_group(handle)
            if await _clipboard_write(browser_session, target) and await _native_paste(handle):
                await asyncio.sleep(0.15)
                landed, shows = _paste_took(target, await _paste_reading(handle))
                if landed:
                    rung = "browser paste"
        if not landed:
            # ERROR channel so multi_act stops: a queued Proceed/Save must never fire on a
            # field that did not take the value.
            msg = (f"paste_text: '{target}' did NOT land in element {index} — it now shows "
                   f"'{shows}'. Do NOT report this field as set. Click the FIRST box of the "
                   f"widget and paste again, or enter the value one character per box.")
            logger.info("⛔ %s", msg)
            return ActionResult(error=msg, metadata={"no_fill": True})
        msg = f"Pasted '{target}' into element {index} via {rung} (the widget now shows '{shows}')"
        logger.info("📋 %s", msg)
        return ActionResult(
            extracted_content=msg, long_term_memory=msg,
            # `paste.value` is what compile records as the step's value, so the provenance
            # binder sees the WHOLE value and can bind it to the run data it came from.
            metadata={"paste": {"value": target, "rung": rung},
                      "interacted_element": _captured_element(node, "")},
        )

    @tools.action(
        "Click ONE control repeatedly — the way to do a repeated step (Save & Next through a "
        "run of employees, Next through the remaining rows). Pass times=N for exactly N "
        "clicks, or times=0 to keep clicking until the control stops advancing — use times=0 "
        "when the task says 'all the remaining ...' and names no number. Between clicks it "
        "waits for the control to be clickable again, so a slow load delays the cadence "
        "instead of eating a click, and it tells you how many clicks actually landed. Use "
        "this instead of calling click N times: it cannot lose count, and it records as ONE "
        "repeatable step."
    )
    async def repeat_click(index: int, times: int = 0, browser_session=None) -> ActionResult:  # injected by name; do not annotate
        """The live twin of SkillApi.repeat_click (skills/api.py).

        Repetition had no first-class expression before this: the agent issued N separate
        clicks and the COMPILER guessed afterwards whether adjacent same-target clicks were
        iterations or slow-app retries — a guess keyed on `kind: loop`, itself inferred from
        the prompt's wording. One counted call removes both the guess and the declaration:
        the step carries its own count, which is exactly the shape codegen already turns
        back into `api.repeat_click` and run_steps already loops.
        """
        _stamp_action()   # verify_save_registered windows on the LAST acting verb
        if browser_session is None:
            return ActionResult(error="repeat_click: BrowserSession not injected")
        node = await browser_session.get_element_by_index(index)
        if node is None:
            return ActionResult(
                error=f"repeat_click: element index {index} is not available — re-read the "
                      f"page and use a current index.",
                metadata={"no_click": True})
        label = _control_name(node, index)
        want = max(0, int(times or 0))
        # Budget check BEFORE clicking anything. The ledger already holds the clicks this
        # segment has spent on this control — including plain `click` calls, so the agent's
        # manual first click counts — and the segment's declared "exactly N" caps the total.
        already = _CLICK_LEDGER.get(_ledger_key(node), 0)
        if _REPEAT_BUDGET is not None:
            remaining = _REPEAT_BUDGET - already
            if remaining <= 0:
                # SATISFIED, not refused — and the difference decided run
                # 20260903_114110_507756 subtask 15. repeat_click(5) had landed all five;
                # the agent then rewrote its own memory between two consecutive steps
                # ("Completed 5 further Save & Next clicks" -> "Next: advance through five
                # more"), called this four more times, and reported the step incomplete
                # QUOTING the old wording: "repeat_click was refused and no further
                # navigation occurred ... this final action was not executed". A correct
                # segment was recorded FAILED and the run stopped with fifteen good subtasks
                # behind it.
                #
                # The agent asked for N clicks on this control. N clicks on this control
                # exist. Its request HOLDS, so the honest answer is success with zero work
                # done. "REFUSED — did NOT click" is a sentence about failure, and it was
                # read as one. (The plain-click twin, _refuse_if_over_budget, already says
                # "and those clicks all landed"; this branch never carried that clause.)
                #
                # Dropping the ERROR channel here does NOT reopen run 20260901_122209's
                # hole. That channel stops batched follow-ups from firing against a page the
                # refused action left unchanged-but-expected-to-change; here nothing is
                # clicked, so the page really is unchanged and no index goes stale. The
                # budget also stays enforced on the other route: a batched plain click on
                # this control still meets _refuse_if_over_budget, the guard that stopped
                # ten employees being paid for a five-employee slice in run 20260902_105732.
                #
                # long_term_memory, like the success receipt: the completion fact has to
                # outlive the agent's memory rewrite, or the only line that survives the turn
                # is the one framing the work as not done.
                msg = (f"repeat_click: 0 additional clicks were needed — {label} has already "
                       f"been clicked {already} time(s) in this step, which is the "
                       f"{_REPEAT_BUDGET} this step asks for, and those clicks all landed. "
                       f"The repetition is COMPLETE and this call changed nothing on the "
                       f"page. This is NOT a failure of your step — do not repeat it and do "
                       f"not report the step incomplete because of it. Verify the end state, "
                       f"and if the rest of your step is done call done with success=true.")
                logger.info("🔁 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True, metadata={"no_click": True})
            if want and want > remaining:
                logger.info("🔁 repeat_click: %s asked for %d but only %d of this step's "
                            "%d remain — clamping", label, want, remaining, _REPEAT_BUDGET)
                want = remaining
            if not want:
                # times=0 under a declared budget: the number IS known, so use it rather than
                # clicking until the control dies (which would run the whole list).
                want = remaining
        # times=0 means "until it stops advancing"; the hard cap is a runaway guard, not an
        # expected stop — reaching it is reported as a failure, never as a finished list.
        cap = want or _REPEAT_HARD_CAP
        # Read BEFORE the first click so a one-iteration relabel is caught too.
        base_name = await _repeat_live_name(browser_session, node)
        done, stopped, relabel = 0, "", ""
        for _ in range(cap):
            try:
                event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                res = await event.event_result(raise_if_any=True, raise_if_none=False)
            except Exception as exc:  # noqa: BLE001 - report the count that DID land
                stopped = f"the click failed ({exc})"
                break
            # browser-use also refuses a click by RETURNING {'validation_error': ...}.
            if isinstance(res, dict) and res.get("validation_error"):
                stopped = f"the click was refused ({res['validation_error']})"
                break
            done += 1
            if done >= cap:
                break
            await asyncio.sleep(_REPEAT_SETTLE_S)
            ready, why = await _repeat_ready(browser_session, node)
            if not ready:
                stopped = why
                break
            now_name = await _repeat_live_name(browser_session, node)
            if _repeat_relabelled(base_name, now_name):
                relabel = now_name
                stopped = (f"the control is now named '{now_name}' (it was "
                           f"'{base_name}') — a DIFFERENT control, not another iteration")
                break

        # Ledger BEFORE the verdicts below: the clicks landed whether or not the count came
        # out right, and the next call must account for them.
        total = _note_clicks(node, done) if done else already
        if done:
            # This control is now what the segment's budget is ABOUT, which is what lets the
            # plain-click paths enforce it without capping unrelated buttons.
            try:
                _REPEAT_TARGETS.add(_ledger_key(node))
            except Exception:  # noqa: BLE001 - accounting must never break the receipt
                pass
        spent = (f" {total} click(s) on it in this step so far"
                 + (f" of the {_REPEAT_BUDGET} this step asks for." if _REPEAT_BUDGET
                    else ".")) if total != done else ""
        if want:
            if done < want:
                # A shortfall rides the ERROR channel so a queued Submit cannot fire on a
                # half-done run, and carries NO repeat metadata: a wrong count must never
                # become a cached step (same rule as a failed paste).
                msg = (f"repeat_click: clicked {label} only {done} of the {want} times asked "
                       f"— {stopped or 'the control stopped being clickable'}. Do NOT report "
                       f"this step done: check how far the list actually got before acting.")
                logger.info("⛔ %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
            msg = (f"Clicked {label} {done} times ({done} of {want} asked).{spent} "
                   f"The repetition is COMPLETE — do NOT call repeat_click on {label} again "
                   f"in this step; verify the end state and move on.")
        else:
            if not done:
                msg = (f"repeat_click: {label} was not clickable at all "
                       f"({stopped or 'no reason reported'}) — nothing was clicked.")
                logger.info("⛔ %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
            if not stopped:
                msg = (f"repeat_click: clicked {label} {done} times and it was STILL "
                       f"advancing at the {_REPEAT_HARD_CAP}-click safety cap — this is not "
                       f"a finished list. Check the page before reporting this step done.")
                logger.info("⛔ %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
            if relabel:
                # NOT "the run is complete": the list ended, but a different control is
                # now sitting where the repeated one was, and the task may well name it.
                # Clicking it as its OWN action is what puts a real step in the recording
                # — the swallowed-Submit bug is precisely a missing step, not a wrong one.
                msg = (f"Clicked {label} {done} times, then STOPPED: {stopped}. The "
                       f"repetition is COMPLETE at {done}.{spent} '{relabel}' is a "
                       f"SEPARATE control — if the task asks you to act on it, click it "
                       f"as its OWN action; do NOT call repeat_click on it.")
            else:
                msg = (f"Clicked {label} {done} times, until it stopped advancing "
                       f"({stopped}) — the run is complete at {done}.{spent} Do NOT call "
                       f"repeat_click on {label} again in this step.")
        logger.info("🔁 %s", msg)
        return ActionResult(
            extracted_content=msg, long_term_memory=msg, include_in_memory=True,
            # `repeat.count` is what compile writes onto the click step, so the replay does
            # the same number of clicks this run did — no adjacency guessing.
            metadata={"repeat": {"count": done, "until_done": not want,
                                 "wait_s": _REPEAT_SETTLE_S},
                      "interacted_element": _captured_element(node, label)},
        )

    @tools.action(
        "Find interactive elements matching `text`, searched in a FRESH page snapshot. Matches "
        "when every word of `text` appears in the element's visible text, aria-label, title, "
        "placeholder, value, name, or id (case-insensitive; punctuation ignored, so '+ Invoice' "
        "matches a button labeled 'Invoice' or id 'btnInvoice'). Returns every match with its "
        "CURRENT click index — use click(index) with that index immediately. Use this to locate "
        "a specific button/link/tab/menu item by its label, especially after an 'Element index "
        "N not available' failure. Pass click_first=True to also click it when exactly one "
        "element matches. When SEVERAL controls share the same name — one icon per grid row, "
        "one checkbox per employee — pass near_text='<text in the row you want>' to scope the "
        "search to that row; that clicks the right one directly, with no index to pick."
    )
    async def find_by_text(text: str, click_first: bool = False, near_text: str = "", browser_session=None) -> ActionResult:  # injected by name; do not annotate
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
        near_tokens = [t for t in re.split(r"[^a-z0-9]+", (near_text or "").strip().lower()) if t]
        # The ROW each candidate sits in — the only thing that tells twelve identical "Net to
        # gross" pencils apart. Probed once, then used BOTH to scope (near_text) and to label
        # the listing. Skipped when there is nothing to disambiguate.
        rows: dict[int, str] = {}
        if matches and (near_tokens or len(matches) > 1):
            rows = await _row_labels(browser_session, matches)
        # `matches` guard: scoping may only NARROW candidates that exist. Without it a query
        # that matched nothing returned "0 element(s) carry 'X', but NONE of them sits in a
        # row containing 'Y'" — which blames the scope for a miss it did not cause — and,
        # worse, returned BEFORE the 0-match path, so passing near_text silently disabled the
        # raw-DOM search and its scroll hunt, the only way to reach a 0-size or off-screen
        # control (run 20260903_122807_063006 subtask 16, steps 5 and 6). Falling through is
        # safe: the raw path refuses ambiguity itself, and the wrong-row hazard needs several
        # same-named controls to exist in the first place.
        if near_tokens and matches:
            scoped = [m for m in matches if _row_matches(rows.get(m[0]), near_tokens)]
            if not scoped:
                # FAILS CLOSED. Falling back to the unscoped candidates would click a row the
                # agent did not ask for, which is the failure this machinery exists to
                # prevent (run 20260817_124339 paid AARAN instead of Harris Duncan off a
                # stale positional index). ERROR channel: a click_first that clicked nothing
                # is a refusal, and multi_act must stop the rest of the batch.
                seen = "; ".join(sorted({r[:40] for r in rows.values()})[:6]) or "none readable"
                msg = (f"find_by_text('{query}', near_text='{near_text}'): {len(matches)} "
                       f"element(s) carry '{query}', but NONE of them sits in a row "
                       f"containing '{near_text}' — nothing was clicked. Rows seen: {seen}. "
                       f"Either the row is not rendered yet (scroll the list itself), or the "
                       f"text you scoped by is not in that row — check it against the rows "
                       f"listed above and try the wording they use.")
                logger.info("🔎 %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
            matches = scoped

        def _line(idx: int, node: Any, label: str) -> str:
            attrs = node.attributes or {}
            parts = [f"index={idx} <{node.tag_name}> text='{label[:80]}'"]
            for attr in ("id", "aria-label"):
                if attrs.get(attr):
                    parts.append(f"{attr}='{attrs[attr][:60]}'")
            # The icon hint is often the ONLY reason a query matched, and leaving it out
            # makes a correct click look like a wrong one. Run 20260903_122807_063006
            # subtask 16: find_by_text('Bulk upload FPS') clicked the right control and
            # reported "text=' FPS' id='btnFPS'" — the app labels that button plain FPS and
            # carries "Bulk upload" only in its icon, which _matching_nodes folds into the
            # haystack and this line then dropped. The agent read the receipt, concluded it
            # had "mis-clicked the manual FPS button" (a control that does not exist; it
            # coined the name), and spent ten steps closing the panel it had just opened.
            # list_actions has always shown these hints — see its _decoded.
            hints = _descendant_icon_hints(node)
            if hints and hints.lower() not in " ".join(parts).lower():
                parts.append(f"icon='{hints[:60]}'")
            # Without this, an ambiguous listing of row-scoped icons is N byte-identical
            # lines and the receipt's own "pick the right index" cannot be followed
            # (run 20260903_102354_153542: 24 of them, then improvised JavaScript).
            if rows.get(idx):
                parts.append(f"row='{rows[idx][:60]}'")
            return " ".join(parts)

        page_url = getattr(state, "url", "") or ""
        if not matches:
            # Fallback: the target may exist in the DOM but be EXCLUDED from the interactive
            # snapshot (a 0x0 button in a Fluent virtualized ScrollablePane, an off-screen
            # control). Query the live DOM directly and, when click_first, click it via its
            # own handler — the only way to reach a functional 0x0 element.
            raw = None
            popup = False
            try:
                expr = _RAW_FIND_JS % (json.dumps(tokens), "true" if click_first else "false")
                raw = await _eval_js(browser_session, expr)
                # Still nothing? A side panel or dialog list owns its own scroll box and
                # renders only the rows near its scroll position, so a row outside the
                # window is not in the DOM at all yet — ABOVE it just like below it.
                # Reset every scroll box to the TOP first (a down-only sweep from
                # mid-list can never reach a target above it — run 20260817_133135),
                # then advance them and look again (the employee picker's new hire was
                # unreachable below the fold — run 20260814_105247, seg 6). The page's
                # own scroll is included both ways.
                #
                # NOT while a CALLOUT is open. A callout owns the screen: what it holds
                # is never behind a page scroll, so the hunt has nothing to win — and a
                # Fluent Callout DISMISSES ON SCROLL, so it has the popup to lose.
                # (Panels/modals are excluded from the probe on purpose — their lists
                # are exactly what the hunt is for; see _CALLOUT_OPEN_JS.)
                # Run 20260817_163352 seg 4: find_by_text('Net amount') — a LABEL, which
                # RAW_FIND_JS (controls only) can never match — swept ~22 scrollTops
                # across the open Net-to-Gross callout; the static-text probe below then
                # honestly reported the label "not in this page's DOM" because the sweep
                # had just closed the popup it lived in.
                if not (raw and not raw.get("error") and raw.get("count")):
                    popup = await _callout_open(browser_session)
                if not popup:
                    if not (raw and not raw.get("error") and raw.get("count")):
                        # Settled, not raced: the reset mounts a different window and the
                        # query must see IT, not the one the page had a frame ago.
                        await _scroll_tops_and_settle(browser_session)
                        raw = await _eval_js(browser_session, expr)
                    for _ in range(_PANEL_SCROLL_ROUNDS):
                        if raw and not raw.get("error") and raw.get("count"):
                            break
                        # _scroll_and_settle, never the raw script: re-querying in the same
                        # tick as the scroll reads the PREVIOUS render window, and twenty
                        # rounds of that walk a virtualized list without ever seeing it
                        # (run 20260903_110957_989862 subtask 6 — the employee was there).
                        moved = await _scroll_and_settle(browser_session, 0.8)
                        if not moved:
                            break
                        raw = await _eval_js(browser_session, expr)
                    if not (raw and not raw.get("error") and raw.get("count")):
                        # A failed hunt must not leave the page parked at the bottom: the
                        # agent's next snapshot should show the page's head, not its floor
                        # (the manual scroll-up from run 20260817_133135, automated).
                        await _scroll_tops_and_settle(browser_session)
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
                        captured = {
                            "node_name": str(el.get("tag") or ""),
                            "attributes": dict(el.get("attrs") or {}),
                            "ax_name": str(raw.get("name") or "").strip(),
                            "hidden_click": True,
                        }
                        if el.get("xpath"):
                            # The exact location, so compile can anchor this click like
                            # any other instead of recording a text search for it.
                            captured["x_path"] = str(el["xpath"])
                        meta = {"interacted_element": captured}
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
                    return _nothing_clicked(msg, click_first)
                names = ", ".join(f"'{n}'" for n in (raw.get("names") or []) if n)
                msg = (f"find_by_text('{query}'): {raw['count']} match(es) exist in the DOM but "
                       f"are NOT clickable via index (0-size/virtualized): {names}. Re-call "
                       f"find_by_text('{query}', click_first=true) to click the best match directly.")
                logger.info("🔎 %s", msg)
                return _nothing_clicked(msg, click_first)
            # Last probe before claiming absence: the query may exist as STATIC text — a
            # label/heading with no control shape ('Period to' in the shadow-DOM modal,
            # run 20260805_123407_334719). The old receipt asserted "not in this page's
            # DOM" for text visibly on screen, and the agent left to hunt other pages.
            static = None
            try:
                static = await _eval_js(browser_session,
                                        _RAW_TEXT_FIND_JS % json.dumps(tokens))
            except Exception as exc:  # noqa: BLE001 - probe is best-effort
                logger.debug("find_by_text static-text probe failed: %s", exc)
            found = ""
            if isinstance(static, dict) and not static.get("error") and static.get("count"):
                found = " ".join(str(static.get("name") or "").split())[:200]
            else:
                found = _state_text_witness(state, tokens, query)
            if found:
                msg = (f"find_by_text('{query}'): no clickable control matches, but the "
                       f"text EXISTS on this page as STATIC text: '{found}'. It is a "
                       "label/heading, not a control — the section IS on this page; do "
                       "NOT navigate away, do NOT scroll-hunt for it, and do NOT click "
                       "an index near it. To operate the field BESIDE this label, "
                       "address it BY the label: for a dropdown/combobox call "
                       f"select_dropdown(near_text='{query}', text='<the option>') — no "
                       "index lookup. For any other field use its own index from the "
                       "page state. Nothing was clicked.")
                logger.info("🔎 %s", msg)
                # no_click: a probe that touched nothing — same compile rule as below.
                return _nothing_clicked(msg, click_first)
            if popup:
                # A popup IS open and nothing in it (or behind it) matches. Sending the
                # agent scroll-hunting here is the one move that destroys the popup, and
                # so is re-clicking the control that opened it — both happened in run
                # 20260817_163352 seg 4 while the Net-to-Gross callout was on screen.
                msg = (
                    f"find_by_text('{query}'): no match, and a POPUP/dialog is open on "
                    f"this page ({page_url}). Its fields are in the page state — read "
                    "them by index rather than by name (a popup's inputs are often "
                    "unlabelled, so the label beside the box is NOT the control's own "
                    "text). Do NOT scroll and do NOT re-click the control that opened "
                    "the popup: either one dismisses it. If you meant something behind "
                    "the popup, close the popup first."
                )
                logger.info("🔎 %s", msg)
                return _nothing_clicked(msg, click_first)
            msg = (
                f"find_by_text('{query}'): 0 matches — nothing carrying this text is "
                f"clickable on the CURRENT page ({page_url}). This is NOT proof the text "
                "is absent from the screen. Do NOT re-issue this string through this or "
                "ANY other tool. Decide which of two causes it is before you act:\n"
                "  1. THE STRING. Re-read your step and retry with the word IT uses. If "
                "what you want is a field with no name of its own (a dropdown/combobox, "
                "an unlabelled input), stop searching for it — address it by the label "
                "beside it, with select_dropdown(near_text='<that label>', ...).\n"
                "  2. THE PAGE. The section may not be open yet, or a click may have "
                "navigated you away. Re-read your step for the action that OPENS this "
                "section and do that first; recover a lost page with go_back.\n"
                "Otherwise scroll (0.5 pages at most) or apply the ELEMENT NOT FOUND "
                "POLICY."
            )
            logger.info("🔎 %s", msg)
            # no_click: this touched nothing — without the stamp, compile treats a
            # metadata-less click_first result as a dropped-metadata click and emits a
            # semantic find_click (observed live: a closed-panel check committed a
            # find_click('save') that then failed every replay on the healthy page).
            return _nothing_clicked(msg, click_first)

        # One control wrapped in same-text divs is not an ambiguity — collapse it first,
        # so the exact-label preference and the single-match click below see the real
        # candidate count (see _collapse_nested_duplicates).
        if click_first and len(matches) > 1:
            matches = _collapse_nested_duplicates(matches)

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
            # Click refusals ride the error channel: multi_act stops the remaining
            # queued actions of the step, so nothing executes on top of a click that
            # never happened (the content-channel refusal let a queued Send fire after
            # a refused fill in run 20260807_095537 seg7 — same hazard here).
            if tag == "select":
                msg = (f"find_by_text('{query}'): found the single match "
                       f"{_line(idx, node, label)} but did NOT click it — it is a native "
                       f"<select>, which cannot be clicked. Use dropdown_options(index={idx}) "
                       f"to list its options, then select_dropdown(index={idx}, "
                       f"text='<option>') to pick one.")
                logger.info("🔎 %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
            if tag == "input" and str(attrs_map.get("type") or "").lower() == "file":
                msg = (f"find_by_text('{query}'): found the single match "
                       f"{_line(idx, node, label)} but did NOT click it — it is a file "
                       f"input; use upload_file on index {idx} instead.")
                logger.info("🔎 %s", msg)
                return ActionResult(error=msg, metadata={"no_click": True})
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
                    return ActionResult(error=msg, metadata={"no_click": True})
            # Same repeat-budget guard the plain click override applies, and for the same
            # reason: this is the OTHER doorway onto a control (the prompt sends the agent
            # here whenever a target looks ambiguous), so leaving it unguarded would let a
            # redo walk straight past the refusal.
            refusal = _refuse_if_over_budget(node, label)
            if refusal is not None:
                return refusal
            pre = await _dialog_state(browser_session, node)
            t0 = _stamp_action()
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
                return ActionResult(error=msg, metadata={"no_click": True})
            # This click landed, so it counts toward the segment's repeat ledger exactly as a
            # plain click does — otherwise the budget could be spent through this path
            # without ever being charged.
            _note_clicks(node, 1)
            # Record WHAT we clicked so script_compile can turn this custom action into a real
            # click step (a custom action carries no index, so browser-use captures no
            # interacted_element for it). Same DOMInteractedElement shape a built-in click records.
            captured = _captured_element(node, label)
            meta = {"interacted_element": captured} if captured else None
            # Same network+dialog receipts as the click override: a Save clicked through
            # find_by_text fired its POST invisibly (run 20260807_095537 seg6) and the
            # agent, seeing nothing, clicked Save again — a duplicate create.
            suffix, outcome = await _click_outcome_suffix(browser_session, t0, pre, node)
            meta = _with_write_outcome(meta, outcome)
            msg = (f"find_by_text('{query}'): clicked the single match "
                   f"{_line(idx, node, label)}" + suffix)
            logger.info("🔎 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True, metadata=meta)

        shown = matches[:25]
        lines = [_line(idx, node, label) for idx, node, label in shown]
        tail = "" if len(matches) <= 25 else f"\n...and {len(matches) - 25} more — narrow your text."
        ambiguous = click_first and len(matches) > 1
        guidance = (
            f"Ambiguous: {len(matches)} candidates, so NOTHING WAS CLICKED. Do NOT call "
            f"find_by_text('{query}') again unchanged. Either pick the right index from "
            f"the list above (each line names the ROW it sits in) and click(index) NOW, "
            f"or re-call it scoped: find_by_text('{query}', near_text='<text in the row "
            f"you want>', click_first=True), which clicks that row's one directly."
            if ambiguous
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
        logger.info("🔎 find_by_text('%s'): %d match(es)%s", query, len(matches),
                    " — ambiguous, nothing clicked" if ambiguous else "")
        # no_click: a candidate LISTING — the agent clicks by index next; compiling this
        # result as a find_click would bake a phantom duplicate click into the recording.
        if ambiguous:
            # ERROR channel: a click_first that clicked NOTHING is a refusal, and
            # multi_act stops the step's remaining queued actions on an error. On the
            # success channel it did not: run 20260824_155123 seg 2 batched
            # find_by_text('Get OTP', click_first) with a follow-up click(index) picked
            # from the PREVIOUS snapshot; the refusal left the page unchanged, the stale
            # click fired anyway onto the panel's nameless close button, and the Payroll
            # Review panel was lost — twice, in the same run.
            return ActionResult(error=content, metadata={"no_click": True})
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
            # The VALUE keeps line structure (labeled fields auto-split downstream and
            # bindings slice line parts); the flat name stays the fallback and the
            # element ax_name — every matcher assumes whitespace-flattened names.
            lines = [str(ln) for ln in (raw.get("lines") or []) if str(ln).strip()]
            raw_value = ("\n".join(lines) if len(lines) >= 2
                         else " ".join(str(raw.get("name") or "").split()))[:1000]
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
            # Line-preserving normalization: spaces collapse WITHIN lines, blank lines
            # drop, breaks survive — the structure field auto-split and line bindings
            # rely on.
            text_value = normalize_block_text(node.get_all_children_text(max_depth=5))
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
        "Ground truth for saves: did YOUR LAST ACTION write to the server? Call this right "
        "after clicking the final Save. CONFIRMED means the server accepted the write — "
        "proceed. REFUSED means it fired and the server rejected it — fix what the message "
        "says, then save again. UNCONFIRMED means no write was seen, which is NOT proof of "
        "failure (some saves in this app commit without network traffic): look for the record "
        "on the page, and do not re-enter the data just because this tool did not say "
        "CONFIRMED. Read-only — changes nothing."
    )
    async def verify_save_registered() -> ActionResult:
        # Windowed to the LAST ACTION, from the live collector the click receipts already
        # use. The previous implementation asked a marker-gated probe (_SAVE_PROBE) for the
        # segment's FIRST create-write, which (a) was never installed for a task that
        # declares no marker — it answered "not available for this run; verify via the UI
        # instead", which is how the agent ended up eyeballing search_page and re-entering
        # a payment it had already saved (run 20260824_165824 seg 4) — and (b) saturated
        # once any save landed, so save #1 vouched for save #3.
        #
        # There is no fallback branch because there is no reachable state for one:
        # runner.py installs the live collector under `if network_collector is not None`
        # and the marker probe under the strictly stronger `if success_marker and
        # network_collector is not None`, tearing both down together — so a live-collector
        # miss means no collector at all, and nothing to verify against.
        msg = (await _verify_last_action_write() if _LIVE_NETWORK is not None
               else "Save verification is not available for this run; verify via the UI "
                    "instead.")
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
        "heading or row text, read the decoded names, then click the matching index. "
        "Read-only, and it ENDS the turn: read the indices, then click on your next turn.",
        # Whatever is queued behind this in the same turn was chosen WITHOUT the indices it
        # returns. Run 20260903_100115_102078 issued list_actions and click(index=1) in one
        # turn twice over: `1` meant "the first thing you are about to list", while the real
        # indices on that page were 1180/1290/7413, and browser-use answers a missing index
        # with a plain extracted_content (tools/service.py) — no error, so the rest of the
        # batch sailed on. multi_act breaks on this flag the same way it does for
        # navigate/search/switch/evaluate: the agent gets the listing, then chooses.
        terminates_sequence=True,
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
        if near_tokens and anchor is None:
            # The landmark is NOT on the page. Falling through here listed the page's first
            # 30 named controls under the heading "Clickable controls near '<landmark>'",
            # because the window filter below is skipped when there is no anchor to window
            # around — an answer to a question the tool never managed to ask.
            #
            # Run 20260903_100115_102078 subtask 16 is the cost. The bulk-upload grid had
            # never been opened, so "Select Employee", then "Anas Burns", then "Employees"
            # all missed; all three returned 30 confident-looking controls. The agent
            # concluded the grid was in front of it, skipped the sentence that opens it
            # ("select Bulk upload FPS"), and spent the segment hunting checkboxes that did
            # not exist — clicking a nameless button into an import modal and cancelling it.
            #
            # SUCCESS channel, by the intent rule in _text_miss: this is a probe, and "what
            # is near this?" is a question whose answer can honestly be none. Stopping a
            # batch is terminates_sequence's job, not this branch's.
            msg = (f"list_actions: '{near_text}' is not on the page, so there is nothing to "
                   f"list near it — no controls are reported. This usually means the "
                   f"section holding it is not open yet: do the action your step names to "
                   f"open it, then call list_actions again.")
            logger.info("🧭 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg,
                                include_in_memory=True)

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
