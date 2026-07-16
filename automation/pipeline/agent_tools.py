"""Custom agent tools the prompts rely on, registered on a browser-use ``Tools`` registry.

The system/expander prompts (see prompts.py) instruct the agent to call seven actions that are
NOT part of browser-use's built-in set:

  * skip_step(reason)          — escape hatch: abandon the current objective, keep going.
  * fail_and_stop(reason)      — escape hatch: terminate the whole run as a failure.
  * capped_scroll(down, pages) — discovery scroll capped at 0.5 pages per call.
  * find_by_text(text)         — find interactive elements by label in a FRESH snapshot,
                                 returning their current click indexes (optionally clicking).
  * list_actions(near_text)    — list clickable controls near a heading/row, decoding the
                                 nameless icon buttons (Fluent/SVG) browser-use renders blank.
  * verify_save_registered()   — ground truth for saves: did a create-write actually hit the
                                 server this run? (probe wired by the Runner per run).
  * detect_layout_issues()     — heuristic layout/overflow scan of the current page.
  * run_accessibility_scan()   — WCAG scan via vendored axe-core (assets/axe.min.js).

`build_tools()` returns a `Tools` instance with these registered (plus every built-in EXCEPT
`evaluate` — see below, since `Tools()` starts from the default registry). The Runner passes
it to `Agent(tools=...)`.

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

import json
import logging
import re
from pathlib import Path
from typing import Any

from browser_use import Tools
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.events import ClickElementEvent
from browser_use.dom.views import DOMInteractedElement

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
_RAW_FIND_JS = r"""
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
    return { count: out.length, clicked: clicked, name: top.name,
             names: out.slice(0, 8).map(function (o) { return o.name; }) };
  } catch (e) { return { error: String(e) }; }
})()
"""


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


def build_tools() -> Tools:
    """Return a browser-use `Tools` registry with our custom actions added and `evaluate`
    removed (JS form-fills are unrecordable — see module docstring)."""
    tools = Tools(exclude_actions=["evaluate"])

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
                    msg = (f"find_by_text('{query}'): the target was not in the interactive "
                           f"snapshot (a 0-size/off-screen control) — located it in the DOM and "
                           f"clicked it directly: '{raw.get('name')}'. Verify the page responded "
                           f"as expected before proceeding.")
                    logger.info("🔎 %s", msg)
                    return ActionResult(extracted_content=msg, long_term_memory=msg,
                                        include_in_memory=True)
                names = ", ".join(f"'{n}'" for n in (raw.get("names") or []) if n)
                msg = (f"find_by_text('{query}'): {raw['count']} match(es) exist in the DOM but "
                       f"are NOT clickable via index (0-size/virtualized): {names}. Re-call "
                       f"find_by_text('{query}', click_first=true) to click the best match directly.")
                logger.info("🔎 %s", msg)
                return ActionResult(extracted_content=msg, long_term_memory=msg,
                                    include_in_memory=True)
            msg = (
                f"find_by_text('{query}'): 0 matches on the CURRENT page ({page_url}). "
                "The element is not in this page's DOM. FIRST check: is this the page you think "
                "you are on? If a previous click navigated you away (e.g. back to a list page), "
                "recover with go_back or re-open the right section. Otherwise scroll with "
                "capped_scroll or apply the ELEMENT NOT FOUND POLICY; do not repeat this exact query."
            )
            logger.info("🔎 %s", msg)
            return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

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
            try:
                event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("find_by_text click failed: %s", exc)
                return ActionResult(error=f"find_by_text: found '{label[:80]}' but click failed: {exc}")
            # Record WHAT we clicked so script_compile can turn this custom action into a real
            # click step (a custom action carries no index, so browser-use captures no
            # interacted_element for it). Same DOMInteractedElement shape a built-in click records.
            meta: dict[str, Any] | None = None
            try:
                meta = {"interacted_element":
                        DOMInteractedElement.load_from_enhanced_dom_tree(node).to_dict()}
            except Exception as exc:  # noqa: BLE001 - recording the target is best-effort
                logger.debug("find_by_text: could not capture interacted element: %s", exc)
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
        return ActionResult(extracted_content=content, long_term_memory=memory, include_in_memory=True)

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
                   "fix those exact fields, and click Save again. Do NOT report success until "
                   "this tool returns CONFIRMED.")
        logger.info("🧾 %s", msg.split(".")[0])
        return ActionResult(extracted_content=msg, long_term_memory=msg, include_in_memory=True)

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
