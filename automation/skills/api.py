"""SkillApi — the ONLY surface generated skill code may touch.

The division of labor that makes code-as-cache safe: generated code owns CONTROL FLOW
(ordering, arguments, later: branches), while this runtime owns ELEMENT RESOLUTION and
recovery. Every verb resolves its target through the skill's ANCHOR BUNDLE (semantic
handle -> ranked selectors + self-healing fingerprint, stored in <sid>.anchors.json next
to the code) using the exact machinery replay already trusts: ranked-candidate resolution,
fingerprint healing, transient-detach retries, forced-click interception recovery, flyout
reopen, and the post-action settle. DOM drift therefore heals in the ANCHORS — the code
text is never edited for a moved button.

Every call lands in `log` (run_steps-shaped, plus the anchor handle), so a passed replay
can promote healed selectors back into the anchor bundle (see base.promote_healed_anchors)
and a failure reports exactly which call broke.

`interrupt_handlers` is the registered-reflex seam (dismiss a stray modal, close a toast)
run once when a verb's resolution fails, before the failure is final. Empty by default —
handlers are added deliberately, never invented per skill.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from playwright.async_api import Page

from automation.pipeline.script_compile import (_REOPEN_MS, _SETTLE_MS, _click_with_retry,
                                                _click_and_follow, _esc, _extract_value,
                                                _fill_with_retry,
                                                _find_click, _paste_with_retry,
                                                _resolve_with_scroll,
                                                _select_with_retry, _upload_with_retry,
                                                _wheel_scroll, merge_extract)

logger = logging.getLogger("framework.skills.api")

# repeat_click's readiness poll: after each click (and the recorded wait floor), the
# target must resolve visible+enabled again before the next click — a slow employee
# load must delay the cadence, never eat a click. Cap keeps a dead page an honest
# failure instead of a hang.
_REPEAT_READY_CAP_S = 10.0
_REPEAT_POLL_MS = 250

# App-wide recovery reflexes, run once when a verb fails: async (page) -> bool (True =
# something was dismissed/fixed, retry the verb). Registered explicitly by the framework,
# never baked into individual skills.
InterruptHandler = Callable[[Page], Awaitable[bool]]
interrupt_handlers: list[InterruptHandler] = []

# The footprint of an OPEN custom dropdown menu (react-select listbox / option ids, ARIA
# listboxes). Used by select_option to decide whether typing-to-filter has anywhere to go.
_OPEN_MENU_CSS = '[id$="-listbox"], [role="listbox"], [id*="-option-"]'


class SkillApi:
    """One skill execution's runtime: anchor resolution + recovery + the call ledger."""

    def __init__(self, page: Page, anchors: dict[str, dict[str, Any]], *,
                 timeout_ms: int = 15000) -> None:
        self.page = page
        self.anchors = anchors or {}
        self.timeout_ms = timeout_ms
        self.log: list[dict[str, Any]] = []
        self.executed = 0                      # completed api calls (the ledger position)
        self.extracted: dict[str, str] = {}    # extract verb's output ({label: text})
        self._last_click: dict[str, Any] | None = None   # for flyout-reopen recovery
        # (fill step, ledger index just after it) — the commit rung of click()
        self._last_fill: tuple[dict[str, Any], int] | None = None

    # ------------------------------- internals -------------------------------

    def _step_for(self, handle: str, expect: bool = False) -> dict[str, Any]:
        """The anchor bundle as a script_compile-shaped step (selectors + fingerprint +
        the hidden-dispatch permission when the recorded control was invisible).

        `expect` (clicks only) carries the anchor's `expect_text` — the substituted
        param value naming the target — into the step so _resolve verifies every
        acted-on candidate against it. Fills/selects/extracts leave it behind: their
        value is what gets typed/picked, not the target's name."""
        self._acting()
        anchor = self.anchors.get(handle)
        if not anchor:
            raise KeyError(f"unknown anchor handle {handle!r} "
                           f"(known: {sorted(self.anchors)})")
        step: dict[str, Any] = {"selectors": list(anchor.get("selectors") or []),
                                "fingerprint": anchor.get("fingerprint")}
        if anchor.get("hidden_ok"):
            step["hidden_ok"] = True
        if anchor.get("opens_tab"):
            step["opens_tab"] = True
        if anchor.get("query"):
            # Extract anchors keep their recorded query as the semantic re-find fallback.
            step["query"] = anchor["query"]
        if expect and anchor.get("expect_text"):
            step["expect_text"] = anchor["expect_text"]
        return step

    def _acting(self) -> None:
        """This verb is the segment TOUCHING the page: close the network collector's
        page-load window so the app's boot traffic and the segment's own writes stay
        told apart (agent_tools.note_interaction).

        Every handle verb reaches this through _step_for; the handful that resolve without
        an anchor call it themselves. It must run BEFORE the verb dispatches — a save
        replayed straight after a navigation has to be judged, not written off as the
        page's own load traffic."""
        try:
            from automation.pipeline.agent_tools import note_interaction
            note_interaction()
        except Exception as exc:  # noqa: BLE001 - attribution never breaks a replay
            logger.debug("note_interaction unavailable: %s", exc)

    def _record(self, action: str, handle: str | None, used: str,
                healed: dict[str, Any] | None) -> None:
        entry: dict[str, Any] = {"step": self.executed, "action": action, "used": used}
        if handle is not None:
            entry["handle"] = handle
        if healed:
            entry["healed"] = healed
        self.log.append(entry)
        self.executed += 1

    def _done(self, action: str, **extra: Any) -> None:
        """Ledger entry for verbs with no element resolution (press/wait/goto/type)."""
        self.log.append({"step": self.executed, "action": action, "used": "", **extra})
        self.executed += 1

    def _pending_commit(self) -> tuple[dict[str, Any], list[str]] | None:
        """The fill (and the keys pressed after it) whose EFFECT the click about to be
        recovered reads, or None.

        The tier-0 twin (script_compile._preceding_commit) scans a step LIST; a skill is
        Python, so the same question is answered from the call ledger: the last fill, plus
        every `press` since it, provided nothing but waits and presses came in between."""
        if self._last_fill is None:
            return None
        step, at = self._last_fill
        presses: list[str] = []
        for entry in self.log[at:]:
            if entry.get("action") == "press":
                if entry.get("keys"):
                    presses.append(str(entry["keys"]))
            elif entry.get("action") != "wait":
                return None
        return step, presses

    async def _run_interrupts(self) -> bool:
        """Run the registered reflexes once; True if any claims to have cleared the way."""
        cleared = False
        for handler in interrupt_handlers:
            try:
                if await handler(self.page):
                    cleared = True
            except Exception as exc:  # noqa: BLE001 - a reflex must never add a failure
                logger.debug("interrupt handler %s failed: %s", handler, exc)
        return cleared

    async def _menu_open(self) -> bool | None:
        """Is a custom dropdown menu open right now? None when the page cannot be probed
        (select_option then keeps the legacy type-first behavior)."""
        try:
            loc = self.page.locator(_OPEN_MENU_CSS)
            for n in range(min(await loc.count(), 6)):
                if await loc.nth(n).is_visible():
                    return True
            return False
        except Exception:  # noqa: BLE001 - probe-only; never let detection fail the verb
            return None

    # ------------------------------- verbs -------------------------------

    async def click(self, handle: str) -> None:
        """Click the anchored element. Same recovery ladder as replay: ranked selectors ->
        fingerprint heal -> interrupt reflexes -> flyout reopen via the previous click.
        A value-parameterized anchor carries expect_text: the click only lands on an
        element actually NAMED that value (wrong-business-row guard)."""
        step = self._step_for(handle, expect=True)
        if step.get("opens_tab"):
            # A click that spawns a tab is NOT idempotent, so it skips the ladder below
            # entirely — every rung there re-clicks, and a second click is a second tab
            # (run 20260825_105115). _click_and_follow fires it once, decides whether it
            # landed by looking for the tab, and REBINDS the page so the rest of the skill
            # runs where the recording ran.
            sel, healed, self.page = await _click_and_follow(
                self.page, step, self.timeout_ms)
            self._last_click = step
            self._record("click", handle, sel, healed)
            await self.page.wait_for_timeout(_SETTLE_MS)
            return
        try:
            sel, healed = await _click_with_retry(self.page, step, self.timeout_ms)
        except Exception as exc:  # noqa: BLE001 - recovery ladder before the failure is final
            if await self._run_interrupts():
                sel, healed = await _click_with_retry(self.page, step, _REOPEN_MS)
            else:
                sel, healed = await self._recover_click(handle, step, exc)
        self._last_click = step
        self._record("click", handle, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def _recover_click(self, handle: str, step: dict[str, Any],
                             exc: Exception) -> tuple[str, dict[str, Any] | None]:
        """Re-drive whatever PRODUCED the target, then retry it once. run_steps parity —
        see script_compile._click_with_flyout_recovery for why each rung exists.

        COMMIT rung first: when the target is a row in a list a fill filtered, re-clicking
        a menu opener cannot bring it back, and the fill is the only step that can (run
        20260828_144426 subtask 0). FLYOUT rung second: a submenu item exists only while
        its parent flyout is open, and any re-render closes it. If neither helps, the
        ORIGINAL failure is what gets raised."""
        commit = None if step.get("opens_tab") else self._pending_commit()
        if commit is not None:
            fill_step, presses = commit
            logger.info("click %r unreachable (%s); re-issuing the fill that produced "
                        "its list (%r), then retrying once",
                        handle, str(exc)[:120], str(fill_step.get("value"))[:40])
            try:
                await _fill_with_retry(self.page, fill_step, _REOPEN_MS)
                for keys in presses:
                    await self.page.keyboard.press(keys)
                await self.page.wait_for_timeout(_SETTLE_MS)
                return await _click_with_retry(self.page, step, _REOPEN_MS)
            except Exception as inner:  # noqa: BLE001 - fall through to the flyout rung
                logger.info("commit re-issue did not bring %r back (%s)",
                            handle, str(inner)[:120])
        if self._last_click is not None:
            logger.info("click %r unreachable (%s); re-clicking predecessor to reopen "
                        "its flyout, then retrying once", handle, str(exc)[:120])
            try:
                await _click_with_retry(self.page, self._last_click, _REOPEN_MS)
                await self.page.wait_for_timeout(_SETTLE_MS)
                return await _click_with_retry(self.page, step, _REOPEN_MS)
            except Exception:  # noqa: BLE001 - surface the ORIGINAL failure
                raise exc from None
        raise exc

    async def _await_repeat_ready(self, step: dict[str, Any], handle: str) -> None:
        """Poll until one of the step's selectors resolves visible+enabled again."""
        import time as _time

        deadline = _time.monotonic() + _REPEAT_READY_CAP_S
        sels = (step.get("selectors") or [])[:3]
        while _time.monotonic() < deadline:
            for sel in sels:
                try:
                    loc = self.page.locator(sel)
                    if (await loc.count()) and await loc.first.is_visible() \
                            and await loc.first.is_enabled():
                        return
                except Exception:  # noqa: BLE001 - candidate unprobeable; try the next
                    continue
            await self.page.wait_for_timeout(_REPEAT_POLL_MS)
        raise RuntimeError(
            f"repeat_click: {handle!r} did not become ready again within "
            f"{_REPEAT_READY_CAP_S:g}s — the page likely stopped advancing")

    async def repeat_click(self, handle: str, count: int, wait_s: float = 0.0) -> None:
        """Click the anchored element `count` times — the compiled form of a recorded
        "exactly N clicks" cadence (Save & Next through N employees). Between clicks:
        the recorded wait as a floor, then a readiness poll on the same target, so a
        slow load delays the cadence instead of eating a click. The first winning
        selector is cached so every iteration follows the same routine; a stale cache
        falls back to the full ladder once."""
        step = self._step_for(handle, expect=True)
        cached: dict[str, Any] | None = None
        for i in range(int(count)):
            try:
                sel, healed = await _click_with_retry(
                    self.page, cached or step, self.timeout_ms)
            except Exception:  # noqa: BLE001 - cached selector went stale; full ladder
                if cached is None:
                    raise
                cached = None
                sel, healed = await _click_with_retry(self.page, step, self.timeout_ms)
            if cached is None and not sel.endswith("(hidden dispatch)"):
                cached = {**step, "selectors": [sel]}
            self._last_click = step
            self._record("click", handle, sel, healed)
            await self.page.wait_for_timeout(_SETTLE_MS)
            if i < int(count) - 1:
                if wait_s:
                    await self.page.wait_for_timeout(int(min(float(wait_s), 3.0) * 1000))
                await self._await_repeat_ready(cached or step, handle)

    async def repeat_until_done(self, handle: str, wait_s: float = 0.0,
                                cap: int = 200) -> int:
        """Click the anchored element until it stops advancing, and return how many landed.

        The counted twin above replays a number; this replays an INTENT. A slice worded
        "click Next for all of the remaining employees" has no number at authoring time, and
        freezing the authoring run's count would silently under-run a longer list — so the
        recording stores the intent and the replay re-discovers the end. The readiness poll
        is the stop signal: the control going away, disabled, or hidden IS the end of the
        list. Reaching `cap` means it was still advancing, which raises rather than passing
        an unfinished run off as complete."""
        step = self._step_for(handle, expect=True)
        cached: dict[str, Any] | None = None
        done = 0
        while done < cap:
            try:
                sel, healed = await _click_with_retry(
                    self.page, cached or step, self.timeout_ms)
            except Exception:  # noqa: BLE001 - cached selector went stale; full ladder once
                if cached is None:
                    raise
                cached = None
                sel, healed = await _click_with_retry(self.page, step, self.timeout_ms)
            if cached is None and not sel.endswith("(hidden dispatch)"):
                cached = {**step, "selectors": [sel]}
            self._last_click = step
            self._record("click", handle, sel, healed)
            done += 1
            await self.page.wait_for_timeout(_SETTLE_MS)
            if wait_s:
                await self.page.wait_for_timeout(int(min(float(wait_s), 3.0) * 1000))
            try:
                await self._await_repeat_ready(cached or step, handle)
            except RuntimeError:
                # The control stopped coming back — that is the end of the list, and the
                # whole point of this verb, so it is a clean finish rather than a failure.
                logger.info("↻ %s: stopped advancing after %d click(s)", handle, done)
                return done
        raise RuntimeError(
            f"repeat_until_done: {handle!r} was still advancing after {cap} clicks — "
            f"refusing to report an unfinished run as complete")

    async def click_indexed(self, handle: str, start: int, count: int) -> None:
        """Click id-indexed grid rows start..start+count-1 via the anchor's selector
        template (`{n}` placeholder) — the durable form of a virtualized-grid run
        whose row-id prefixes regenerate per data load. Each index wheel-scrolls into
        view when the pane hasn't rendered it yet. Positional by intent: the Nth row
        is clicked whoever occupies it, so no name guard applies."""
        self._acting()
        anchor = self.anchors.get(handle) or {}
        template = str(anchor.get("selector_template") or "")
        if "{n}" not in template:
            raise KeyError(f"anchor {handle!r} carries no indexed selector template")
        for n in range(int(start), int(start) + int(count)):
            step = {"selectors": [template.replace("{n}", str(n))]}
            loc, sel = await _resolve_with_scroll(self.page, step, 2500)
            try:
                await loc.click(timeout=5000)
            except Exception:  # noqa: BLE001 - pointer interception; force like click does
                await loc.click(timeout=5000, force=True)
            self._record("click", handle, sel, None)
            await self.page.wait_for_timeout(_SETTLE_MS)

    async def fill(self, handle: str, value: Any, clear: bool = True) -> None:
        """Fill the anchored editable field (resolution refuses non-editable matches)."""
        step = self._step_for(handle)
        step.update({"value": str(value), "clear": bool(clear)})
        try:
            sel, healed = await _fill_with_retry(self.page, step, self.timeout_ms)
        except Exception:  # noqa: BLE001
            if not await self._run_interrupts():
                raise
            sel, healed = await _fill_with_retry(self.page, step, self.timeout_ms)
        self._record("fill", handle, sel, healed)
        self._last_fill = (step, self.executed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def select(self, handle: str, label: Any) -> None:
        """Pick an option on the anchored NATIVE <select> by visible label (option value
        as fallback). select_option fires the change events the page's scripts listen
        for; distinct from select_option below, which drives an already-open custom
        (react-select) menu."""
        step = self._step_for(handle)
        step["value"] = str(label)
        try:
            sel, healed = await _select_with_retry(self.page, step, self.timeout_ms)
        except Exception:  # noqa: BLE001 - one reflex pass before the failure is final
            if not await self._run_interrupts():
                raise
            sel, healed = await _select_with_retry(self.page, step, self.timeout_ms)
        self._record("select", handle, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def select_option(self, label: Any) -> None:
        """Pick an option BY LABEL from the currently open dropdown menu: type the label
        into the menu's focused filter input, then click the matching option (first
        filtered option as last resort). This is exactly the pair compile synthesizes for
        react-select picks — the value replays by VALUE, never by position.

        Every candidate is scoped to an OPEN menu (listbox container, role=option, or a
        react-select option id): a bare page-wide text match must never count, because a
        row cell elsewhere carrying the same word (a status chip reading "Submitted")
        would be picked when the menu failed to open (observed live). For the same
        reason the type-to-filter is skipped when no menu is detectably open — a blind
        type lands in whatever currently has focus. If the menu is closed, the opener
        recorded by the previous click re-opens it, both before the pick and once more
        as the final recovery ladder."""
        self._acting()
        text = str(label)
        # expect_text: the first-filtered-option fallback must still be NAMED the label —
        # if the filter never applied, option-0 is an arbitrary option, not the value.
        option = {"selectors": [f'css=[id$="-listbox"] >> text="{_esc(text)}"',
                                f'role=option[name="{_esc(text)}"]',
                                f'css=[class*="menu"] >> text="{_esc(text)}"',
                                'css=[id$="-option-0"]'],
                  "expect_text": text}

        async def _type_filter() -> None:
            if await self._menu_open() is False:
                logger.info("select_option %r: no open menu; skipping the "
                            "type-to-filter (a blind type would land in whatever "
                            "has focus)", text)
                return
            await self.page.keyboard.type(text, delay=30)
            await self.page.wait_for_timeout(_SETTLE_MS)

        if await self._menu_open() is False and self._last_click is not None:
            logger.info("select_option %r: menu not open yet; re-clicking the opener "
                        "first", text)
            try:
                await _click_with_retry(self.page, self._last_click, _REOPEN_MS)
                await self.page.wait_for_timeout(_SETTLE_MS)
            except Exception as exc:  # noqa: BLE001 - the pick below reports the real failure
                logger.info("select_option %r: opener pre-click failed (%s)",
                            text, str(exc)[:120])
        await _type_filter()
        try:
            sel, healed = await _click_with_retry(self.page, option, self.timeout_ms)
        except Exception as exc:  # noqa: BLE001 - reopen the menu before the failure is final
            if self._last_click is None:
                raise
            logger.info("select_option %r found no pickable option (%s); re-clicking "
                        "the opener and retrying once", text, str(exc)[:120])
            try:
                await _click_with_retry(self.page, self._last_click, _REOPEN_MS)
                await self.page.wait_for_timeout(_SETTLE_MS)
                await _type_filter()
                sel, healed = await _click_with_retry(self.page, option, _REOPEN_MS)
            except Exception:  # noqa: BLE001 - surface the ORIGINAL failure
                raise exc from None
        self._record("select_option", None, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def type_text(self, text: Any) -> None:
        """Type into the FOCUSED element (an open dropdown's filter owns focus)."""
        self._acting()
        await self.page.keyboard.type(str(text), delay=30)
        self._done("type")
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def press(self, keys: str) -> None:
        self._acting()
        await self.page.keyboard.press(keys)
        # keys ride the ledger so _pending_commit can replay the commit
        self._done("press", keys=keys)

    async def wait(self, seconds: float) -> None:
        """Deliberate settle, capped like compiled waits — load-bearing on this slow app."""
        await self.page.wait_for_timeout(int(min(float(seconds), 3.0) * 1000))
        self._done("wait")

    async def scroll(self, pages: float = 0.5, down: bool = True) -> None:
        """Discovery scroll (viewport fractions) as REAL wheel input — window.scrollBy is
        a no-op inside Fluent ScrollablePanes; wheel scrolls whatever a user would."""
        await _wheel_scroll(self.page, float(pages), down=bool(down))
        self._done("scroll")

    async def find_click(self, label: Any) -> None:
        """Semantic click by label: the exact find_by_text algorithm the authoring tool
        used (token match, visible-first, scrollIntoView, direct handler click, scrolling
        between rounds). THE verb for hover-revealed/0-size controls, where selector +
        pointer replay is structurally unstable."""
        self._acting()
        name = await _find_click(self.page, str(label))
        self._record("find_click", None, f"find_click:{name}", None)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def upload(self, handle: str, name: Any) -> None:
        """Attach files.UPLOADS_DIR/<name> to the anchored upload control via
        set_input_files — the native dialog never opens, and a missing/empty file
        raises before anything touches the page (a ghost upload crashes the tab at
        save time — see _upload_with_retry)."""
        step = self._step_for(handle)
        step["value"] = str(name)
        try:
            sel, healed = await _upload_with_retry(self.page, step, self.timeout_ms)
        except Exception:  # noqa: BLE001 - one reflex pass before the failure is final
            if not await self._run_interrupts():
                raise
            sel, healed = await _upload_with_retry(self.page, step, self.timeout_ms)
        self._record("upload", handle, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def extract(self, handle: str, label: str) -> str:
        """Read the anchored element's CURRENT text into the `extracted` ledger — the
        fresh data an aux-tab skill exists to fetch. Same ladder as the tier-0 extract
        step (_extract_value): ranked selectors -> fingerprint heal -> semantic re-find
        by the recorded query; an EMPTY read raises so the segment fails honestly instead
        of replaying a hollow pass."""
        step = self._step_for(handle)
        step["label"] = label
        try:
            value, used, healed = await _extract_value(self.page, step, self.timeout_ms)
        except Exception:  # noqa: BLE001 - one reflex pass before the failure is final
            if not await self._run_interrupts():
                raise
            value, used, healed = await _extract_value(self.page, step, self.timeout_ms)
        # Collision-safe: a second extract sharing this label keeps BOTH values
        # (label_2, ...) instead of clobbering — see script_compile.merge_extract.
        merge_extract(self.extracted, str(label), value)
        self._record("extract", handle, used, healed)
        self.log[-1]["value"] = value[:200]
        return value

    async def paste(self, handle: str, value: Any) -> None:
        """Deliver a WHOLE value to the anchored field through the paste ladder (synthetic
        paste event -> Chrome's paste command -> keystrokes; see _paste_into). The verb for
        a value the page splits across SEVERAL inputs: an ordinary fill would put the first
        character in the first box and the rest nowhere. Raises if the value did not land."""
        step = self._step_for(handle)
        step["value"] = str(value)
        try:
            sel, healed = await _paste_with_retry(self.page, step, self.timeout_ms)
        except Exception:  # noqa: BLE001 - one reflex pass before the failure is final
            if not await self._run_interrupts():
                raise
            sel, healed = await _paste_with_retry(self.page, step, self.timeout_ms)
        self._record("paste", handle, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    def noted(self, label: str) -> str:
        """The value an EARLIER step of this same skill extracted, read live.

        The compiled form of a {{noted:label}} token (codegen._value_expr): the merged OTP
        slice copies the code and pastes it, so the paste must read what THIS run's copy
        captured. Raises rather than returning a stale or empty value — a skill that types
        the wrong OTP passes its gate (the URL is identical either side of the wall) and
        the whole rest of the task then runs against a page that never opened."""
        value = str((self.extracted or {}).get(label) or "").strip()
        if not value:
            raise KeyError(
                f"noted value {label!r} is empty — the step that captures it either did "
                f"not run or read nothing (have: {sorted(self.extracted)})")
        return value

    async def copy(self, handle: str, label: str) -> str:
        """extract, plus the clipboard — the replay twin of the copy_text tool, so a
        replayed paste that carries no text of its own still has the value to deliver."""
        value = await self.extract(handle, label)
        self.log[-1]["action"] = "copy"
        try:
            await self.page.evaluate("(t) => navigator.clipboard.writeText(t)", value)
        except Exception as exc:  # noqa: BLE001 - clipboard is a convenience, never a gate
            logger.debug("skill clipboard write failed: %s", exc)
        return value

    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self._done("goto")
