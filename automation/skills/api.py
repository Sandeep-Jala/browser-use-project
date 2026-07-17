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
                                                _esc, _fill_with_retry, _find_click,
                                                _wheel_scroll)

logger = logging.getLogger("framework.skills.api")

# App-wide recovery reflexes, run once when a verb fails: async (page) -> bool (True =
# something was dismissed/fixed, retry the verb). Registered explicitly by the framework,
# never baked into individual skills.
InterruptHandler = Callable[[Page], Awaitable[bool]]
interrupt_handlers: list[InterruptHandler] = []


class SkillApi:
    """One skill execution's runtime: anchor resolution + recovery + the call ledger."""

    def __init__(self, page: Page, anchors: dict[str, dict[str, Any]], *,
                 timeout_ms: int = 15000) -> None:
        self.page = page
        self.anchors = anchors or {}
        self.timeout_ms = timeout_ms
        self.log: list[dict[str, Any]] = []
        self.executed = 0                      # completed api calls (the ledger position)
        self._last_click: dict[str, Any] | None = None   # for flyout-reopen recovery

    # ------------------------------- internals -------------------------------

    def _step_for(self, handle: str) -> dict[str, Any]:
        """The anchor bundle as a script_compile-shaped step (selectors + fingerprint +
        the hidden-dispatch permission when the recorded control was invisible)."""
        anchor = self.anchors.get(handle)
        if not anchor:
            raise KeyError(f"unknown anchor handle {handle!r} "
                           f"(known: {sorted(self.anchors)})")
        step: dict[str, Any] = {"selectors": list(anchor.get("selectors") or []),
                                "fingerprint": anchor.get("fingerprint")}
        if anchor.get("hidden_ok"):
            step["hidden_ok"] = True
        return step

    def _record(self, action: str, handle: str | None, used: str,
                healed: dict[str, Any] | None) -> None:
        entry: dict[str, Any] = {"step": self.executed, "action": action, "used": used}
        if handle is not None:
            entry["handle"] = handle
        if healed:
            entry["healed"] = healed
        self.log.append(entry)
        self.executed += 1

    def _done(self, action: str) -> None:
        """Ledger entry for verbs with no element resolution (press/wait/goto/type)."""
        self.log.append({"step": self.executed, "action": action, "used": ""})
        self.executed += 1

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

    # ------------------------------- verbs -------------------------------

    async def click(self, handle: str) -> None:
        """Click the anchored element. Same recovery ladder as replay: ranked selectors ->
        fingerprint heal -> interrupt reflexes -> flyout reopen via the previous click."""
        step = self._step_for(handle)
        try:
            sel, healed = await _click_with_retry(self.page, step, self.timeout_ms)
        except Exception as exc:  # noqa: BLE001 - recovery ladder before the failure is final
            if await self._run_interrupts():
                sel, healed = await _click_with_retry(self.page, step, _REOPEN_MS)
            elif self._last_click is not None:
                # Flyout reopen (run_steps parity): the target may live in a menu the app
                # re-render closed; only re-clicking its opener can bring it back.
                logger.info("click %r unreachable (%s); re-clicking predecessor to reopen "
                            "its flyout, then retrying once", handle, str(exc)[:120])
                try:
                    await _click_with_retry(self.page, self._last_click, _REOPEN_MS)
                    await self.page.wait_for_timeout(_SETTLE_MS)
                    sel, healed = await _click_with_retry(self.page, step, _REOPEN_MS)
                except Exception:  # noqa: BLE001 - surface the ORIGINAL failure
                    raise exc from None
            else:
                raise
        self._last_click = step
        self._record("click", handle, sel, healed)
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
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def select_option(self, label: Any) -> None:
        """Pick an option BY LABEL from the currently open dropdown menu: type the label
        into the menu's focused filter input, then click the matching option (first
        filtered option as last resort). This is exactly the pair compile synthesizes for
        react-select picks — the value replays by VALUE, never by position."""
        text = str(label)
        await self.page.keyboard.type(text, delay=30)
        await self.page.wait_for_timeout(_SETTLE_MS)
        option = {"selectors": [f'role=option[name="{_esc(text)}"]',
                                f'text="{_esc(text)}"',
                                'css=[id$="-option-0"]']}
        sel, healed = await _click_with_retry(self.page, option, self.timeout_ms)
        self._record("select_option", None, sel, healed)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def type_text(self, text: Any) -> None:
        """Type into the FOCUSED element (an open dropdown's filter owns focus)."""
        await self.page.keyboard.type(str(text), delay=30)
        self._done("type")
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def press(self, keys: str) -> None:
        await self.page.keyboard.press(keys)
        self._done("press")

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
        name = await _find_click(self.page, str(label))
        self._record("find_click", None, f"find_click:{name}", None)
        await self.page.wait_for_timeout(_SETTLE_MS)

    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self._done("goto")
