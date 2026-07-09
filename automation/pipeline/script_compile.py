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
# independently of the volatile instance counter.
_REACT_SELECT_PART = re.compile(r"^react-select-\d+-(?P<part>option-\d+|listbox|placeholder)$")
_RS_OPTION = re.compile(r"^(?P<instance>react-select-\d+)-option-\d+$")
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
    # 7. Positional xpath — last resort.
    xpath = element.get("x_path")
    if xpath:
        cands.append("xpath=/" + xpath.lstrip("/"))

    seen: set[str] = set()
    return [c for c in cands if not (c in seen or seen.add(c))]


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


def _dropdown_option_steps(
    element: dict[str, Any], steps: list[dict[str, Any]]
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
    recording_path: str | Path, max_steps: int | None = None
) -> list[dict[str, Any]]:
    """Turn a saved agent history JSON into an ordered list of {action, ...} steps.

    `max_steps` keeps only the first N agent steps. Used to cut a recording at the step where
    the create-write fired (see ground_truth["write_step"]), so an agent that flailed AFTER
    the record was actually saved never gets its post-save junk into the script.
    """
    data = json.loads(Path(recording_path).read_text())
    steps: list[dict[str, Any]] = []
    history = data.get("history", [])
    if max_steps is not None and max_steps > 0:
        history = history[:max_steps]
    # Assert a known starting page: the agent's flow began on this URL, so replay must too.
    # Without it, replay silently depends on wherever the browser happened to be left.
    start_url = ((history[0].get("state") or {}).get("url") if history else None)
    if start_url and start_url.startswith("http"):
        steps.append({"action": "goto", "url": start_url})
    for item in history:
        actions = (item.get("model_output") or {}).get("action") or []
        elements = (item.get("state") or {}).get("interacted_element") or []
        for i, action in enumerate(actions):
            if not action:
                continue
            name = next(iter(action))
            params = action[name] or {}
            element = elements[i] if i < len(elements) else None
            if name == "navigate" and params.get("url"):
                _push_step(steps, {"action": "goto", "url": params["url"]})
            elif name == "click" and element:
                synth = _dropdown_option_steps(element, steps)
                if synth is not None:
                    for s in synth:
                        _push_step(steps, s)
                    continue
                sels = _selectors(element)
                if sels:
                    _push_step(steps, {"action": "click", "selectors": sels})
            elif name == "input" and element:
                sels = _selectors(element)
                if sels:
                    step = {"action": "fill", "selectors": sels,
                            "value": params.get("text", ""), "clear": params.get("clear", True)}
                    # Typing into a react-select filter input: stamp the select instance so a
                    # following option click can be resolved BY the typed label (and so
                    # parameterize() groups retried fills of this field together).
                    m = _RS_INPUT.match((element.get("attributes") or {}).get("id") or "")
                    if m:
                        step["field_id"] = m.group("instance")
                    _push_step(steps, step)
            elif name == "send_keys" and params.get("keys"):
                _push_step(steps, {"action": "press", "keys": params["keys"]})
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
    recording_path: str | Path, steps_path: str | Path, max_steps: int | None = None
) -> list[dict[str, Any]]:
    """Compile `recording_path` and write the step list to `steps_path` atomically."""
    steps = compile_recording(recording_path, max_steps=max_steps)
    _atomic_write(Path(steps_path), json.dumps(steps, indent=2))
    return steps


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


async def _resolve(page: Page, step: dict[str, Any], timeout_ms: int,
                   require_editable: bool = False):
    """Return a locator for the first candidate that resolves to EXACTLY ONE visible element.

    Refusing to act on an ambiguous match is what prevents "clicks somewhere else": rather than
    silently taking `.first`, we require a unique hit. Non-final candidates get a short probe
    budget; the last candidate gets the full timeout and, only then, a logged `.first` concession.
    With `require_editable` (fill steps), a candidate that resolves to a non-editable node is
    skipped so replay falls through to a candidate that hits the real input.
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
        except Exception:  # noqa: BLE001 - try the next candidate
            errors.append(f"{sel} -> not visible")
            continue
        count = await loc.count()
        if count == 1:
            candidate = loc.first
        elif last:
            # Exhausted durable candidates; act on the first visible match but record it.
            logger.warning("ambiguous selector %r matched %d nodes; using .first", sel, count)
            candidate = loc.first
        else:
            errors.append(f"{sel} -> {count} matches (ambiguous)")
            continue
        if require_editable:
            try:
                if not await candidate.is_editable():
                    inner = candidate.locator(_EDITABLE_SEL)
                    if await inner.count() == 0:
                        errors.append(f"{sel} -> visible but not an editable field")
                        continue
                    # The recorded element was the field's container; type into its input.
                    return inner.first, f"{sel} >> {_EDITABLE_SEL}"
            except Exception:  # noqa: BLE001 - element vanished mid-check; try the next
                errors.append(f"{sel} -> vanished during editability check")
                continue
        return candidate, sel
    raise RuntimeError("no unique candidate matched: " + " | ".join(errors))


async def run_steps(page: Page, steps: list[dict[str, Any]], timeout_ms: int = 15000) -> dict[str, Any]:
    """Execute compiled steps over a Playwright page. Returns {executed, failed_at, error, log}.

    `log` records which concrete selector resolved for each interaction, so a healer/validator can
    see exactly how each step was located (and which candidate won).
    """
    executed = 0
    log: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        try:
            action = step["action"]
            if action == "goto":
                await page.goto(step["url"], wait_until="domcontentloaded", timeout=timeout_ms)
            elif action == "click":
                loc, sel = await _resolve(page, step, timeout_ms)
                try:
                    await loc.click(timeout=5000)
                except Exception:  # noqa: BLE001 - typically "another element intercepts pointer events"
                    # react-select renders a placeholder div UNDER an input container that
                    # intercepts pointer events; a forced click dispatches at the element's
                    # position — i.e. onto the overlaying control, which is the real target.
                    logger.warning("click on %r intercepted/failed; retrying with force", sel)
                    await loc.click(timeout=timeout_ms, force=True)
                log.append({"step": idx, "action": action, "used": sel})
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "fill":
                loc, sel = await _resolve(page, step, timeout_ms, require_editable=True)
                if step.get("clear", True):
                    await loc.fill("", timeout=timeout_ms)
                await loc.fill(step.get("value", ""), timeout=timeout_ms)
                log.append({"step": idx, "action": action, "used": sel})
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "press":
                await page.keyboard.press(step["keys"])
            elif action == "type":
                # Types into the FOCUSED element — used for an open react-select menu, whose
                # filter input owns focus. If focus is elsewhere the subsequent click-by-label
                # times out and validation fails safely (no bad script is ever committed).
                await page.keyboard.type(step["text"], delay=30)
                await page.wait_for_timeout(_SETTLE_MS)
            elif action == "wait":
                await page.wait_for_timeout(int(step.get("seconds", 0) * 1000))
            executed += 1
        except Exception as exc:  # noqa: BLE001 - report where the script broke (app changed?)
            logger.exception("script execution broke at step %d: %s", idx, exc)
            return {"executed": executed, "failed_at": idx,
                    "error": f"{type(exc).__name__}: {exc}", "log": log}
    return {"executed": executed, "failed_at": None, "error": None, "log": log}
