"""Compile a recorded agent run into a lean, deterministic selector script.

browser-use's `rerun_history` replays at the ORIGINAL pace (it re-waits the agent's recorded
delays, including its LLM thinking time). This instead extracts just the essential actions and
a STABLE selector for each interacted element from a saved recording, so the flow can be
re-run fast over Playwright with auto-wait — no LLM, no recorded delays, `wait` steps dropped.

Selector strategy (durability > brevity): prefer a stable attribute the app is unlikely to
re-generate (id that doesn't look auto-numbered, then aria-label / name / title / placeholder),
and fall back to the recorded positional `x_path` only as a last resort.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from playwright.async_api import Page

# ids ending in 3+ digits look auto-generated (e.g. "SearchBox129") — don't anchor on them.
_DYNAMIC_ID = re.compile(r"\d{3,}")


def _esc(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _attr_sel(name: str, value: str) -> str:
    return f'css=[{name}="{_esc(value)}"]'


def _selector(element: dict[str, Any]) -> str | None:
    """Best stable Playwright selector for a recorded element, falling back to its xpath.

    Priority (most → least durable on a re-rendering app): a non-auto-numbered id, then a
    distinguishing attribute, then href (for links), then the element's visible text /
    accessibility name, and only as a last resort the positional xpath.
    """
    attrs = element.get("attributes") or {}
    tag = (element.get("node_name") or "").lower()
    idv = attrs.get("id")
    if idv and not _DYNAMIC_ID.search(idv):
        return _attr_sel("id", idv)
    for key in ("aria-label", "name", "title", "placeholder", "data-testid"):
        if attrs.get(key):
            return _attr_sel(key, attrs[key])
    if attrs.get("href"):
        return f'css={tag or "*"}[href="{_esc(attrs["href"])}"]'
    ax_name = (element.get("ax_name") or "").strip()
    if ax_name:
        return f'text="{_esc(ax_name)}"'  # exact visible-text / accessible-name match
    xpath = element.get("x_path")
    if xpath:
        return "xpath=/" + xpath.lstrip("/")
    return None


def compile_recording(recording_path: str | Path) -> list[dict[str, Any]]:
    """Turn a saved agent history JSON into an ordered list of {action, ...} steps."""
    data = json.loads(Path(recording_path).read_text())
    steps: list[dict[str, Any]] = []
    for item in data.get("history", []):
        actions = (item.get("model_output") or {}).get("action") or []
        elements = (item.get("state") or {}).get("interacted_element") or []
        for i, action in enumerate(actions):
            if not action:
                continue
            name = next(iter(action))
            params = action[name] or {}
            element = elements[i] if i < len(elements) else None
            if name == "navigate" and params.get("url"):
                steps.append({"action": "goto", "url": params["url"]})
            elif name == "click" and element:
                sel = _selector(element)
                if sel:
                    steps.append({"action": "click", "selector": sel})
            elif name == "input" and element:
                sel = _selector(element)
                if sel:
                    steps.append({"action": "fill", "selector": sel,
                                  "value": params.get("text", ""), "clear": params.get("clear", True)})
            elif name == "send_keys" and params.get("keys"):
                steps.append({"action": "press", "keys": params["keys"]})
            # `wait` and `done` are intentionally dropped — Playwright auto-waits on locators.
    return steps


def save_steps(recording_path: str | Path, steps_path: str | Path) -> list[dict[str, Any]]:
    """Compile `recording_path` and write the step list to `steps_path`."""
    steps = compile_recording(recording_path)
    steps_path = Path(steps_path)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.write_text(json.dumps(steps, indent=2))
    return steps


async def run_steps(page: Page, steps: list[dict[str, Any]], timeout_ms: int = 15000) -> dict[str, Any]:
    """Execute compiled steps over a Playwright page. Returns {executed, failed_at, error}."""
    executed = 0
    for idx, step in enumerate(steps):
        try:
            action = step["action"]
            if action == "goto":
                await page.goto(step["url"], wait_until="domcontentloaded", timeout=timeout_ms)
            elif action == "click":
                await page.locator(step["selector"]).first.click(timeout=timeout_ms)
            elif action == "fill":
                loc = page.locator(step["selector"]).first
                if step.get("clear", True):
                    await loc.fill("", timeout=timeout_ms)
                await loc.fill(step.get("value", ""), timeout=timeout_ms)
            elif action == "press":
                await page.keyboard.press(step["keys"])
            executed += 1
        except Exception as exc:  # noqa: BLE001 - report where the script broke (app changed?)
            return {"executed": executed, "failed_at": idx, "error": f"{type(exc).__name__}: {exc}"}
    return {"executed": executed, "failed_at": None, "error": None}
