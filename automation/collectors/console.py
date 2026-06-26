"""Browser console log capture via Playwright.

Two Playwright page events together cover what a developer sees in the console:
  * ``console`` — every ``console.log/info/warn/error/...`` call from page JS, with its
    rendered text, type (level), and source location.
  * ``pageerror`` — uncaught exceptions thrown on the page (which do not surface as console
    messages but are genuine errors).

Both are normalized into a single list of entries with the agent ``step`` they fired on,
``severity``, a human ``type`` label, ``text``, ``source`` (url), and ``is_error`` /
``is_warning`` / ``is_debug`` flags, then summarized. Output is one JSON file per run.
"""
from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import ConsoleMessage, Error, Page

from automation.collectors.base import Collector

logger = logging.getLogger("framework.collector.console")

# Map a Playwright console message type to (severity, is_error, is_warning, is_debug).
_ERROR_TYPES = {"error", "assert"}
_WARNING_TYPES = {"warning", "warn"}
_DEBUG_TYPES = {"debug", "trace", "verbose"}


def _severity(level: str) -> str:
    if level in _ERROR_TYPES:
        return "error"
    if level in _WARNING_TYPES:
        return "warning"
    if level in _DEBUG_TYPES:
        return "debug"
    return "info"


class ConsoleCollector(Collector):
    """Collects browser console messages + uncaught page errors for one run."""

    name = "console"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._entries: list[dict[str, Any]] = []

    def _attach_page(self, page: Page) -> None:
        page.on("console", self._on_console)
        page.on("pageerror", self._on_page_error)

    # ---------------- Playwright event handlers (sync) ----------------

    def _on_console(self, msg: ConsoleMessage) -> None:
        if not self._active:
            return
        try:
            level = (msg.type or "log").lower()
            severity = _severity(level)
            location = msg.location or {}
            self._entries.append(
                {
                    "step": self.current_step,
                    "channel": "console",
                    "type": "JS Console",
                    "level": level,
                    "severity": severity,
                    "text": msg.text,
                    "source": location.get("url"),
                    "line": location.get("lineNumber"),
                    "is_error": severity == "error",
                    "is_warning": severity == "warning",
                    "is_debug": severity == "debug",
                    "is_exception": False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("console handler error: %s", exc)

    def _on_page_error(self, error: Error) -> None:
        if not self._active:
            return
        try:
            self._entries.append(
                {
                    "step": self.current_step,
                    "channel": "pageerror",
                    "type": "JS Exception",
                    "level": "error",
                    "severity": "error",
                    "text": getattr(error, "message", str(error)),
                    "source": None,
                    "line": None,
                    "stack": getattr(error, "stack", None),
                    "is_error": True,
                    "is_warning": False,
                    "is_debug": False,
                    "is_exception": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pageerror handler error: %s", exc)

    # ---------------- results ----------------

    def results(self) -> dict[str, Any]:
        errors = [e for e in self._entries if e.get("is_error")]
        warnings = [e for e in self._entries if e.get("is_warning")]
        debug = [e for e in self._entries if e.get("is_debug")]
        exceptions = [e for e in self._entries if e.get("is_exception")]
        return {
            "summary": {
                "total": len(self._entries),
                "errors": len(errors),
                "warnings": len(warnings),
                "debug": len(debug),
                "exceptions": len(exceptions),
            },
            "errors": errors,
            "warnings": warnings,
            "entries": self._entries,
        }
