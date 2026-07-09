"""Shared collector interface.

Every telemetry collector is an independent, toggleable add-on that listens to a single
run's Playwright `BrowserContext` and writes its output into that run's artifacts
directory. The Runner connects Playwright to the agent's browser over CDP and hands each
collector the context, so collectors use Playwright's native events (``page.on("console")``,
``page.on("response")``, ...) rather than raw CDP. Keeping the contract here lets the Runner
treat all collectors uniformly:

    collector = SomeCollector(context, artifacts_dir)
    await collector.start()      # attach Playwright listeners to current + future pages
    ... agent runs ...
    await collector.stop()       # stop listening
    path = collector.write()     # persist results() to <artifacts>/<name>.json
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page

logger = logging.getLogger("framework.collector")


class Collector(ABC):
    """Base class for all telemetry collectors.

    Subclasses attach their listeners in ``_attach_page`` (called for every page already
    open when the run starts and every page opened later). The base class handles wiring
    that up across the context and persisting results.
    """

    #: short, filename-safe identifier; also the default output filename stem.
    name: str = "collector"

    def __init__(self, context: BrowserContext, artifacts_dir: Path) -> None:
        self.context = context
        self.artifacts_dir = Path(artifacts_dir)
        self._active = False
        # Agent step the run is currently on; the Runner bumps this at each step start so
        # captured events can be attributed to a step (0 == before the first step / setup).
        self.current_step = 0

    async def start(self) -> None:
        """Attach listeners to every current page and to any page opened later."""
        self._active = True
        for page in self.context.pages:
            self._safe_attach(page)
        self.context.on("page", self._safe_attach)

    async def stop(self) -> None:
        """Stop recording. Listeners stay registered but are gated by ``_active``."""
        self._active = False

    def _safe_attach(self, page: Page) -> None:
        try:
            self._attach_page(page)
        except Exception as exc:  # noqa: BLE001 - a bad page must not break the run
            logger.exception("%s: failed to attach to page: %s", self.name, exc)

    @abstractmethod
    def _attach_page(self, page: Page) -> None:
        """Register this collector's event handlers on a single page."""

    @abstractmethod
    def results(self) -> Any:
        """Return the collected data as a JSON-serializable structure."""

    def write(self) -> Path | None:
        """Persist results() to <artifacts_dir>/<name>.json. Returns the path written."""
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.artifacts_dir / f"{self.name}.json"
            out_path.write_text(json.dumps(self.results(), indent=2, default=str))
            return out_path
        except Exception as exc:  # noqa: BLE001 - persistence must not crash the run
            logger.exception("collector %s failed to write results: %s", self.name, exc)
            return None
