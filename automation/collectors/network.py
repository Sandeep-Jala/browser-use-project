"""Network request/response logging via Playwright.

Listens to each page's Playwright network events — ``request`` (a request was issued),
``response`` (headers arrived), and ``requestfailed`` (the request never completed). Each
request is aggregated into one record keyed by the Playwright ``Request`` object, capturing
the agent step it fired on, method, URL, resource type, status (+ a coarse 2xx/3xx/4xx/5xx
class), wall-clock duration, request/response headers, and failure text. Output is one JSON
file per run.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from playwright.async_api import Page, Request, Response

from automation.collectors.base import Collector

logger = logging.getLogger("framework.collector.network")

# Requests slower than this (ms) are flagged as "slow" in the summary.
_SLOW_MS = 1000.0


def _status_class(status: int | None, failed: bool) -> str:
    if failed:
        return "failed"
    if status is None:
        return "pending"
    return f"{status // 100}xx"


class NetworkCollector(Collector):
    """Collects network traffic for one run via Playwright page events."""

    name = "network"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Request object -> aggregated record. Request instances are unique per request
        # and stay alive for the run, so they make a safe correlation key.
        self._records: dict[Request, dict[str, Any]] = {}
        self._started_at: dict[Request, float] = {}

    def _attach_page(self, page: Page) -> None:
        page.on("request", self._on_request)
        page.on("response", self._on_response)
        page.on("requestfinished", self._on_request_finished)
        page.on("requestfailed", self._on_request_failed)

    # ---------------- Playwright event handlers (sync) ----------------

    def _record_for(self, request: Request) -> dict[str, Any]:
        return self._records.setdefault(
            request,
            {
                "step": self.current_step,
                "url": request.url,
                "method": request.method,
                "resourceType": request.resource_type,
                "status": None,
                "statusText": None,
                "duration_ms": None,
                "failed": False,
                "errorText": None,
                "is_error": False,
                "request_headers": {},
                "response_headers": {},
                "status_class": "pending",
            },
        )

    def _finish(self, request: Request, record: dict[str, Any]) -> None:
        start = self._started_at.get(request)
        if start is not None and record.get("duration_ms") is None:
            record["duration_ms"] = round((time.monotonic() - start) * 1000, 2)

    def _on_request(self, request: Request) -> None:
        if not self._active:
            return
        record = self._record_for(request)
        self._started_at.setdefault(request, time.monotonic())
        try:
            record["request_headers"] = dict(request.headers)
        except Exception as exc:  # noqa: BLE001
            logger.exception("request headers error: %s", exc)

    def _on_response(self, response: Response) -> None:
        if not self._active:
            return
        try:
            request = response.request
            record = self._record_for(request)
            status = response.status
            record.update(
                {
                    "status": status,
                    "statusText": response.status_text,
                    "is_error": status >= 400,
                    "status_class": _status_class(status, record["failed"]),
                }
            )
            try:
                record["response_headers"] = dict(response.headers)
            except Exception as exc:  # noqa: BLE001
                logger.exception("response headers error: %s", exc)
            self._finish(request, record)
        except Exception as exc:  # noqa: BLE001
            logger.exception("response handler error: %s", exc)

    def _on_request_finished(self, request: Request) -> None:
        if not self._active:
            return
        self._finish(request, self._record_for(request))

    def _on_request_failed(self, request: Request) -> None:
        if not self._active:
            return
        try:
            record = self._record_for(request)
            record.update(
                {
                    "failed": True,
                    "errorText": request.failure,
                    "is_error": True,
                    "status_class": "failed",
                }
            )
            self._finish(request, record)
        except Exception as exc:  # noqa: BLE001
            logger.exception("requestfailed handler error: %s", exc)

    # ---------------- results ----------------

    def results(self) -> dict[str, Any]:
        requests = list(self._records.values())
        failed = [r for r in requests if r.get("failed")]

        def _in_class(cls: str) -> int:
            return sum(1 for r in requests if r.get("status_class") == cls)

        fetch_xhr = sum(1 for r in requests if r.get("resourceType") in {"fetch", "xhr"})
        slow = sum(
            1 for r in requests if (r.get("duration_ms") or 0) >= _SLOW_MS and not r.get("failed")
        )
        return {
            "summary": {
                "total": len(requests),
                "fetch_xhr": fetch_xhr,
                "http_2xx_3xx": _in_class("2xx") + _in_class("3xx"),
                "http_4xx": _in_class("4xx"),
                "http_5xx": _in_class("5xx"),
                "failed": len(failed),
                "slow": slow,
            },
            "requests": requests,
        }
