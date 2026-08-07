"""Network request/response logging via Playwright.

Listens to each page's Playwright network events — ``request`` (a request was issued),
``response`` (headers arrived), and ``requestfailed`` (the request never completed). Each
request is aggregated into one record keyed by the Playwright ``Request`` object, capturing
the agent step it fired on, method, URL, resource type, status (+ a coarse 2xx/3xx/4xx/5xx
class), wall-clock duration, request/response headers, and failure text. Output is one JSON
file per run.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from playwright.async_api import Page, Request, Response

from automation.collectors.base import Collector

logger = logging.getLogger("framework.collector.network")

# Requests slower than this (ms) are flagged as "slow" in the summary.
_SLOW_MS = 1000.0

# Create-write response BODIES are captured (JSON only, capped) — they are the run's
# authoritative record of what it just created (new ids/refs), which the hybrid engine's
# runtime bindings resolve against. Reads are never captured.
_WRITE_METHODS = {"POST", "PUT", "PATCH"}
_BODY_CAP = 16 * 1024
_MAX_BODIES = 50

# Credential-bearing headers are never persisted: network.json and report.html are meant to
# be shared, and a raw Authorization header is a live session token.
_SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie",
                      "x-api-key", "api-key", "x-auth-token"}
_REDACTED = "«redacted»"


def _redact(headers: dict[str, Any]) -> dict[str, Any]:
    """Copy `headers` with credential values replaced (case-insensitive key match)."""
    return {k: (_REDACTED if k.lower() in _SENSITIVE_HEADERS else v)
            for k, v in headers.items()}


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
        # Requests dropped by the capture scope (issued by an out-of-scope page). The
        # drop must be STICKY: membership here keeps a later response/finished event from
        # resurrecting the request as a partial record via _record_for's setdefault.
        self._skipped: set[Request] = set()
        self._bodies_captured = 0

    def _attach_page(self, page: Page) -> None:
        # Bind the page so each record notes WHICH page issued the request (`page_url`) —
        # evidence for assertion scoping (helper-tab traffic vs. the app's own).
        page.on("request", lambda r: self._on_request(r, page))
        page.on("response", self._on_response)
        page.on("requestfinished", self._on_request_finished)
        page.on("requestfailed", self._on_request_failed)

    # ---------------- Playwright event handlers (sync) ----------------

    def _record_for(self, request: Request, page: Page | None = None) -> dict[str, Any]:
        record = self._records.setdefault(
            request,
            {
                "step": self.current_step,
                "url": request.url,
                "method": request.method,
                "resourceType": request.resource_type,
                "page_url": None,
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
        if page is not None and not record.get("page_url"):
            try:
                record["page_url"] = page.url
            except Exception:  # noqa: BLE001 - page may already be closed
                pass
        return record

    def _finish(self, request: Request, record: dict[str, Any]) -> None:
        start = self._started_at.get(request)
        if start is not None and record.get("duration_ms") is None:
            record["duration_ms"] = round((time.monotonic() - start) * 1000, 2)

    def _on_request(self, request: Request, page: Page | None = None) -> None:
        if not self._active:
            return
        page_url: str | None = None
        if page is not None:
            try:
                page_url = page.url
            except Exception:  # noqa: BLE001 - page may already be closed (fail open)
                page_url = None
        if not self._page_in_scope(page_url):
            self._skipped.add(request)
            return
        record = self._record_for(request, page)
        self._started_at.setdefault(request, time.monotonic())
        try:
            record["request_headers"] = _redact(dict(request.headers))
        except Exception as exc:  # noqa: BLE001
            logger.exception("request headers error: %s", exc)

    def _on_response(self, response: Response) -> None:
        if not self._active:
            return
        try:
            request = response.request
            if request in self._skipped:
                return
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
                record["response_headers"] = _redact(dict(response.headers))
            except Exception as exc:  # noqa: BLE001
                logger.exception("response headers error: %s", exc)
            self._maybe_capture_body(request, response, record)
            self._finish(request, record)
        except Exception as exc:  # noqa: BLE001
            logger.exception("response handler error: %s", exc)

    def _maybe_capture_body(self, request: Request, response: Response,
                            record: dict[str, Any]) -> None:
        """Schedule body capture for a successful create-write's JSON response (the sync
        event handler cannot await). Narrow by construction: writes only, JSON only,
        size- and count-capped — the run's created-record ids/refs, not a traffic dump."""
        if (request.method not in _WRITE_METHODS
                or not (200 <= (response.status or 0) < 400)
                or self._bodies_captured >= _MAX_BODIES):
            return
        ctype = ""
        try:
            ctype = str(response.headers.get("content-type", ""))
        except Exception:  # noqa: BLE001 - headers unreadable; treat as non-JSON
            pass
        if "json" not in ctype.lower():
            return
        self._bodies_captured += 1

        async def _capture() -> None:
            try:
                record["body"] = (await response.text())[:_BODY_CAP]
            except Exception as exc:  # noqa: BLE001 - body gone (redirect/stream); skip
                logger.debug("body capture skipped for %s: %s", record.get("url"), exc)

        try:
            asyncio.get_running_loop().create_task(_capture())
        except RuntimeError:  # no running loop (sync tests): capture is best-effort
            pass

    def _on_request_finished(self, request: Request) -> None:
        if not self._active or request in self._skipped:
            return
        self._finish(request, self._record_for(request))

    def _on_request_failed(self, request: Request) -> None:
        if not self._active or request in self._skipped:
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

    # ---------------- live queries ----------------

    # Write methods the live-receipt probe reports (DELETE included: a destructive click
    # deserves a receipt too; the body-capture set above stays creates-only).
    _RECEIPT_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def writes_since(self, t0: float) -> list[dict[str, Any]]:
        """Fetch/xhr WRITE requests that STARTED at/after monotonic `t0`, oldest first,
        as {"record", "started", "settled"}. `record` is the LIVE aggregation dict —
        status/body may land after this call, so callers re-poll rather than copy. This
        is the click receipt's ground truth: the server's own answer to "did my save
        actually happen", available mid-run instead of only in network.json afterwards."""
        out: list[dict[str, Any]] = []
        for request, started in list(self._started_at.items()):
            if started < t0 or request in self._skipped:
                continue
            record = self._records.get(request)
            if record is None:
                continue
            if record.get("method") not in self._RECEIPT_WRITE_METHODS:
                continue
            if record.get("resourceType") not in ("fetch", "xhr"):
                continue
            settled = record.get("duration_ms") is not None or bool(record.get("failed"))
            out.append({"record": record, "started": started, "settled": settled})
        out.sort(key=lambda w: w["started"])
        return out

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
                # Requests dropped because an out-of-scope page issued them (helper-tab
                # ad stacks) — recorded so the artifact says what it excluded.
                "out_of_scope": len(self._skipped),
            },
            "requests": requests,
        }
