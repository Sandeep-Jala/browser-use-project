"""Failure-capture helpers.

Currently provides a single helper used by `login.py` to snapshot the page when login
fails, so failures are debuggable after the browser has closed. Screenshots are written
to the configured errors directory (gitignored).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page

logger = logging.getLogger("framework.error_capture")

async def save_login_error_screenshot(
    page: Page,
    errors_dir: str | Path,
    name: str,
) -> Path | None:
    """Save a full-page screenshot of `page` into `errors_dir`.

    Returns the path written, or None if the screenshot itself failed (we never let a
    capture failure mask the original login error).
    """
    errors_dir = Path(errors_dir)
    try:
        errors_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = errors_dir / f"{name}_{timestamp}.png"
        await page.screenshot(path=str(out_path), full_page=True)
        print(f"[!] Saved failure screenshot: {out_path}")
        return out_path
    except Exception as exc:  # noqa: BLE001 - best-effort, must not raise
        logger.exception("Failed to capture error screenshot (%s)", name)
        return None
