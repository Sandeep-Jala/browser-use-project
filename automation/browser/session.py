"""Browser session handoff.

The framework never launches its own browser for the agent. Instead it attaches
browser-use to the *already-authenticated* Chromium that `login.py` started (with its CDP
debugging port open). This keeps the authenticated session intact — no second login, no
second window.

`attach_session` returns a `BrowserSession` connected over CDP. Telemetry is NOT captured
here — the hybrid engine attaches its own Playwright CDP connection and collectors
(pipeline/runner.py); this profile only carries keep-alive and settle timing.
"""
from __future__ import annotations

from browser_use import BrowserProfile, BrowserSession

from automation.config import Config


async def attach_session(cdp_url: str, config: Config) -> BrowserSession:
    """Attach a browser-use BrowserSession to an existing browser via CDP.

    Args:
        cdp_url: The CDP endpoint of the already-running authenticated browser,
            e.g. "http://localhost:9222".
        config: Framework configuration (used for telemetry options in later phases).

    Returns:
        A connected `BrowserSession`. The session connects lazily on first use / when the
        Agent starts; we do not kill the underlying browser here since `login.py` owns it.
    """
    # `keep_alive=True` ensures browser-use does not tear down the browser that login.py
    # owns when the agent finishes. The timing fields make each step snapshot AFTER this
    # slow React app has painted instead of mid-render — stale element indexes from a
    # mid-render snapshot are a direct misclick source. Costs ~1-2s per step; each avoided
    # misclick saves ≥3 LLM steps plus the risk of poisoning a library recording.
    # Viewport is deliberately NOT set here: login.py's context owns screenshot geometry.
    profile = BrowserProfile(
        keep_alive=True,
        minimum_wait_page_load_time=1.0,           # default 0.25 — let the SPA paint
        wait_for_network_idle_page_load_time=1.5,  # default 0.5
        wait_between_actions=0.5,                  # default 0.1 — DOM settles between actions
        highlight_elements=True,                   # explicit (browser-use 0.13.3 default)
    )

    session = BrowserSession(
        cdp_url=cdp_url,
        browser_profile=profile,
    )
    return session
