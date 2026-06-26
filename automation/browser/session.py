"""Browser session handoff.

The framework never launches its own browser for the agent. Instead it attaches
browser-use to the *already-authenticated* Chromium that `login.py` started (with its CDP
debugging port open). This keeps the authenticated session intact — no second login, no
second window.

`attach_session` returns a `BrowserSession` connected over CDP. Telemetry-related profile
options (recording, HAR, etc.) are intentionally left off in Phase 0; later phases extend
the profile here.
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
    # Minimal profile for Phase 0. `keep_alive=True` ensures browser-use does not try to
    # tear down the browser that login.py owns when the agent finishes.
    profile = BrowserProfile(keep_alive=True)

    session = BrowserSession(
        cdp_url=cdp_url,
        browser_profile=profile,
    )
    return session
