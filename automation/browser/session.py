"""Browser session ownership.

INVERTED handoff (2026-07-23, tested end-to-end before migrating — see memory
file-upload-support): browser-use LAUNCHES the browser, and login.py CONNECTS to it over
CDP to authenticate, instead of the other way round. The session therefore classifies as
LOCAL (`is_local=True`), which is the code path browser-use actually maintains: the
upload_file FileSystem/allowlist validation runs (a remote-classified session waves ANY
path through — the ghost-upload crash), and the downloads watchdog manages its own
setDownloadBehavior. The prior architecture (login.py launches, browser-use attaches via
cdp_url) classified REMOTE and required compensating machinery for both.

`launch_session` builds the session but does NOT start it — the caller owns the
lifecycle: main() starts it, hands its cdp_url to login and to the replay engine's own
Playwright connection, and kills it at process end (`keep_alive=True` keeps browser-use
from tearing the browser down between agent segments).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from browser_use import BrowserProfile, BrowserSession

from automation.config import Config

logger = logging.getLogger("framework.session")

# Geometry comes from the real OS WINDOW, not viewport emulation: on a browser it owns,
# browser-use re-asserts an emulated viewport around every step's snapshot, and each
# re-assert reflows the page — a visible resize flicker on every single step (observed
# live right after the inversion). A plain window has no emulation to re-assert. The
# height adds ~90px of browser chrome so the CONTENT area stays ~1440×900 — the geometry
# this app's dense screens need (1280×720 crops toolbars and table rows off-screen).
_WINDOW = {"width": 1440, "height": 990}

# Chrome's "Save password?" bubble, every single run. It is not that the dismissal fails to
# stick — browser-use hands Chrome a BRAND NEW temp profile on every launch, so there is
# never a profile that remembers being told no. The fix has to be applied at launch, not
# clicked away: these are the prefs behind Settings > Autofill > "Offer to save passwords".
_NO_PASSWORD_PREFS = {
    "credentials_enable_service": False,          # the save-password bubble itself
    "credentials_enable_autosignin": False,       # the auto sign-in prompt
    "profile": {
        "password_manager_enabled": False,
        "password_manager_leak_detection": False,  # "your password was found in a breach"
    },
}

# Launch flags covering the same ground from the other side. The keychain pair matters on
# macOS specifically: without them Chrome asks the OS for "Chrome Safe Storage" access on
# first launch, which is a SECOND password-shaped popup and easy to mistake for the first.
# --disable-features merges with browser-use's own list rather than replacing it
# (BrowserProfile.get_args), so naming features here is safe.
_NO_PASSWORD_ARGS = [
    "--password-store=basic",
    "--use-mock-keychain",
    "--disable-save-password-bubble",
    "--disable-features=PasswordLeakDetection,AutofillEnableAccountWalletStorage",
]


def _write_no_password_prefs(session: BrowserSession) -> Path | None:
    """Seed the password prefs into the profile Chrome is ABOUT to launch.

    Timing is the whole trick: `user_data_dir` is None on a freshly built BrowserProfile and
    only becomes a real (already-created, empty) temp directory when BrowserSession validates
    it — which is why this takes the SESSION, reads `session.browser_profile`, and runs before
    `session.start()`. Pydantic copies the profile into the session, so the BrowserProfile
    object the caller built is not the one that launches; writing through it would silently
    do nothing.

    Returns the Preferences path, or None if the profile could not be seeded (in which case
    the launch flags still apply and a run is otherwise unaffected).
    """
    profile = session.browser_profile
    user_data_dir = getattr(profile, "user_data_dir", None)
    if not user_data_dir:
        logger.debug("no user_data_dir on the session profile; skipping password prefs")
        return None
    prefs_path = Path(user_data_dir) / (profile.profile_directory or "Default") / "Preferences"
    try:
        # Merge, never clobber: a seeded or reused profile may already carry state, and
        # Chrome treats an unparseable Preferences file as a corrupt profile.
        existing: dict = {}
        if prefs_path.exists():
            try:
                existing = json.loads(prefs_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("existing Preferences unreadable (%s); writing fresh", exc)
        merged = {**existing, **_NO_PASSWORD_PREFS,
                  "profile": {**existing.get("profile", {}), **_NO_PASSWORD_PREFS["profile"]}}
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        prefs_path.write_text(json.dumps(merged), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - a popup is never worth failing the launch over
        logger.warning("could not disable Chrome's password prompts: %s", exc)
        return None
    logger.debug("password manager disabled via %s", prefs_path)
    return prefs_path


def launch_session(config: Config, downloads_staging: Path | None = None) -> BrowserSession:
    """Build the browser-OWNING session (not started).

    Args:
        config: Framework configuration (headless flag, timing posture).
        downloads_staging: Stable folder for browser-use's downloads watchdog. The hybrid
            engine re-points download behavior at each run's artifacts downloads/ dir via
            CDP; anything that still lands here is copied out by the downloads_since
            safety net, so a downloaded file can never evaporate with a temp dir.
    """
    profile_kwargs: dict = {}
    if downloads_staging is not None:
        downloads_staging.mkdir(parents=True, exist_ok=True)
        profile_kwargs["downloads_path"] = str(downloads_staging)
    # Timing fields: make each step snapshot AFTER this slow React app has painted instead
    # of mid-render — stale element indexes from a mid-render snapshot are a direct
    # misclick source. Costs ~1-2s per step; each avoided misclick saves ≥3 LLM steps.
    profile = BrowserProfile(
        keep_alive=True,
        headless=config.headless,
        # No `viewport` (and no device_scale_factor — an emulation-only knob): emulation
        # is what flickered. window_size is inert after launch.
        viewport=None,
        window_size=dict(_WINDOW),
        minimum_wait_page_load_time=1.0,           # default 0.25 — let the SPA paint
        wait_for_network_idle_page_load_time=1.5,  # default 0.5
        wait_between_actions=0.5,                  # default 0.1 — DOM settles between actions
        highlight_elements=True,                   # explicit (browser-use 0.13.3 default)
        args=list(_NO_PASSWORD_ARGS),
        **profile_kwargs,
    )
    session = BrowserSession(browser_profile=profile)
    # AFTER the session exists (that is what materializes the temp profile dir) and BEFORE
    # the caller starts it — see _write_no_password_prefs.
    _write_no_password_prefs(session)
    return session
