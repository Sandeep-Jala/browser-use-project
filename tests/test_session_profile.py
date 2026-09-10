"""launch_session's BrowserProfile must carry the launch posture exactly.

Guards the kwarg NAMES against the pinned browser-use 0.13.3: a renamed/typo'd profile
field would silently fall back to its default in a live run — here it fails an assert
(or raises at construction) instead. No browser is needed: the session launches lazily
on start(), which these tests never call.

The INVERTED handoff (2026-07-23) rides on this profile: no cdp_url means browser-use
LAUNCHES and owns the browser (`is_local=True` — the code path where upload validation
and download handling actually work), login.py connects over CDP afterwards, and the
window/viewport geometry that used to live in login.py's context is owned here.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from automation.browser.session import _write_no_password_prefs, launch_session


def _config(headless=True):
    return SimpleNamespace(headless=headless)


def test_profile_carries_timing_and_geometry_fields(tmp_path):
    session = launch_session(_config(), tmp_path / "staging")
    profile = session.browser_profile
    assert profile.keep_alive is True
    assert profile.minimum_wait_page_load_time == 1.0
    assert profile.wait_for_network_idle_page_load_time == 1.5
    assert profile.wait_between_actions == 0.5
    assert profile.highlight_elements is True
    # Geometry: we pass viewport=None + window_size. Under HEADLESS the profile
    # validator derives an emulated viewport from window_size (there is no OS window to
    # supply geometry — and nobody watches headless, so emulation is harmless there).
    assert (profile.viewport.width, profile.viewport.height) == (1440, 990)
    assert profile.headless is True


def test_no_cdp_url_means_browser_use_owns_the_browser(tmp_path):
    """The whole point of the inversion: launching (not attaching) classifies the
    session LOCAL, which is the browser-use code path where upload_file validates paths
    and the downloads watchdog manages download behavior."""
    session = launch_session(_config(headless=False), tmp_path / "staging")
    assert session.browser_profile.cdp_url is None
    assert session.browser_profile.is_local is True
    assert session.browser_profile.headless is False
    ws = session.browser_profile.window_size
    # 990 = ~900 of CONTENT plus browser chrome; emulation-free geometry.
    assert (ws.width, ws.height) == (1440, 990)
    # HEADFUL (every live run) keeps viewport=None — no emulation, so nothing gets
    # re-asserted around each step: this is the fix for the per-step resize flicker
    # the user reported after the inversion.
    assert session.browser_profile.viewport is None


def test_downloads_staging_created_and_carried(tmp_path):
    staging = tmp_path / "artifacts" / ".downloads_staging"
    session = launch_session(_config(), staging)
    assert str(session.browser_profile.downloads_path) == str(staging)
    assert staging.is_dir()
    # No staging dir -> profile default (browser-use picks its own); the per-run CDP
    # override + downloads_since copy safety net still route files into run artifacts.
    session = launch_session(_config(), None)
    assert session is not None


# ------------------------- Chrome's "Save password?" bubble -------------------------
# It fired on every single run because browser-use hands Chrome a FRESH temp profile each
# launch, so no profile ever remembers being told no. Verified live: Chrome rewrote the
# seeded Preferences from 3 keys to 41 (taking ownership) and KEPT both values false.

def _prefs(session) -> dict:
    path = (Path(session.browser_profile.user_data_dir)
            / session.browser_profile.profile_directory / "Preferences")
    return json.loads(path.read_text(encoding="utf-8"))


def test_password_manager_is_disabled_in_the_profile_chrome_will_launch(tmp_path):
    prefs = _prefs(launch_session(_config(), tmp_path / "staging"))

    assert prefs["credentials_enable_service"] is False        # the save bubble itself
    assert prefs["credentials_enable_autosignin"] is False
    assert prefs["profile"]["password_manager_enabled"] is False
    assert prefs["profile"]["password_manager_leak_detection"] is False


def test_prefs_land_in_the_sessions_own_profile_not_the_one_we_built(tmp_path):
    """The subtle one. BrowserProfile.user_data_dir is None until BrowserSession VALIDATES
    it, and pydantic validation COPIES the profile — so the object launch_session builds is
    not the object Chrome launches from. Seeding through it writes nothing, and the bubble
    comes back with no error anywhere."""
    session = launch_session(_config(), tmp_path / "staging")

    user_data_dir = session.browser_profile.user_data_dir
    assert user_data_dir, "the session's profile is the one that gets a real temp dir"
    assert (Path(user_data_dir) / "Default" / "Preferences").is_file()


def test_launch_flags_cover_the_macos_keychain_prompt_too(tmp_path):
    """--use-mock-keychain / --password-store=basic stop Chrome asking the OS for 'Chrome
    Safe Storage' access, which is a second password-shaped popup on macOS."""
    args = launch_session(_config(), tmp_path / "staging").browser_profile.get_args()

    assert "--password-store=basic" in args
    assert "--use-mock-keychain" in args
    # browser-use MERGES every --disable-features value into one arg; ours must not have
    # displaced its own list (that would silently re-enable 30 components).
    features = [a for a in args if a.startswith("--disable-features=")]
    assert len(features) == 1
    assert "PasswordLeakDetection" in features[0]
    assert "AutofillServerCommunication" in features[0]


def test_existing_profile_prefs_are_merged_not_clobbered(tmp_path):
    """Chrome treats an unparseable Preferences file as a corrupt profile, and a reused
    profile may already carry state worth keeping."""
    session = launch_session(_config(), tmp_path / "staging")
    prefs_path = (Path(session.browser_profile.user_data_dir) / "Default" / "Preferences")
    prefs_path.write_text(json.dumps({
        "keep_me": 1,
        "credentials_enable_service": True,               # to be overridden
        "profile": {"exit_type": "Normal", "password_manager_enabled": True},
    }), encoding="utf-8")

    assert _write_no_password_prefs(session) == prefs_path
    prefs = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert prefs["keep_me"] == 1                          # unrelated key survived
    assert prefs["profile"]["exit_type"] == "Normal"      # nested sibling survived
    assert prefs["credentials_enable_service"] is False   # ours won
    assert prefs["profile"]["password_manager_enabled"] is False


def test_corrupt_existing_prefs_are_replaced_rather_than_crashing(tmp_path):
    session = launch_session(_config(), tmp_path / "staging")
    prefs_path = (Path(session.browser_profile.user_data_dir) / "Default" / "Preferences")
    prefs_path.write_text("{not json at all", encoding="utf-8")

    assert _write_no_password_prefs(session) == prefs_path
    assert json.loads(prefs_path.read_text())["credentials_enable_service"] is False


def test_unseedable_profile_degrades_instead_of_failing_the_launch():
    """No temp dir (a browser-use change, say) must not stop a run — the flags still apply."""
    assert _write_no_password_prefs(
        SimpleNamespace(browser_profile=SimpleNamespace(
            user_data_dir=None, profile_directory="Default"))) is None
