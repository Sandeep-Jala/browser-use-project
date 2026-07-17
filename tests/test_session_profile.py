"""attach_session's BrowserProfile must carry the misclick-mitigation timing fields.

Guards the kwarg NAMES against the pinned browser-use 0.13.3: a renamed/typo'd profile
field would silently fall back to its default in a live run — here it fails an assert
(or raises at construction) instead. No browser is needed: the session connects lazily.
"""
from automation.browser.session import attach_session


async def test_profile_carries_timing_fields():
    session = await attach_session("http://localhost:1", config=None)
    profile = session.browser_profile
    assert profile.keep_alive is True
    assert profile.minimum_wait_page_load_time == 1.0
    assert profile.wait_for_network_idle_page_load_time == 1.5
    assert profile.wait_between_actions == 0.5
    assert profile.highlight_elements is True
    # login.py's context owns screenshot geometry; the profile must not fight it.
    assert profile.viewport is None
