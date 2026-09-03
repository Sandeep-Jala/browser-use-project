"""A recorded `close` of the tab a segment worked in, and the replay that must repeat it.

Motivating failure (run 20260902_091047_561480, subtask 8 = entry 7320039db9ba7e26 "…then
click submit, and then close this tab"): the recording's last action is
`close(tab_id=FBAC)` — the portal tab it had been acting in — and its tab list drops from
two to one. compile_recording had no branch for `close`, so the action was dropped with no
warning and the compiled skill ended at the Submit click. On replay the portal tab
therefore stayed open, stayed pinned as HybridSession's aux page, and subtask 9 (whose own
recorded first action was a `switch` back to the app tab, also dropped) ran its Data
Request grid selectors against the portal:

    RuntimeError: no unique candidate matched:
      css=[role="row"]:has-text("PR/01797494/27/CDR110") span -> no match | ...

Only the `close` half is fixed here. A `switch` is still dropped — replay follows the tab
a recorded click OPENED (test_tab_following) and now the tab a recorded close SHUT, but it
has no way to name an arbitrary tab the agent switched to.
"""

import asyncio
from types import SimpleNamespace

from playwright.async_api import async_playwright
from test_compile_coverage import _item, _write

from automation.pipeline import hybrid
from automation.pipeline import script_compile as sc
from automation.skills import codegen
from automation.skills.api import SkillApi
from tests.test_heal_promotion import _launch

APP = {"url": "http://app/datarequests", "title": "Data Requests", "target_id": "4E6D"}
PORTAL = {"url": "http://portal/calcdatarequest", "title": "Portal", "target_id": "FBAC"}


def _close_item(tab_id, tabs, url, result=None):
    """One recorded close, with the tab list and current URL the recorder captured."""
    item = _item({"close": {"tab_id": tab_id}}, url=url, result=result)
    item["state"]["tabs"] = tabs
    return item


# ------------------------------- compile -------------------------------


def test_closing_the_tab_the_recording_was_acting_in_compiles(tmp_path):
    steps = sc.compile_recording(
        _write(tmp_path, [_close_item("FBAC", [APP, PORTAL], PORTAL["url"])]),
        emit_start_goto=False)
    assert steps == [{"action": "close_tab"}]


def test_closing_some_other_tab_is_dropped(tmp_path):
    """Recorded tab ids are per-run: a close of a tab the recording was NOT acting in
    names something replay cannot identify, and guessing would shut the app tab."""
    steps = sc.compile_recording(
        _write(tmp_path, [_close_item("4E6D", [APP, PORTAL], PORTAL["url"])]),
        emit_start_goto=False)
    assert steps == []


def test_a_refused_close_compiles_nothing(tmp_path):
    """The last-tab guard (agent_tools.build_tools) stamps no_close and shuts nothing —
    the same phantom-action rule the no_click/no_fill/no_scroll refusals follow."""
    refusal = [{"error": "REFUSED — did NOT close tab #FBAC: it is the LAST open tab",
                "metadata": {"no_close": True}}]
    steps = sc.compile_recording(
        _write(tmp_path, [_close_item("FBAC", [PORTAL], PORTAL["url"], result=refusal)]),
        emit_start_goto=False)
    assert steps == []


def test_the_real_recording_keeps_its_close():
    """Fixture from the live recording rather than a hand-written belief about the shape:
    repeat_click Next x11, find_by_text Submit, close."""
    steps = sc.compile_recording("library/7320039db9ba7e26.recording.json",
                                 emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "click", "close_tab"]
    assert steps[0]["count"] == 11


# ------------------------------- replay -------------------------------


async def _two_tabs(body):
    """An app tab and a portal tab in one context — the shape every close_tab replays in."""
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            context = await browser.new_context()
            app = await context.new_page()
            await app.set_content('<button id="in-app">Data Request</button>')
            portal = await context.new_page()
            await portal.set_content('<button id="in-portal">Submit</button>')
            return await body(app, portal, context)
        finally:
            await browser.close()


def test_run_steps_closes_its_tab_and_carries_on_in_the_survivor():
    async def body(app, portal, _context):
        out = await sc.run_steps(portal, [
            {"action": "click", "selectors": ['css=#in-portal']},
            {"action": "close_tab"},
            # Resolvable only in the app tab: proves the replay actually moved.
            {"action": "click", "selectors": ['css=#in-app']},
        ], timeout_ms=5000)
        return out, portal.is_closed(), app.is_closed()

    out, portal_closed, app_closed = asyncio.run(_two_tabs(body))
    assert out["error"] is None and out["executed"] == 3
    assert portal_closed is True and app_closed is False


def test_close_tab_leaves_the_last_open_tab_alone():
    """The replay twin of the live agent's last-tab refusal: closing the only tab destroys
    the page the workflow runs in. Nothing left to close is the post-close state already —
    a no-op, not a failed segment."""
    async def body(app, portal, _context):
        await portal.close()
        out = await sc.run_steps(app, [
            {"action": "close_tab"},
            {"action": "click", "selectors": ['css=#in-app']},
        ], timeout_ms=5000)
        return out, app.is_closed()

    out, app_closed = asyncio.run(_two_tabs(body))
    assert out["error"] is None and out["executed"] == 2
    assert app_closed is False


def test_the_code_tier_closes_its_tab_too():
    async def body(app, portal, _context):
        api = SkillApi(portal, {"submit": {"selectors": ['css=#in-portal']}},
                       timeout_ms=5000)
        await api.click("submit")
        await api.close_tab()
        return api.page is app, portal.is_closed(), api.executed

    moved, closed, executed = asyncio.run(_two_tabs(body))
    assert moved is True and closed is True and executed == 2


def test_close_tab_transpiles_into_the_code_tier():
    code, _anchors = codegen.transpile("sid", [
        {"action": "click", "selectors": ['css=#in-portal']},
        {"action": "close_tab"},
    ])
    assert "await api.close_tab()" in code
    assert codegen.lint_code(code) == []


# ------------------------------- the session's pin -------------------------------


async def test_a_closed_aux_pin_is_released_and_the_recording_follows():
    """A replayed close_tab takes the helper tab out from under HybridSession's aux pin.
    The pin must go with it — a stale pin is how the portal tab kept owning the session —
    and browser-use's focus must land on the survivor: the RecordingWatchdog streams ONE
    CDP session, so an unaligned focus freezes the run video from here on."""
    hs = hybrid.HybridSession.__new__(hybrid.HybridSession)
    closed_aux = SimpleNamespace(url="http://portal/calcdatarequest",
                                 is_closed=lambda: True)
    app = SimpleNamespace(url="http://app/datarequests", is_closed=lambda: False)
    hs._aux_page = closed_aux
    hs._main_page = app
    hs.pw_browser = SimpleNamespace(contexts=[SimpleNamespace(pages=[app])])
    focused = []

    async def fake_focus(page):
        focused.append(page)

    hs._focus_browser_use = fake_focus
    assert hs.current_page() is app
    assert hs._aux_page is None            # the dead pin no longer owns the session
    await asyncio.sleep(0)                 # let the fire-and-forget focus task run
    assert focused == [app]


def test_close_tab_lands_on_a_survivor_in_another_browser_context():
    """The app tab and the tab a click opened are normally siblings in one context, but
    the session pins pages across ALL contexts (HybridSession._pick_main_page). Refusing
    here because the only survivor sits in another context would silently leave the tab
    open — the very failure this step exists to prevent."""
    async def body():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                app = await (await browser.new_context()).new_page()
                await app.set_content('<button id="in-app">Data Request</button>')
                portal = await (await browser.new_context()).new_page()
                await portal.set_content('<button id="in-portal">Submit</button>')
                landed = await sc._close_and_return(portal)
                return landed is app, portal.is_closed()
            finally:
                await browser.close()

    landed_on_app, closed = asyncio.run(body())
    assert landed_on_app is True and closed is True
