"""Replay-side candidate NAMING must see label-ASSOCIATED names, not only the
element's own text/aria-label.

Runs 20260817_114057 + 115232: the Add-Employee 'Student loan' click is a Fluent
Toggle (id=Toggle2105 — volatile counter id, so the step anchors by positional xpath
gated on expect_text). A Fluent toggle's name lives in <label for=...> OUTSIDE the
button; the gate read only inner_text and aria-label, refused the RIGHT element on
every replay, and archive_if_failing retired the entry after 2 runs — a permanent
commit → 2 failed replays (each a ~100s takeover) → archive → re-author loop for any
label-named control (toggles, checkboxes, radios)."""
from playwright.async_api import async_playwright

from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch

_TOGGLE_PAGE = """
    <label for="Toggle21">Student loan</label>
    <button id="Toggle21" type="button" role="switch"></button>
    <span aria-labelledby="lbl-a lbl-b" id="combo"></span>
    <div id="lbl-a">Postgraduate</div><div id="lbl-b">loan</div>
    <label>Wrapped <button id="inner" type="button"></button></label>
    <button id="bare" type="button"></button>
"""


async def _page(pw):
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.set_content(_TOGGLE_PAGE)
    return page


async def test_label_for_names_a_fluent_toggle():
    async with async_playwright() as pw:
        page = await _page(pw)
        assert await sc._candidate_names_value(
            page.locator("#Toggle21"), "Student loan")


async def test_aria_labelledby_and_enclosing_label_name_too():
    async with async_playwright() as pw:
        page = await _page(pw)
        assert await sc._candidate_names_value(
            page.locator("#combo"), "Postgraduate loan")
        assert await sc._candidate_names_value(page.locator("#inner"), "Wrapped")


async def test_unnamed_element_still_fails_closed():
    async with async_playwright() as pw:
        page = await _page(pw)
        assert not await sc._candidate_names_value(
            page.locator("#bare"), "Student loan")


_ROW_PAGE = """
    <div role="row"><span>Aaran Duncan</span>
      <div id="rowcb" role="checkbox"></div></div>
    <table><tr><td>Harris Duncan</td><td><input id="cb2" type="checkbox"></td></tr></table>
"""


async def test_row_label_js_names_the_containing_row():
    # The probe behind anonymous-checkbox receipts: from the checkbox, find the row
    # element (role=row / tr / li) and return its visible text.
    from automation.pipeline import agent_tools

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_ROW_PAGE)
        expr = "el => (" + agent_tools._ROW_LABEL_JS + ").call(el)"
        assert "Aaran Duncan" in await page.locator("#rowcb").evaluate(expr)
        assert "Harris Duncan" in await page.locator("#cb2").evaluate(expr)


async def test_resolve_accepts_a_label_named_toggle_via_xpath():
    # The archived step's exact shape: positional xpath selector + expect_text naming
    # the control via its associated label. Toggle21 is the first <button> in <body>.
    step = {"action": "click", "selectors": ["xpath=/html/body/button[1]"],
            "expect_text": "Student loan"}
    async with async_playwright() as pw:
        page = await _page(pw)
        loc, sel, _healed = await sc._resolve(page, step, 3000)
        assert sel.startswith("xpath=")
        assert await loc.get_attribute("id") == "Toggle21"
