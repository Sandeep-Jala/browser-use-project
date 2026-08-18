"""Replay-side candidate NAMING must see label-ASSOCIATED names, not only the
element's own text/aria-label.

Runs 20260817_114057 + 115232: the Add-Employee 'Student loan' click is a Fluent
Toggle (id=Toggle2105 — volatile counter id, so the step anchors by positional xpath
gated on expect_text). A Fluent toggle's name lives in <label for=...> OUTSIDE the
button; the gate read only inner_text and aria-label, refused the RIGHT element on
every replay, and archive_if_failing retired the entry after 2 runs — a permanent
commit → 2 failed replays (each a ~100s takeover) → archive → re-author loop for any
label-named control (toggles, checkboxes, radios)."""
import pytest
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


"""Run 20260818_091836 seg 4: the same gate refused an ICON-ONLY container.

The Net-to-Gross recording captured two clicks that landed on the grid CELL rather
than on the pencil inside it (`Clicked td ""`). Compile stamps expect_text from
browser-use's ax_name, and a table cell's accessibility name is built from its
CONTENTS — so the step carries expect_text="Net to gross" while the cell's own
inner_text is '' (a Fluent icon renders through a CSS glyph), its aria-label is
absent, and no label points at it. Every replay therefore died on
    "1 visible match(es), none named \"Net to gross\""
and the whole segment fell into a dirty-state agent takeover at the popup.

Capture reads the name off the contents; replay must read the same layer back.
"""

_ICON_CELL_PAGE = """
    <table><tr>
      <td id="cell"><i id="pencil" role="img" aria-label="Net to gross"
                       title="Net to gross"></i></td>
      <td id="other"><i role="img" aria-label="Delete row"></i></td>
      <td id="empty"></td>
    </tr></table>
"""


async def _icon_page(pw):
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.set_content(_ICON_CELL_PAGE)
    return page


async def test_icon_only_cell_is_named_by_the_icon_it_contains():
    async with async_playwright() as pw:
        page = await _icon_page(pw)
        assert await sc._candidate_names_value(page.locator("#cell"), "Net to gross")
        # The element that owns the label keeps working through the earlier readers.
        assert await sc._candidate_names_value(page.locator("#pencil"), "Net to gross")


async def test_a_differently_named_container_still_fails_closed():
    # The wrong-row guard: contents-derived naming must not turn into "anything goes".
    async with async_playwright() as pw:
        page = await _icon_page(pw)
        assert not await sc._candidate_names_value(page.locator("#other"), "Net to gross")
        assert not await sc._candidate_names_value(page.locator("#empty"), "Net to gross")


async def test_resolve_accepts_the_icon_cell_via_its_recorded_xpath():
    # The exact failing step from library/39d62877f6ddd9e7.template.json (call 8).
    step = {"action": "click", "selectors": ["xpath=/html/body/table/tbody/tr/td[1]"],
            "expect_text": "Net to gross"}
    async with async_playwright() as pw:
        page = await _icon_page(pw)
        loc, sel, _healed = await sc._resolve(page, step, 3000)
        assert sel.startswith("xpath=")
        assert await loc.get_attribute("id") == "cell"


_GRID_ROW = """
    <table><tbody>
      <tr id="r10"><td>Jan-27</td><td><input value="5000"></td>
        <td>£3,315.64</td><td><i aria-label="Net to gross"></i></td></tr>
      <tr id="r11"><td>Feb-27</td><td><input value="5000"></td>
        <td>£3,315.64</td><td><i aria-label="Net to gross"></i></td></tr>
    </tbody></table>
"""


async def test_a_scattered_expect_verifies_the_way_find_by_text_matched():
    """The row click of run 20260818_11xx. find_by_text('Feb-27 Net to gross') matched
    this <tr> on a haystack — 'Feb-27' from the month cell, 'Net to gross' from the
    pencil's aria-label, tokens anywhere. _names_value then demanded them CONSECUTIVE in
    the row's text and refused the row on every replay; only the hover-reveal fallback
    clicked it, unverified."""
    step = {"action": "click", "selectors": ["xpath=/html/body/table/tbody/tr[2]"],
            "expect_text": "Feb-27 Net to gross", "expect_scattered": True}
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_GRID_ROW)
        loc, _sel, _healed = await sc._resolve(page, step, 3000)
        assert await loc.get_attribute("id") == "r11"

        # The SAME step without the flag keeps today's strict rule and refuses.
        strict = dict(step)
        strict.pop("expect_scattered")
        with pytest.raises(RuntimeError, match="none named"):
            await sc._resolve(page, strict, 1000)


async def test_scattered_still_refuses_a_row_missing_one_token():
    # Scattered ≠ anything goes: every token must still be present on the candidate.
    step = {"action": "click", "selectors": ["xpath=/html/body/table/tbody/tr[1]"],
            "expect_text": "Feb-27 Net to gross", "expect_scattered": True}
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_GRID_ROW)          # tr[1] is Jan-27
        with pytest.raises(RuntimeError, match="none named"):
            await sc._resolve(page, step, 1000)


async def test_the_fingerprint_gate_ignores_a_fluent_counter_id():
    """The popup fill's xpath candidate: it lands on the RIGHT input, but the gate
    compared the recorded TextField99 against the live id and called it positional
    drift. Volatile ids are already skipped — TextField99 just was not classified so."""
    fp = {"tag": "input", "attrs": {"id": "TextField99", "type": "number"}}
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content('<input id="TextField130" type="number">')
        assert await sc._xpath_matches_fingerprint(page.locator("#TextField130"), fp)
        # A genuinely stable id must still be compared, or positional drift goes unseen.
        assert not await sc._xpath_matches_fingerprint(
            page.locator("#TextField130"), {"tag": "input", "attrs": {"id": "btnSave"}})
