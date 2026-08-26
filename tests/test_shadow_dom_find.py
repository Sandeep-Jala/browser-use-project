"""Shadow-DOM piercing for the raw finders.

Motivating failure (run 20260805_123407_334719): the Add Expenses or Benefits modal
renders inside an open shadow root. find_by_text('Period to') answered "the element is
not in this page's DOM" while the label was visibly on screen — RAW_FIND_JS and
RAW_TEXT_FIND_JS walked document.body's light tree only, so every label lookup inside
the modal failed and the agent fell back to guessing anonymous combobox indexes (May-26
landed in Period FROM). Both finders now walk the composed tree through OPEN shadow
roots; closed roots stay invisible by construction."""
import asyncio
import json

from playwright.async_api import async_playwright

from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch


def _text_expr(tokens):
    return sc.RAW_TEXT_FIND_JS % json.dumps(tokens)


def _find_expr(tokens, click=False):
    return sc.RAW_FIND_JS % (json.dumps(tokens), "true" if click else "false")


# Light DOM carries none of the query text: every match below exists ONLY behind the
# shadow boundary, so a body-scoped walk (the old behaviour) returns count 0.
_SHADOW_PAGE = """
    <div id="light">nothing relevant here</div>
    <div id="host"></div>
    <script>
      document.getElementById('host').attachShadow({mode: 'open'}).innerHTML =
        '<div class="row"><div class="lbl">Period to</div>' +
        '<button onclick="window.__saved=1">Save entry</button></div>';
    </script>
"""


async def _page(pw, content=_SHADOW_PAGE):
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.set_content(content)
    return page


async def test_raw_text_finder_sees_static_text_in_open_shadow_root():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw)
        raw = await page.evaluate(_text_expr(["period", "to"]))
        assert raw.get("count") == 1
        assert "Period to" in (raw.get("name") or "")


async def test_raw_find_clicks_a_control_inside_an_open_shadow_root():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw)
        raw = await page.evaluate(_find_expr(["save", "entry"], click=True))
        assert raw.get("count") == 1
        assert raw.get("clicked") is True
        assert raw.get("name") == "Save entry"
        assert await page.evaluate("window.__saved") == 1


async def test_shadow_match_reports_no_light_dom_xpath():
    # A positional xpath is resolved against the DOCUMENT at replay time; for a
    # shadow-tree element that path would land on some unrelated light-DOM node.
    # No anchor is honest; a wrong-element anchor is not.
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw)
        raw = await page.evaluate(_text_expr(["period", "to"]))
        assert ((raw.get("element") or {}).get("xpath") or "") == ""


async def test_nested_open_shadow_roots_are_walked():
    from playwright.async_api import async_playwright

    nested = """
        <div id="outer"></div>
        <script>
          var outer = document.getElementById('outer').attachShadow({mode: 'open'});
          outer.innerHTML = '<div id="inner">padding text</div>';
          outer.getElementById('inner').attachShadow({mode: 'open'}).innerHTML =
            '<span>Employee contribution</span>';
        </script>
    """
    async with async_playwright() as pw:
        page = await _page(pw, nested)
        raw = await page.evaluate(_text_expr(["employee", "contribution"]))
        assert raw.get("count") == 1
        assert "Employee contribution" in (raw.get("name") or "")



# ---------- scrolling INSIDE a panel to reach virtualized rows (2026-08-13) ----------
# Run 20260814_105247 seg 6: the Add Data Request panel lists employees in its own
# scroll container and only renders the visible rows, so the newly created employee was
# not in the DOM at all. Wheel-scrolling the viewport centre moves the page behind the
# panel, not the panel — the finder must scroll the CONTAINERS.

_PANEL_PAGE = """
    <div id="main" style="height:400px">background grid</div>
    <div id="panel" style="position:fixed;right:0;top:0;width:300px;height:200px;
                           overflow-y:auto">
      <div id="rows"></div>
    </div>
    <script>
      var rows = document.getElementById('rows');
      var names = ['Louie Cameron','Arron McIntosh','Iain Black','Jaden Reid',
                   'Jude Clark','Aleksander Millar','Ruben Reid','Brody Docherty',
                   'Gordon Donaldson','Nathan Brown','Kaden Wilson','Artur Boyle'];
      // Virtualization stand-in: only rows near the scroll position exist. Like a real
      // virtualizer, the container keeps its FULL scroll height and row visibility is
      // computed from each row's data position — never from live layout: a display:none
      // row reads offsetTop 0, which collapses the panel's scrollHeight, clamps
      // scrollTop to ~80, and turns the test into a timing lottery (measured 2/10
      // failures) instead of a scroll-to-reach exercise.
      rows.style.position = 'relative';
      rows.style.height = (names.length * 40) + 'px';
      names.forEach(function (n, i) {
        var d = document.createElement('button');
        d.style.cssText = 'position:absolute;height:40px;top:' + (i * 40) + 'px;';
        d.dataset.idx = i; d.textContent = n;
        rows.appendChild(d);
      });
      function prune() {
        var p = document.getElementById('panel');
        Array.prototype.forEach.call(rows.children, function (c) {
          var top = c.dataset.idx * 40, vis = top > p.scrollTop - 80 &&
                                              top < p.scrollTop + p.clientHeight + 80;
          c.style.display = vis ? 'block' : 'none';
          c.textContent = vis ? names[c.dataset.idx] : '';
        });
      }
      prune();
      document.getElementById('panel').addEventListener('scroll', prune);
    </script>
"""


async def test_scroll_containers_reaches_a_row_below_a_panels_fold():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, _PANEL_PAGE)
        # Not rendered yet: the finder sees nothing.
        assert (await page.evaluate(_text_expr(["artur", "boyle"]))).get("count") == 0
        # Scrolling the panel's own container brings it into the DOM.
        for _ in range(6):
            moved = await sc._scroll_containers(page, 0.8)
            if (await page.evaluate(_text_expr(["artur", "boyle"]))).get("count"):
                break
            assert moved, "no scrollable container moved"
        assert (await page.evaluate(_text_expr(["artur", "boyle"]))).get("count") == 1


async def test_find_click_scrolls_panels_to_reach_its_target():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, _PANEL_PAGE)
        name = await sc._find_click(page, "Artur Boyle")
        assert "Artur Boyle" in name


# ---------- sweep direction: targets ABOVE the fold + failed-hunt reset ----------
# Run 20260817_133135: every miss-recovery scrolled DOWN-only from wherever the page
# stood, so a target ABOVE the current scroll was never re-rendered, and a failed hunt
# left the page parked at the very bottom — the user had to scroll back up by hand.


async def test_find_click_recovers_a_row_above_the_current_scroll():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, _PANEL_PAGE)
        await page.evaluate("document.getElementById('panel').scrollTop = 9999")
        await page.wait_for_timeout(150)          # prune() hides the top rows
        name = await sc._find_click(page, "Louie Cameron")   # row 1 of 12
        assert "Louie Cameron" in name


async def test_failed_hunt_resets_scroll_to_top():
    import pytest
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, _PANEL_PAGE)
        with pytest.raises(RuntimeError):
            await sc._find_click(page, "Zebra Nonexistent Person")
        assert await page.evaluate(
            "document.getElementById('panel').scrollTop") == 0


async def test_resolve_with_scroll_resets_top_first_and_after_failure(monkeypatch):
    import pytest

    calls = []

    async def fake_resolve(page, step, timeout):
        raise RuntimeError("miss")

    async def fake_tops(page):
        calls.append("top")
        return 1

    async def fake_wheel(page, pages, down=True):
        calls.append("wheel")

    monkeypatch.setattr(sc, "_resolve", fake_resolve)
    monkeypatch.setattr(sc, "_scroll_tops", fake_tops)
    monkeypatch.setattr(sc, "_wheel_scroll", fake_wheel)
    with pytest.raises(RuntimeError):
        await sc._resolve_with_scroll(object(), {"selectors": ["xpath=/x"]}, 1000)
    assert calls and calls[0] == "top"            # the hunt STARTS at the top
    assert calls[-1] == "top"                     # a failed hunt does not strand


# ------------------- shadow-hosted elements are never anchored by xpath -------------------
# Run 20260825_115047: the Add Payments amount field lives in an OPEN shadow root, and its
# committed anchor was an absolute xpath — which Playwright's document-scoped xpath engine
# can never resolve across a shadow boundary. The entry failed every replay with
# "no unique candidate matched: xpath=… -> no match" and could not have done otherwise.

_SHADOW_INPUT_PAGE = """
    <div id="root"><div id="host"></div></div>
    <script>
      const sr = document.getElementById('host').attachShadow({mode: 'open'});
      sr.innerHTML =
        '<div><div><input type="text" inputmode="decimal" class="form-control"'
        + ' placeholder="" style="width:150px"></div></div>';
    </script>
"""


async def _counts(page, selectors):
    out = {}
    for sel in selectors:
        try:
            out[sel] = await page.locator(sel).count()
        except Exception:  # noqa: BLE001 - an invalid engine counts as no match
            out[sel] = "error"
    return out


def test_xpath_cannot_cross_a_shadow_boundary_but_css_can():
    """The measurement the whole anchor rule rests on. If Playwright ever changes this,
    this test tells us before a library entry silently stops replaying."""
    async def go():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                page = await browser.new_page()
                await page.set_content(_SHADOW_INPUT_PAGE)
                return await _counts(page, [
                    "xpath=/html/body/div/div/div/div/input",
                    'xpath=//input[@inputmode="decimal"]',
                    'css=[inputmode="decimal"]',
                    "css=.form-control",
                ])
            finally:
                await browser.close()

    counts = asyncio.run(go())
    assert counts["xpath=/html/body/div/div/div/div/input"] == 0
    assert counts['xpath=//input[@inputmode="decimal"]'] == 0
    assert counts['css=[inputmode="decimal"]'] == 1
    assert counts["css=.form-control"] == 1


def test_the_compiled_css_anchor_resolves_where_the_xpath_could_not():
    """End to end: the selector compile now emits for that element actually finds it."""
    element = {"node_name": "INPUT", "x_path": "html/body/div/div/div/div/input",
               "attributes": {"type": "text", "inputmode": "decimal",
                              "class": "form-control form-control-sm", "placeholder": ""}}
    assert sc._selectors(element)[0].startswith("xpath=")          # light-DOM policy
    shadow_sels = sc._selectors(element, shadow_contained=True)
    assert all(not s.startswith("xpath=") for s in shadow_sels)
    assert shadow_sels == ['css=[inputmode="decimal"]']

    async def go():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                page = await browser.new_page()
                await page.set_content(_SHADOW_INPUT_PAGE)
                loc, sel, _healed = await sc._resolve(
                    page, {"selectors": shadow_sels}, 3000, require_editable=True)
                await loc.fill("4000")
                return sel, await loc.input_value()
            finally:
                await browser.close()

    sel, value = asyncio.run(go())
    assert sel == 'css=[inputmode="decimal"]' and value == "4000"


# --------- label-scoped anchors for shadow controls with no identity of their own ---------
# The expense/deduction panels' <select>s carry nothing but class and style, and xpath
# cannot reach them at all. The label beside them is their whole identity — and css reaches
# through the boundary to use it. Measured: two segments cost 320k tokens a run without
# this (run 20260825_132238).

_LABELLED_PANEL_PAGE = """
    <div id="root">
      <div class="r"><label>Amount</label><input type="text" value="light-dom"></div>
      <div id="host"></div>
    </div>
    <script>
      const sr = document.getElementById('host').attachShadow({mode: 'open'});
      sr.innerHTML =
        '<div class="p">'
      + '<div class="r"><label>Type</label><select class="fs"><option>Deduction</option></select></div>'
      + '<div class="r"><label>Period to</label><select class="fs"><option>Jun-26</option></select></div>'
      + '<div class="r"><label>Name</label><select class="fs"><option>Salary sacrifice</option></select></div>'
      + '</div>';
    </script>
"""


def test_a_bare_has_text_scope_is_ambiguous_but_the_child_constrained_one_is_not():
    """Why _label_scoped_css looks the way it does. `:has-text()` matches every ANCESTOR
    holding the text, so the obvious form selects the whole panel's controls."""
    async def go():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                page = await browser.new_page()
                await page.set_content(_LABELLED_PANEL_PAGE)
                return await _counts(page, [
                    "css=select.fs",
                    'css=div:has-text("Period to") select',            # the naive form
                    sc._label_scoped_css("select", "Period to"),       # what we emit
                ])
            finally:
                await browser.close()

    counts = asyncio.run(go())
    assert counts["css=select.fs"] == 3                       # class alone: no good
    assert counts['css=div:has-text("Period to") select'] == 3  # ancestors match too
    assert counts[sc._label_scoped_css("select", "Period to")] == 1


def test_the_label_scoped_anchor_picks_the_right_control_through_the_boundary():
    async def go():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                page = await browser.new_page()
                await page.set_content(_LABELLED_PANEL_PAGE)
                out = {}
                for label in ("Type", "Period to", "Name"):
                    step = {"selectors": [sc._label_scoped_css("select", label)]}
                    loc, _sel, _healed = await sc._resolve(page, step, 3000)
                    out[label] = await loc.input_value()
                return out
            finally:
                await browser.close()

    assert asyncio.run(go()) == {"Type": "Deduction", "Period to": "Jun-26",
                                 "Name": "Salary sacrifice"}


def test_a_label_that_repeats_outside_the_shadow_root_stays_ambiguous():
    """Honest limit: "Amount" also names a light-DOM field on this page, so the label
    anchor is not unique and _resolve falls through to the next candidate rather than
    picking one. That is why a real attribute is still ranked ahead of it."""
    async def go():
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                page = await browser.new_page()
                await page.set_content(_LABELLED_PANEL_PAGE)
                return await _counts(page, [sc._label_scoped_css("input", "Amount")])
            finally:
                await browser.close()

    assert list(asyncio.run(go()).values()) == [1]  # only the light-DOM one has an input row


# ---------------- the label rung, measured against the LIVE markup ----------------
# _label_scoped_css is the last anchor a control with no attribute identity has. Its
# previous shape required the control's DIRECT parent to hold the label text, and on
# 2026-08-25 that shape was measured against the real Add Employee form: 0 matches for
# EVERY field on it. Two library entries were committed with it as their only anchor and
# failed every replay ("no unique candidate matched: css=*:has(> input):has-text(\"NI
# number\") > input -> no match", runs 20260825_142300 / _144015 / _145430).
#
# The markup below is captured VERBATIM from the running app (the NI number field group
# and the Student loan toggle), so this test cannot encode a belief about the markup the
# way a hand-written fixture can. Only the two react-select chevron <path d> blobs are
# elided - noise, not structure.

_LIVE_NI_GROUP = """<div class="ms-Stack css-1246"><div class="ms-StackItem css-489"><label class="ms-Label labelStyle-1247">NI number<div class="ms-TooltipHost root-1248" role="none"><i data-icon-name="info" aria-hidden="true" class="clsIcon-1249"></i><div hidden="" id="tooltip1574" style="position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0px; border: 0px; overflow: hidden; white-space: nowrap;">National insurance numbers (NINo) should appear in the following combination of letters and numbers - two letters, six numbers, one letter. For example: AB 123456 C</div></div></label><div class="ms-TextField inputItem-1250"><div class="ms-TextField-wrapper"><div class="ms-TextField-fieldGroup fieldGroup-1134"><input type="text" id="TextField1575" class="ms-TextField-field field-1228" maxlength="9" aria-invalid="false" value=""></div></div></div></div><div class="ms-StackItem css-489"><label class="ms-Label labelStyle-1247">NI Category<div class="ms-TooltipHost root-1251" role="none"><i data-icon-name="info" aria-hidden="true" class="clsIcon-1249"></i><div hidden="" id="tooltip1580" style="position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0px; border: 0px; overflow: hidden; white-space: nowrap;"></div></div></label><div class="ms-Stack css-761"><div class="rs-container container-701"><span id="react-select-17-live-region" class="rs-a11y-text a11yText-699"></span><span class="rs-a11y-text a11yText-699" aria-live="polite" aria-atomic="false" aria-relevant="additions text"></span><div class="rs-control control-702"><div class="rs-value-container valueContainer-727"><div class="rs-placeholder placeholder-725" id="react-select-17-placeholder">Select</div><div class="rs-input-container inputContainer-709" data-value=""><input class="rs-input input-708" autocapitalize="none" autocomplete="off" autocorrect="off" id="react-select-17-input" spellcheck="false" tabindex="0" type="text" aria-autocomplete="list" aria-expanded="false" aria-haspopup="true" role="combobox" aria-describedby="react-select-17-placeholder" value="" style="color: inherit; background: 0px center; opacity: 1; width: 100%; grid-area: 1 / 2; font: inherit; min-width: 2px; border: 0px; margin: 0px; outline: 0px; padding: 0px;"></div></div><div class="rs-indicators-container indicatorsContainer-707"><span class="rs-indicator-separator indicatorSeparator-706"></span><div class="dropdown-indicator rs-dropdown-indicator dropdownIndicator-703" aria-hidden="true"><svg height="20" width="20" viewBox="0 0 20 20" aria-hidden="true" focusable="false" style="display: inline-block; fill: currentcolor; line-height: 1; stroke: currentcolor; stroke-width: 0;"><path></path></svg></div></div></div></div></div></div><div class="ms-StackItem css-489"><label class="ms-Label labelStyle-1247">Payment Mode</label><div class="ms-Stack css-838"><div class="rs-container container-701"><span id="react-select-18-live-region" class="rs-a11y-text a11yText-699"></span><span class="rs-a11y-text a11yText-699" aria-live="polite" aria-atomic="false" aria-relevant="additions text"></span><div class="rs-control control-702"><div class="rs-value-container valueContainer-727 value-container--has-value"><div class="rs-single-value singleValue-726">Other</div><div class="rs-input-container inputContainer-709" data-value=""><input class="rs-input input-708" autocapitalize="none" autocomplete="off" autocorrect="off" id="react-select-18-input" spellcheck="false" tabindex="0" type="text" aria-autocomplete="list" aria-expanded="false" aria-haspopup="true" role="combobox" value="" style="color: inherit; background: 0px center; opacity: 1; width: 100%; grid-area: 1 / 2; font: inherit; min-width: 2px; border: 0px; margin: 0px; outline: 0px; padding: 0px;"></div></div><div class="rs-indicators-container indicatorsContainer-707"><span class="rs-indicator-separator indicatorSeparator-706"></span><div class="dropdown-indicator rs-dropdown-indicator dropdownIndicator-703" aria-hidden="true"><svg height="20" width="20" viewBox="0 0 20 20" aria-hidden="true" focusable="false" style="display: inline-block; fill: currentcolor; line-height: 1; stroke: currentcolor; stroke-width: 0;"><path></path></svg></div></div></div></div></div></div></div>"""

_LIVE_TOGGLE = """<span class="value-1071"><div class="ms-Stack css-1288"><div class="ms-Toggle is-enabled clsToggle-1283"><label class="ms-Label ms-Toggle-label label-1286" for="Toggle1638" id="Toggle1638-label">Student loan</label><div class="ms-Toggle-innerContainer container-1285"><button class="ms-Toggle-background pill-1270" aria-checked="false" aria-labelledby="Toggle1638-label" data-is-focusable="true" data-ktp-target="true" id="Toggle1638" role="switch" type="button"><span class="ms-Toggle-thumb thumb-1271"></span></button></div></div></div></span>"""


async def _fixture_page(pw, markup):
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.set_content(f'<div id="form">{markup}</div>')
    return browser, page


def test_the_old_label_shape_found_nothing_on_the_live_markup():
    """The regression itself, pinned: four levels of Fluent nesting sit between the
    element carrying the label and the input, so a `> input` child constraint can never
    match. If this ever starts matching, the shape below is no longer needed."""
    async def go():
        async with async_playwright() as pw:
            browser, page = await _fixture_page(pw, _LIVE_NI_GROUP)
            try:
                return await page.locator(
                    'css=*:has(> input):has-text("NI number") > input').count()
            finally:
                await browser.close()

    assert asyncio.run(go()) == 0


def test_the_label_rung_resolves_the_control_its_label_names():
    """Every field in the captured group, located by nothing but the text beside it."""
    async def go():
        async with async_playwright() as pw:
            browser, page = await _fixture_page(pw, _LIVE_NI_GROUP)
            try:
                out = {}
                for label, tag in [("NI number", "input"), ("NI Category", "input"),
                                   ("Payment Mode", "input")]:
                    loc = page.locator(sc._label_scoped_css(tag, label))
                    count = await loc.count()
                    out[label] = (count,
                                  await loc.first.get_attribute("id") if count == 1
                                  else None)
                return out
            finally:
                await browser.close()

    out = asyncio.run(go())
    assert out["NI number"] == (1, "TextField1575")        # the attribute-less field
    assert out["NI Category"] == (1, "react-select-17-input")
    assert out["Payment Mode"] == (1, "react-select-18-input")


def test_the_label_rung_reaches_a_toggle_button_too():
    async def go():
        async with async_playwright() as pw:
            browser, page = await _fixture_page(pw, _LIVE_TOGGLE)
            try:
                loc = page.locator(sc._label_scoped_css("button", "Student loan"))
                return await loc.count(), await loc.first.get_attribute("id")
            finally:
                await browser.close()

    assert asyncio.run(go()) == (1, "Toggle1638")


def test_a_compiled_attributeless_fill_replays_through_the_label_rung():
    """End to end on the live markup: what compile emits for an attribute-less input,
    minus its recorded xpath (the shape that broke), still resolves and fills."""
    element = {"node_name": "INPUT", "x_path": "html/body/div/div/div/div/input",
               "attributes": {"type": "text", "id": "TextField1575", "maxlength": "9",
                              "class": "ms-TextField-field field-1228"}}
    sels = sc._selectors(element, label="NI number")
    assert sels[0].startswith("xpath=")                     # positional anchor still leads
    assert sels[-1] == sc._label_scoped_css("input", "NI number")

    async def go():
        async with async_playwright() as pw:
            browser, page = await _fixture_page(pw, _LIVE_NI_GROUP)
            try:
                loc, sel, _healed = await sc._resolve(
                    page, {"selectors": sels[1:]}, 3000, require_editable=True)
                await loc.fill("AB124557C")
                return sel, await loc.input_value()
            finally:
                await browser.close()

    sel, value = asyncio.run(go())
    assert value == "AB124557C"
    assert ":text-is" in sel


def test_an_element_with_a_real_attribute_gets_no_label_rung():
    """The rung is offered ONLY where the attribute ladder came back empty - a real
    attribute beats a label every time."""
    named = {"node_name": "INPUT", "x_path": "html/body/div/input",
             "attributes": {"placeholder": "First Name", "type": "text"}}
    assert sc._selectors(named, label="First name") == [
        "xpath=/html/body/div/input", 'css=[placeholder="First Name"]']
