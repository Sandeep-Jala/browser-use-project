"""Shadow-DOM piercing for the raw finders.

Motivating failure (run 20260805_123407_334719): the Add Expenses or Benefits modal
renders inside an open shadow root. find_by_text('Period to') answered "the element is
not in this page's DOM" while the label was visibly on screen — RAW_FIND_JS and
RAW_TEXT_FIND_JS walked document.body's light tree only, so every label lookup inside
the modal failed and the agent fell back to guessing anonymous combobox indexes (May-26
landed in Period FROM). Both finders now walk the composed tree through OPEN shadow
roots; closed roots stay invisible by construction."""
import json

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
