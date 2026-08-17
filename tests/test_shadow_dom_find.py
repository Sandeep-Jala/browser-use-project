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
