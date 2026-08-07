"""The stale-fill re-find and dialog-probe JS, executed against real Chromium.

Companions to tests/test_stale_fill.py (which fakes the JS): these prove the actual
templates — FIELD_REFIND_JS locates the live twin of a re-rendered field (light DOM and
open shadow roots), focuses it so keyboard insertion lands, and reads it back;
DIALOG_COUNT_JS / DIALOG_ANCESTOR_JS see dialogs wherever they render."""
import json

from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch


def _refind(op, attr="placeholder", value="Cost", tag="input", clear=True):
    return sc.FIELD_REFIND_JS % {
        "tag": json.dumps(tag), "attr": json.dumps(attr), "value": json.dumps(value),
        "op": json.dumps(op), "clear": json.dumps(clear),
    }


async def _page(pw, content):
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.set_content(content)
    return page


async def test_refind_focuses_unique_twin_and_reads_back_inserted_text():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <input placeholder="Description">
            <input placeholder="Cost">
        """)
        found = await page.evaluate(_refind("focus"))
        assert found["count"] == 1
        assert found["focused"] is True
        assert found["label"] == "Cost"
        await page.keyboard.insert_text("200")
        got = await page.evaluate(_refind("read"))
        assert got["value"] == "200"


async def test_refind_pierces_open_shadow_root():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="host"></div>
            <script>
              document.getElementById('host').attachShadow({mode: 'open'}).innerHTML =
                '<input placeholder="Cost">';
            </script>
        """)
        found = await page.evaluate(_refind("focus"))
        assert found["count"] == 1
        await page.keyboard.insert_text("200")
        got = await page.evaluate(_refind("read"))
        assert got["value"] == "200"


async def test_refind_reports_ambiguity_and_invisibility():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <input placeholder="Cost">
            <input placeholder="Cost">
            <div style="display:none"><input placeholder="Amount"></div>
        """)
        two = await page.evaluate(_refind("focus"))
        assert two["count"] == 2
        # The hidden field is not a live twin — zero matches, not a blind fill.
        hidden = await page.evaluate(_refind("focus", value="Amount"))
        assert hidden["count"] == 0


async def test_dialog_count_sees_visible_dialogs_only():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div role="dialog" style="width:200px;height:100px">open</div>
            <div role="dialog" style="display:none">hidden</div>
            <div class="ms-Modal is-open" style="width:10px;height:10px">fluent</div>
        """)
        got = await page.evaluate(sc.DIALOG_COUNT_JS)
        assert got["open"] == 2


async def test_dialog_count_sees_dialog_inside_open_shadow_root():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="host"></div>
            <script>
              document.getElementById('host').attachShadow({mode: 'open'}).innerHTML =
                '<div role="dialog" style="width:200px;height:100px">modal</div>';
            </script>
        """)
        got = await page.evaluate(sc.DIALOG_COUNT_JS)
        assert got["open"] == 1


async def test_dialog_ancestor_probe_walks_composed_tree():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div role="dialog" style="width:300px;height:200px">
              <button id="save">Save</button>
            </div>
            <button id="outside">Elsewhere</button>
        """)
        expr = "(el) => (%s).call(el)" % sc.DIALOG_ANCESTOR_JS
        inside = await page.evaluate(expr, await page.query_selector("#save"))
        outside = await page.evaluate(expr, await page.query_selector("#outside"))
        assert inside is True
        assert outside is False


# ------------------------- dialog identity stamping -------------------------
# The global dialog COUNT lies when panels chain (run 20260807_095537: Save closed its
# dialog, the Send-Email panel opened, count never dropped → false "STILL OPEN"). The
# stamp marks THE dialog the clicked element lives in; the post-click question is then
# about that specific element.

_STAMP_EXPR = "(el) => (%s).call(el)" % sc.DIALOG_STAMP_JS


async def test_dialog_stamp_marks_the_ancestor_and_sweeps_old_stamps():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="a" role="dialog" style="width:300px;height:200px">
              <button id="save-a">Save</button>
            </div>
            <div id="b" class="ms-Modal is-open" style="width:300px;height:200px">
              <button id="save-b">Save</button>
            </div>
        """)
        got = await page.evaluate(_STAMP_EXPR, await page.query_selector("#save-a"))
        assert got == {"stamped": True}
        stamped = await page.evaluate(
            "document.querySelectorAll('[%s]').length" % sc.DIALOG_WATCH_ATTR)
        assert stamped == 1
        assert await page.evaluate(
            "document.getElementById('a').hasAttribute('%s')" % sc.DIALOG_WATCH_ATTR)
        # A later click in another dialog sweeps the old stamp first — exactly one
        # watched dialog at a time.
        got = await page.evaluate(_STAMP_EXPR, await page.query_selector("#save-b"))
        assert got == {"stamped": True}
        assert not await page.evaluate(
            "document.getElementById('a').hasAttribute('%s')" % sc.DIALOG_WATCH_ATTR)
        assert await page.evaluate(
            "document.getElementById('b').hasAttribute('%s')" % sc.DIALOG_WATCH_ATTR)


async def test_dialog_stamp_false_outside_dialogs():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, '<button id="plain">Save</button>')
        got = await page.evaluate(_STAMP_EXPR, await page.query_selector("#plain"))
        assert got == {"stamped": False}


async def test_stamped_open_survives_a_chained_panel():
    """The seg6 failure at JS level: the stamped dialog unmounts, a NEW panel opens, the
    global count never drops — the stamped probe still answers 'closed'."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="a" role="dialog" style="width:300px;height:200px">
              <button id="save">Save</button>
            </div>
        """)
        await page.evaluate(_STAMP_EXPR, await page.query_selector("#save"))
        assert (await page.evaluate(sc.DIALOG_STAMPED_OPEN_JS))["present"] is True
        await page.evaluate("""() => {
            document.getElementById('a').remove();
            const next = document.createElement('div');
            next.setAttribute('role', 'dialog');
            next.style.cssText = 'width:300px;height:200px';
            next.textContent = 'Send email';
            document.body.appendChild(next);
        }""")
        assert (await page.evaluate(sc.DIALOG_COUNT_JS))["open"] == 1  # count masked...
        assert (await page.evaluate(sc.DIALOG_STAMPED_OPEN_JS))["present"] is False


async def test_stamped_open_treats_hidden_dialog_as_closed():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="a" role="dialog" style="width:300px;height:200px">
              <button id="save">Save</button>
            </div>
        """)
        await page.evaluate(_STAMP_EXPR, await page.query_selector("#save"))
        await page.evaluate("document.getElementById('a').style.display = 'none'")
        assert (await page.evaluate(sc.DIALOG_STAMPED_OPEN_JS))["present"] is False


async def test_stamp_and_probe_pierce_open_shadow_roots():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, """
            <div id="host"></div>
            <script>
              document.getElementById('host').attachShadow({mode: 'open'}).innerHTML =
                '<div id="a" role="dialog" style="width:300px;height:200px">' +
                '<button id="save">Save</button></div>';
            </script>
        """)
        save = await page.evaluate_handle(
            "document.getElementById('host').shadowRoot.getElementById('save')")
        got = await page.evaluate(_STAMP_EXPR, save)
        assert got == {"stamped": True}
        assert (await page.evaluate(sc.DIALOG_STAMPED_OPEN_JS))["present"] is True
