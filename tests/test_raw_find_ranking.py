"""RAW_FIND_JS name-specificity ranking and the hidden-click receipt.

Motivating failure (two live runs of nps_02): find_by_text('Reviews', click_first=true)
blind-clicked the hidden "Add reviews (Alt+Shift+R)" header button — the matcher required
only that every query token appear somewhere in the candidate's haystack, and the sort key
was visibility alone. Ranking now prefers, within a visibility tier, an element NAMED
'Reviews' over 'Add reviews'; and when nothing better-named exists, the receipt warns
instead of advising the re-call that looped the agent into the same wrong click."""
import json

from automation.pipeline import agent_tools
from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch


def _expr(tokens, click=False):
    return sc.RAW_FIND_JS % (json.dumps(tokens), "true" if click else "false")


# ------------------------------- ranking on live DOM -------------------------------


async def test_exact_name_beats_prefix_beats_substring_beats_scattered():
    """DOM order deliberately puts the WRONG candidates first: without the rank key the
    stable visible-first sort would return 'Add reviews' on top (the observed misfire)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <button title="Add reviews (Alt+Shift+R)">+</button>
            <button><i title="reviews icon"></i>Open</button>
            <button>Reviews archive</button>
            <button>Reviews</button>
        """)
        raw = await page.evaluate(_expr(["reviews"]))
        assert raw["count"] == 4
        assert raw["names"] == [
            "Reviews",                      # exact name match
            "Reviews archive",              # word-aligned prefix
            "Add reviews (Alt+Shift+R)",    # whole-phrase substring
            "Open",                         # tokens only in a child icon hint (scattered)
        ]
        assert raw["clicked"] is False


async def test_visibility_stays_the_primary_key():
    """An invisible exact match must NOT outrank a visible lesser match — visible-first
    remains the primary sort key, name specificity only breaks ties within a tier."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <div style="display:none"><button>Reviews</button></div>
            <button title="Add reviews (Alt+Shift+R)">+</button>
        """)
        raw = await page.evaluate(_expr(["reviews"]))
        assert raw["names"][0] == "Add reviews (Alt+Shift+R)"


async def test_click_lands_on_the_best_ranked_match():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <button onclick="window.__hit = 'add'" title="Add reviews">+</button>
            <button onclick="window.__hit = 'reviews'">Reviews</button>
        """)
        raw = await page.evaluate(_expr(["reviews"], click=True))
        assert raw["clicked"] is True
        assert raw["name"] == "Reviews"
        assert await page.evaluate("window.__hit") == "reviews"


# --------------------- click refusal: invisible + name mismatch ---------------------
#
# Motivating failure (payroll run 20260803_112915): find_by_text('Aaran Macleod')
# fell to the raw-DOM path, "clicked" the 0-size background employees-grid row whose
# name was 'AM Aaran Macleod Tax Code : 1257L ...' with a bare el.click() (a no-op —
# React listens on mousedown), and reported "✅ ALREADY CLICKED" four times; the agent
# looped until fail_and_stop. An INVISIBLE candidate whose name does not match the
# query is never the intended target — refuse the click and say so.


async def test_no_click_when_best_match_is_invisible_and_name_mismatched():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <div style="display:none">
              <button onclick="window.__hit = 'row'"
                      title="AM Aaran Macleod Tax Code : 1257L Cum">row</button>
            </div>
        """)
        raw = await page.evaluate(_expr(["aaran", "macleod"], click=True))
        assert raw["count"] == 1
        assert raw["clicked"] is False
        assert await page.evaluate("window.__hit === undefined")


async def test_invisible_exact_name_still_clicks():
    """The legit 0-size case (Fluent's 'Send NPS survey request' icon) must keep working:
    when the name IS the query, invisibility alone is not a reason to refuse."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <div style="position:absolute;width:0;height:0;overflow:hidden">
              <button onclick="window.__hit = 'nps'"
                      title="Send NPS survey request">go</button>
            </div>
        """)
        raw = await page.evaluate(_expr(["send", "nps", "survey", "request"], click=True))
        assert raw["clicked"] is True
        assert await page.evaluate("window.__hit") == "nps"


async def test_invisible_prefix_extension_still_clicks():
    # rank 1 (word-aligned prefix) is a trusted name — same tier the receipt trusts.
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <div style="display:none">
              <button onclick="window.__hit = 'ok'" title="Send NPS survey request">go</button>
            </div>
        """)
        raw = await page.evaluate(_expr(["send", "nps"], click=True))
        assert raw["clicked"] is True
        assert await page.evaluate("window.__hit") == "ok"


async def test_visible_mismatch_still_clicks():
    """Visibility keeps its old meaning: a visible best match is clicked even when the
    name is broader than the query (the pre-existing 'Add reviews' warn-but-click path)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <button onclick="window.__hit = 'add'" title="Add reviews (Alt+Shift+R)">+</button>
        """)
        raw = await page.evaluate(_expr(["reviews"], click=True))
        assert raw["clicked"] is True
        assert await page.evaluate("window.__hit") == "add"


async def test_raw_click_fires_full_mouse_sequence():
    """React widgets (react-select options) select on mousedown, not click — the raw
    path must dispatch the full mousedown→mouseup→click sequence."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <button id="b">Reviews</button>
            <script>
              window.__seq = [];
              var b = document.getElementById('b');
              ['mousedown', 'mouseup', 'click'].forEach(function (t) {
                b.addEventListener(t, function () { window.__seq.push(t); });
              });
            </script>
        """)
        raw = await page.evaluate(_expr(["reviews"], click=True))
        assert raw["clicked"] is True
        assert await page.evaluate("window.__seq") == ["mousedown", "mouseup", "click"]


# ------------------------------- hidden-click receipt -------------------------------


def test_receipt_matching_name_keeps_the_recall_advice():
    msg = agent_tools._hidden_click_receipt(
        "Send NPS Survey Request", "Send NPS survey request")
    assert "ALREADY CLICKED" in msg
    assert "NAME MISMATCH" not in msg
    assert "re-call find_by_text('Send NPS Survey Request', click_first=true) once" in msg


def test_receipt_trusts_a_prefix_extension_of_the_query():
    # Clicking 'Send NPS survey request' for the query "Send NPS" is the right control.
    msg = agent_tools._hidden_click_receipt("Send NPS", "Send NPS survey request")
    assert "NAME MISMATCH" not in msg


def test_receipt_warns_and_forbids_recall_on_a_wrong_named_control():
    msg = agent_tools._hidden_click_receipt("Reviews", "Add reviews (Alt+Shift+R)")
    assert "ALREADY CLICKED" in msg          # the click DID happen — that stays unmistakable
    assert "NAME MISMATCH" in msg
    assert "do NOT re-call find_by_text('Reviews')" in msg
