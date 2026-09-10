"""A replayed click that opens a tab, and the steps that must run INSIDE it.

Motivating failure (run 20260825_105115, subtask 5d6391bcce5b8c25 — the OTP portal): the
committed skill's first step clicks an external-link button that opens a new tab, and steps
2-5 act on that tab's contents. A compiled skill runs every step against the ONE Playwright
Page it started with, and a popup is a different Page, so step 2 hunted the app tab for a
button that exists only in the new one:

    replay_error: RuntimeError: no unique candidate matched:
      xpath=/html/body/div[3]/.../button[1] -> no match

Two defects, one symptom. The other: _click_with_retry's force fallback re-fired the click
after Playwright timed out on the popup-opening link, so ONE recorded click became two live
tabs (network.json: two GET /links/... and two POST /public/handshake). Two extra tabs then
made adopt_announced_tab adopt none, and the postcondition measured the app tab — failing a
subtask whose work had actually succeeded.

Compile stamps the opener (`opens_tab`) from the recording's own state.tabs growth, so
replay follows only where the recording went; a stray ad/consent popup can never capture a
replay.
"""
import asyncio
import functools
import http.server
import json
import socketserver
import threading

from playwright.async_api import async_playwright
from test_compile_coverage import _item, _write

from automation.pipeline import script_compile as sc
from automation.skills import codegen
from automation.skills.api import SkillApi
from tests.test_heal_promotion import _launch

OPENER = '<a id="go" href="b.html" target="_blank">open the portal</a>'
LANDED = '<button id="only-in-b">Proceed Securely</button>'


def _serve(tmp_path):
    (tmp_path / "a.html").write_text(OPENER)
    (tmp_path / "b.html").write_text(LANDED)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_a):
            pass

    httpd = socketserver.TCPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/a.html"


async def _on_page(tmp_path, body):
    httpd, url = _serve(tmp_path)
    try:
        async with async_playwright() as pw:
            browser = await _launch(pw)
            try:
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(url)
                return await body(page, context)
            finally:
                await browser.close()
    finally:
        httpd.shutdown()


# ------------------------------- compile: the stamp -------------------------------


_LINK = {"node_name": "a", "attributes": {"id": "go"}, "ax_name": "open the portal"}
_BUTTON = {"node_name": "button", "attributes": {"id": "only-in-b"},
           "ax_name": "Proceed Securely"}


def _click_item(tab_count, element=None):
    """One recorded click, with `tab_count` tabs open when the item STARTED.

    Distinct elements per item on purpose: _push_step collapses two ADJACENT clicks on the
    same target as a slow-app retry, which is also what the real trace looks like — the
    opener is a link, the step after it is a button inside the tab it opened."""
    item = _item({"click": {"index": 7}}, element=element or _LINK)
    item["state"]["tabs"] = [{"target_id": f"T{i}"} for i in range(tab_count)]
    return item


def test_the_click_that_grew_the_tab_list_is_stamped(tmp_path):
    steps = sc.compile_recording(
        _write(tmp_path, [_click_item(1), _click_item(2, _BUTTON)]),
        emit_start_goto=False)
    assert steps[0]["opens_tab"] is True
    # The click INSIDE the new tab is not an opener; only the growth point is stamped.
    assert "opens_tab" not in steps[1]


def test_no_tab_growth_stamps_nothing(tmp_path):
    steps = sc.compile_recording(
        _write(tmp_path, [_click_item(1), _click_item(1, _BUTTON)]),
        emit_start_goto=False)
    assert steps and all("opens_tab" not in s for s in steps)


def test_the_real_otp_recording_stamps_its_external_link_click():
    # Fixture from the live recording rather than a hand-written belief about the shape.
    # The OTP slices were reworded and the part2 entry (5d6391bcce5b8c25) was replaced by
    # this whole-flow entry; its history grows state.tabs 1 -> 2 across the same external
    # link, so the stamp is still read off a real trace and not off this file.
    steps = sc.compile_recording("library/07044b6a0dbf7988.recording.json",
                                 emit_start_goto=False)
    # Asserted as "exactly one stamp, and it sits at index 5" rather than as a full-length
    # list: this fixture is a LIVE library entry, so re-authoring the slice legitimately
    # changes how many steps trail the stamp (it has been 11 and is now 9) without touching
    # the thing under test.
    assert [i for i, s in enumerate(steps) if s.get("opens_tab")] == [5]
    # The one stamp is on the external link itself, and the steps after it are the ones
    # that only exist in the tab it opened.
    assert steps[5]["fingerprint"]["tag"] == "a"
    assert steps[5]["expect_text"] == "Open payroll review request as client"
    assert [steps[6]["expect_text"], steps[-1]["expect_text"]] == [
        "Already have an OTP", "Proceed Securely"]


def test_the_stamp_survives_into_the_tier_1_anchor_bundle():
    # Lost here, the code tier replays the pre-fix behaviour even with everything else
    # fixed — the anchor bundle is all api.click ever sees.
    _code, anchors = codegen.transpile("sid", [
        {"action": "click", "selectors": ["css=#go"], "opens_tab": True},
    ])
    anchor = next(iter(anchors.values()))
    assert anchor["opens_tab"] is True
    assert SkillApi(None, anchors)._step_for(next(iter(anchors)))["opens_tab"] is True


# ------------------------------- replay: follow the tab -------------------------------


def test_run_steps_follows_the_recorded_click_into_its_new_tab(tmp_path):
    async def body(page, _context):
        return await sc.run_steps(page, [
            {"action": "click", "selectors": ["css=#go"], "opens_tab": True},
            {"action": "click", "selectors": ["css=#only-in-b"]},
        ], timeout_ms=5000)

    out = asyncio.run(_on_page(tmp_path, body))
    assert out["error"] is None
    assert out["executed"] == 2


def test_without_the_stamp_the_second_step_cannot_find_its_target(tmp_path):
    # The exact regression: step 2 searches the tab it never left.
    async def body(page, _context):
        return await sc.run_steps(page, [
            {"action": "click", "selectors": ["css=#go"]},
            {"action": "click", "selectors": ["css=#only-in-b"]},
        ], timeout_ms=2000)

    out = asyncio.run(_on_page(tmp_path, body))
    assert out["error"] is not None
    assert out["failed_at"] == 1


def test_the_code_tier_rebinds_its_page_too(tmp_path):
    async def body(page, _context):
        api = SkillApi(page, {"go": {"selectors": ["css=#go"], "opens_tab": True},
                              "proceed": {"selectors": ["css=#only-in-b"]}},
                       timeout_ms=5000)
        await api.click("go")
        moved = api.page is not page
        await api.click("proceed")      # only resolvable in the new tab
        return moved, api.executed

    moved, executed = asyncio.run(_on_page(tmp_path, body))
    assert moved is True and executed == 2


# ------------------------------- one click, one tab -------------------------------


def test_a_click_that_lands_and_still_raises_is_not_re_clicked(tmp_path, monkeypatch):
    """Playwright's click can time out on a popup-opening link AFTER dispatching. Retrying
    is how one recorded click became two live tabs; the only honest question left is
    whether a tab appeared."""
    seen = {}

    async def raising_click(page, step, timeout_ms, once=False):
        seen["once"] = once
        await page.context.new_page()          # the click DID land
        raise RuntimeError("Timeout 5000ms exceeded")

    monkeypatch.setattr(sc, "_click_with_retry", raising_click)

    async def body(page, context):
        _sel, _healed, landed = await sc._click_and_follow(
            page, {"action": "click", "selectors": ["css=#go"], "opens_tab": True}, 3000)
        return landed is not page, len(context.pages)

    followed, page_count = asyncio.run(_on_page(tmp_path, body))
    assert seen["once"] is True        # the force/transient retries are off for this step
    assert followed is True
    assert page_count == 2             # exactly ONE new tab, not the two the retry made


def test_a_click_that_raises_and_opens_nothing_still_fails(tmp_path, monkeypatch):
    async def raising_click(page, step, timeout_ms, once=False):
        raise RuntimeError("no unique candidate matched")

    monkeypatch.setattr(sc, "_click_with_retry", raising_click)

    async def body(page, _context):
        try:
            await sc._click_and_follow(
                page, {"action": "click", "selectors": ["css=#go"], "opens_tab": True}, 1000)
        except RuntimeError as exc:
            return str(exc)
        return None

    assert "no unique candidate" in asyncio.run(_on_page(tmp_path, body))


def test_a_recorded_opener_that_opens_nothing_stays_put(tmp_path, monkeypatch):
    async def silent_click(page, step, timeout_ms, once=False):
        return "css=#go", None

    monkeypatch.setattr(sc, "_click_with_retry", silent_click)

    async def body(page, _context):
        _sel, _healed, landed = await sc._click_and_follow(
            page, {"action": "click", "selectors": ["css=#go"], "opens_tab": True}, 1000)
        return landed is page

    # Stay on the page we can see; the next step then fails honestly against it rather
    # than acting silently on the wrong one.
    assert asyncio.run(_on_page(tmp_path, body)) is True
