"""Heal-persistence tests: promote_healed (pure JSON rewriting) and _heal_locate's winner
capture (real Chromium over a static page — no app, no credentials)."""
import json
import os
import shutil

import pytest

from automation.pipeline import script_compile as sc


def _chromium_executable() -> str | None:
    """A Chromium binary to run the DOM tests against: whatever `playwright install` put in
    PLAYWRIGHT_BROWSERS_PATH, or a stable `chromium` symlink/binary if the exact build the
    pinned playwright version expects isn't present."""
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    for candidate in (os.path.join(root, "chromium"),):
        if root and os.path.exists(candidate):
            return os.path.realpath(candidate)
    return shutil.which("chromium") or shutil.which("chromium-browser")


async def _launch(pw):
    try:
        return await pw.chromium.launch()
    except Exception:
        exe = _chromium_executable()
        if exe is None:
            pytest.skip("no Chromium available for DOM-level heal tests")
        return await pw.chromium.launch(executable_path=exe)


def _script(tmp_path, steps):
    path = tmp_path / "task.steps.json"
    path.write_text(json.dumps(steps))
    return path


def _healed_entry(step, *, attrs=None, tag="button", role=None, text="", bounds=None):
    return {"step": step, "action": "click", "used": "healed:button#save",
            "healed": {"tag": tag, "role": role, "text": text,
                       "attrs": attrs or {}, "bounds": bounds}}


def test_promoted_selectors_prepend_and_old_ones_survive(tmp_path):
    path = _script(tmp_path, [
        {"action": "click", "selectors": ['css=[id="old-id"]', "xpath=/html/body/div[1]"],
         "fingerprint": {"tag": "button", "attrs": {"id": "old-id"}}},
    ])
    log = [_healed_entry(0, attrs={"id": "new-id", "name": "save"}, text="Save")]

    assert sc.promote_healed(path, log) == [0]

    step = json.loads(path.read_text())[0]
    # New durable candidates lead; the previous anchors remain as fallbacks.
    assert step["selectors"][0] == 'role=button[name="Save"]'
    assert 'css=[id="new-id"]' in step["selectors"]
    assert step["selectors"].index('css=[id="new-id"]') < step["selectors"].index('css=[id="old-id"]')
    assert "xpath=/html/body/div[1]" in step["selectors"]
    # Fingerprint refreshed from the winner.
    assert step["fingerprint"]["attrs"]["id"] == "new-id"
    assert step["fingerprint"]["attrs"]["name"] == "save"
    assert step["fingerprint"]["text"] == "Save"


def test_promotion_dedupes_and_caps(tmp_path):
    old = [f'css=[data-old="{i}"]' for i in range(7)] + ['css=[id="stable"]']
    path = _script(tmp_path, [{"action": "click", "selectors": old}])
    log = [_healed_entry(0, attrs={"id": "stable", "name": "n1", "title": "t1"}, text="Go")]

    sc.promote_healed(path, log)

    sels = json.loads(path.read_text())[0]["selectors"]
    assert len(sels) == sc._MAX_SELECTORS
    assert len(set(sels)) == len(sels)  # deduped
    assert sels[0] == 'role=button[name="Go"]'
    assert sels.index('css=[id="stable"]') < sels.index('css=[data-old="0"]')


def test_legacy_single_selector_step_is_normalized(tmp_path):
    path = _script(tmp_path, [{"action": "fill", "selector": 'css=[name="qty"]', "value": "5"}])
    log = [{"step": 0, "action": "fill", "used": "healed:input#TextField9",
            "healed": {"tag": "input", "attrs": {"name": "quantity"}, "text": ""}}]

    sc.promote_healed(path, log)

    step = json.loads(path.read_text())[0]
    assert "selector" not in step
    assert step["selectors"] == ['css=[name="quantity"]', 'css=[name="qty"]']


def test_winner_without_durable_anchor_refreshes_fingerprint_only(tmp_path):
    path = _script(tmp_path, [
        {"action": "click", "selectors": ["xpath=/html/body/div[2]"],
         "fingerprint": {"tag": "div", "attrs": {}}},
    ])
    # Dynamic id (3+ digit run) yields no durable selector; long text is rejected too.
    log = [_healed_entry(0, tag="div", attrs={"id": "TextField1444"}, text="x" * 70,
                         bounds={"x": 1, "y": 2, "width": 3, "height": 4})]

    assert sc.promote_healed(path, log) == [0]

    step = json.loads(path.read_text())[0]
    assert step["selectors"] == ["xpath=/html/body/div[2]"]  # unchanged
    assert step["fingerprint"]["attrs"]["id"] == "TextField1444"
    assert step["fingerprint"]["bounds"] == {"x": 1, "y": 2, "width": 3, "height": 4}
    assert "text" not in step["fingerprint"]  # >60 chars: announcement blob, not a label


def test_no_heals_leaves_file_untouched(tmp_path):
    steps = [{"action": "click", "selectors": ['css=[id="a"]']}]
    path = _script(tmp_path, steps)
    before = path.read_text()

    assert sc.promote_healed(path, [{"step": 0, "action": "click", "used": 'css=[id="a"]'}]) == []
    assert path.read_text() == before


def test_out_of_range_step_is_ignored(tmp_path):
    path = _script(tmp_path, [{"action": "click", "selectors": ['css=[id="a"]']}])
    assert sc.promote_healed(path, [_healed_entry(5, attrs={"id": "b"})]) == []


async def test_heal_locate_returns_winner_identity():
    """End-to-end heal on live DOM: every recorded anchor is stale, the fingerprint still
    finds the renamed element, and the returned winner carries its new identity."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <form>
              <label>Quantity <input id="QtyField2" name="quantity" placeholder="Qty"></label>
              <label>Price <input id="PriceField7" name="unit_price" placeholder="Price"></label>
              <button type="submit">Save</button>
            </form>
        """)
        # Recorded before the app renamed QtyField1 -> QtyField2; id anchor is stale but
        # name/placeholder still match.
        fingerprint = {"tag": "input", "attrs": {"id": "QtyField1", "name": "quantity",
                                                 "placeholder": "Qty"}}
        healed = await sc._heal_locate(page, fingerprint, editable=True)
        assert healed is not None
        loc, label, winner = healed
        assert label.startswith("healed:")
        assert winner["tag"] == "input"
        assert winner["attrs"]["id"] == "QtyField2"
        assert winner["attrs"]["name"] == "quantity"
        assert await loc.get_attribute("id") == "QtyField2"
        await browser.close()


async def test_heal_locate_refuses_ambiguous_lookalikes():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
            <input name="amount" placeholder="Amount">
            <input name="amount" placeholder="Amount">
        """)
        fingerprint = {"tag": "input", "attrs": {"name": "amount", "placeholder": "Amount"}}
        assert await sc._heal_locate(page, fingerprint, editable=True) is None
        await browser.close()
