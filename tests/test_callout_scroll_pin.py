"""Callout scroll pin: the injected JS driven against real Chromium (no app, no credentials).

A Fluent Callout dismisses itself when anything OUTSIDE it scrolls. Every earlier guard
chased one SOURCE of that scroll — our scroll tools (_refuse_if_callout), our text hunt
(2026-08-17), browser-use's nudge (the 2026-08-20 pre-scroll band, which moved the page on
purpose and so was still movement). None held, because the killing scroll is usually not
ours.

The mechanism, in the user's own words (2026-08-24): the page normally never moves on a
click. It moves when the target sits PARTLY OUTSIDE the viewport bounds but is still visible
and clickable — the click must scroll it into view to reach it. The popup opens, that scroll
lands, and the callout vanishes taking the Net amount box with it. Full screen only helps by
accident, by putting the pencil fully inside the bounds.

So these tests assert the invariant instead of chasing sources: while a callout is open the
page does not move, and the dismisser never even sees the event. Every dismisser here is
registered AFTER the pin, which is the real ordering — Fluent attaches its scroll listener
when the callout opens, and the pin is installed at document start.
"""
import urllib.parse

from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch

# A page taller than the viewport, a Fluent-shaped callout, and a target sitting ON the
# bottom edge — the Feb-27 pencil in miniature. The dismisser mimics a real Callout: it
# ignores scrolls originating INSIDE itself (a Callout does not self-dismiss on its own
# scrollable content) and hides rather than detaches, so a test can still read inner state.
MARKUP = """
<style>
  body { margin: 0 }
  #tall { height: 4000px }
  .ms-Callout { position: fixed; right: 20px; top: 200px; width: 220px; height: 160px;
                background: #eef }
  #inner { height: 60px; overflow: auto }
  #pencil { position: absolute; left: 10px; height: 24px }
</style>
<div id="tall">page</div>
<button id="pencil">pencil</button>
<script>
  // The pencil straddles the fold: visible and clickable, but not FULLY in bounds, so any
  // driver that wants it in view must scroll the page to reach it.
  document.getElementById('pencil').style.top = (window.innerHeight - 8) + 'px';

  window.__openCallout = function () {
    var c = document.createElement('div');
    c.className = 'ms-Callout';
    c.id = 'callout';
    c.innerHTML = 'Salary to take home<div id=inner><div style="height:400px">tall</div></div>';
    document.body.appendChild(c);
    window.__dismissed = false;
    var dismiss = function (ev) {
      var t = ev.target;
      if (t && t.nodeType === 1 && c.contains(t)) return;   // Fluent ignores its own scrolls
      window.__dismissed = true;
      c.style.display = 'none';
    };
    window.addEventListener('scroll', dismiss, false);
    document.addEventListener('scroll', dismiss, true);
    return true;
  };

  window.__state = function () {
    var c = document.getElementById('callout');
    var S = window['%(FLAG)s'] || {};
    return { dismissed: !!window.__dismissed,
             alive: !!c && c.style.display !== 'none',
             scrollY: Math.round(window.scrollY),
             reverts: S.reverts || 0, stopped: S.stopped || 0, open: !!S.open,
             innerScrollTop: (document.getElementById('inner') || {}).scrollTop };
  };
</script>
""" % {"FLAG": sc.CALLOUT_SCROLL_PIN_FLAG}


async def _settle(page):
    """Scroll events are dispatched during the browser's next rendering step, not
    synchronously with the scroll — assert before that and BOTH the pin and the dismisser
    look like they never ran while scrollY has already moved. Two frames is the wait."""
    await page.evaluate(
        "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


async def _page(pw, *, pin: bool):
    """A real document with the pin installed as an INIT SCRIPT (document start), which is
    what login.py and HybridSession.open do — the pin's listener must be registered before
    the dismisser Fluent adds when the callout opens. Navigation is a real document load:
    set_content rewrites in place and never fires init scripts."""
    browser = await _launch(pw)
    context = await browser.new_context()
    if pin:
        await context.add_init_script(sc.CALLOUT_SCROLL_PIN_JS)
    page = await context.new_page()
    await page.goto("data:text/html," + urllib.parse.quote(MARKUP))
    await page.evaluate("window.__openCallout()")
    # Let the pin's MutationObserver see the callout and snapshot the baseline offset.
    await _settle(page)
    return page


async def test_partly_clipped_target_scrolled_into_view_keeps_the_popup():
    """THE regression. Scrolling a below-the-fold target into view while a callout is open
    is exactly what the click does when the pencil straddles the bottom edge. The pin must
    put the page back AND swallow the event, so the dismisser never fires."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=True)
        assert (await page.evaluate("window.__state()"))["open"] is True

        await page.evaluate("document.getElementById('pencil').scrollIntoView({block:'center'})")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["dismissed"] is False      # the popup never learned of the scroll
        assert state["alive"] is True
        assert state["scrollY"] == 0            # and the page did not move
        assert state["reverts"] >= 1            # ... because the pin put it back
        assert state["stopped"] >= 1


async def test_without_the_pin_the_same_scroll_kills_the_popup():
    """The control: identical page, no pin. This is the bug as it stands, and it is what
    makes the test above meaningful rather than vacuous."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=False)

        await page.evaluate("document.getElementById('pencil').scrollIntoView({block:'center'})")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["dismissed"] is True
        assert state["alive"] is False
        assert state["scrollY"] > 0             # the page moved, which is what killed it


async def test_scrolling_inside_the_popup_still_works():
    """A Callout's own scrollable content must keep scrolling — Fluent does not dismiss on
    it, and pinning it would break every popup with a list in it."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=True)

        await page.evaluate("document.getElementById('inner').scrollTop = 50")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["innerScrollTop"] == 50    # not reverted
        assert state["dismissed"] is False
        assert state["alive"] is True


async def test_page_scrolls_freely_once_the_popup_closes():
    """The pin releases: with no callout on screen the page scrolls normally, so ordinary
    navigation and the virtualized-list hunts are untouched."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=True)

        await page.evaluate("document.getElementById('callout').remove()")
        await _settle(page)
        await page.evaluate("window.scrollTo(0, 0); window.scrollBy(0, 120)")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["open"] is False
        assert state["scrollY"] == 120


async def test_pin_is_idempotent_on_one_document():
    """The per-step re-assert (agent_tools.ensure_callout_scroll_pin) evaluates the same
    script on every step; a second install must not add a second listener, or one scroll
    would be reverted twice and the counters would lie."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=True)

        assert (await page.evaluate(sc.CALLOUT_SCROLL_PIN_JS))["already"] is True
        await page.evaluate("document.getElementById('pencil').scrollIntoView({block:'center'})")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["reverts"] == 1            # one listener, one revert
        assert state["alive"] is True


async def test_panels_and_modals_are_not_pinned():
    """`.ms-Callout` ONLY. The Add Data Request side PANEL owns the employee list whose rows
    only container scrolling reveals (runs 20260814_105247 / 20260817_110501 /
    20260817_133135); a guard that counted panels would kill the hunt those runs exist to
    protect."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        page = await _page(pw, pin=True)
        await page.evaluate("document.getElementById('callout').remove()")
        await _settle(page)
        await page.evaluate(
            "var p = document.createElement('div');"
            "p.className = 'ms-Panel';"
            "p.style.cssText = 'position:fixed;top:0;width:200px;height:300px';"
            "document.body.appendChild(p);")
        await _settle(page)

        await page.evaluate("window.scrollTo(0, 0); window.scrollBy(0, 90)")
        await _settle(page)
        state = await page.evaluate("window.__state()")

        assert state["open"] is False           # a Panel is not a Callout
        assert state["scrollY"] == 90           # so the page still scrolls
