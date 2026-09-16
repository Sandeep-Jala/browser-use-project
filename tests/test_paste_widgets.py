"""The paste ladder against REAL widgets in a real browser.

Every rung of _paste_into exists for a widget shape that defeats the others, and a stub
cannot tell them apart — the stub version of these tests passed while the ladder had the
rungs in an order that could not fill a plain auto-advancing OTP box. Measured here:

  onPaste handler          -> rung 1 (synthetic ClipboardEvent)
  auto-advance, no onPaste -> rung 2 (keystrokes)
  isTrusted-only onPaste   -> rung 3 (Chrome's own paste command)
  none of the above        -> honest refusal, never a false pass

Rung 3 needs a SECURE origin: navigator.clipboard is undefined on about:blank, so that
case is served over http://127.0.0.1 (a secure context) with the clipboard permission
granted — which is what browser-use's own profile grants the live run.

Harness note: every case gets a FRESH page. page.set_content reuses the JS realm, so a
second document whose script re-declares a top-level binding throws SyntaxError, silently
attaches no listeners, and the widget then behaves like an inert one.
"""
import asyncio
import functools
import http.server
import pathlib
import socketserver
import threading

import pytest
from playwright.async_api import async_playwright

from automation.pipeline.script_compile import _paste_into
from tests.test_heal_promotion import _launch

OTP = "502956"
BOXES = "\n".join(f'<input aria-label="Please enter OTP character {n}" maxlength="1">'
                  for n in range(1, 7))
BOX1 = '[aria-label="Please enter OTP character 1"]'

_TEMPLATE = """<div>%s</div>
<script>
(function () {
  var boxes = Array.prototype.slice.call(document.querySelectorAll('input'));
  boxes.forEach(function (b, i) {
    if (PASTE_ON) {
      b.addEventListener('paste', function (e) {
        if (TRUSTED_ONLY && !e.isTrusted) { return; }
        e.preventDefault();
        var t = (e.clipboardData || window.clipboardData).getData('text');
        t.split('').slice(0, boxes.length - i).forEach(function (c, k) {
          boxes[i + k].value = c;
        });
      });
    }
    if (ADVANCE_ON) {
      b.addEventListener('input', function () {
        if (b.value && boxes[i + 1]) { boxes[i + 1].focus(); }
      });
    }
  });
})();
</script>""" % BOXES


def widget(*, paste=True, advance=True, trusted_only=False):
    return (_TEMPLATE.replace("PASTE_ON", "true" if paste else "false")
                     .replace("ADVANCE_ON", "true" if advance else "false")
                     .replace("TRUSTED_ONLY", "true" if trusted_only else "false"))


async def _run(html, selector=BOX1, *, url=None, clipboard=False):
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            context = await browser.new_context()
            if clipboard:
                await context.grant_permissions(["clipboard-read", "clipboard-write"])
            page = await context.new_page()
            if url:
                await page.goto(url)
            else:
                await page.set_content(html)
            took, shows = await _paste_into(page, page.locator(selector), OTP)
            boxes = await page.eval_on_selector_all("input", "ns => ns.map(n => n.value)")
            return took, shows, boxes
        finally:
            await browser.close()


def _serve(tmp_path, html):
    (tmp_path / "w.html").write_text(html)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_a):
            pass

    httpd = socketserver.TCPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/w.html"


def test_onpaste_widget_is_filled_by_the_synthetic_event():
    took, shows, boxes = asyncio.run(_run(widget()))
    assert (took, shows) == (True, OTP)
    assert boxes == list(OTP)


def test_widget_with_no_paste_handler_is_filled_by_keystrokes():
    # The common shape: each box advances focus as you type. Rung 1 leaves it untouched.
    took, shows, boxes = asyncio.run(_run(widget(paste=False)))
    assert (took, shows) == (True, OTP)
    assert boxes == list(OTP)


def test_paste_only_widget_that_cannot_be_typed_into_is_still_filled():
    took, _shows, boxes = asyncio.run(_run(widget(advance=False)))
    assert took is True and boxes == list(OTP)


def test_istrusted_only_widget_needs_the_browser_paste_command(tmp_path):
    httpd, url = _serve(tmp_path, widget(advance=False, trusted_only=True))
    try:
        took, _shows, boxes = asyncio.run(
            _run(None, url=url, clipboard=True))
    finally:
        httpd.shutdown()
    # Neither the synthetic event (rejected) nor typing (no auto-advance) can fill this.
    assert took is True and boxes == list(OTP)


def test_an_inert_widget_refuses_instead_of_reporting_a_partial_fill():
    took, shows, _boxes = asyncio.run(_run(widget(paste=False, advance=False)))
    # Box 1 takes the first character and the rest goes nowhere. That is NOT a paste, and
    # calling it one would let a queued "Proceed Securely" fire on a wrong code.
    assert took is False and shows == "5"


def test_a_single_field_takes_the_whole_value():
    took, shows, boxes = asyncio.run(
        _run('<input aria-label="Code">', selector='[aria-label="Code"]'))
    assert (took, shows) == (True, OTP) and boxes == [OTP]


def test_a_readonly_field_refuses():
    took, _shows, boxes = asyncio.run(
        _run('<input aria-label="Code" readonly>', selector='[aria-label="Code"]'))
    assert took is False and boxes == [""]
