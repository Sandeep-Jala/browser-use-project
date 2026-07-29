"""Reveal-CSS injection: the installer JS (real Chromium over static markup, no app or
credentials), the config flag, and the flag-gated injection in the replay path.

The stylesheet (script_compile.REVEAL_CSS) forces the app's hover-revealed/0-size controls
visible so they enter the agent's interactive snapshot and pass replay _resolve's
visibility gate instead of depending on the RAW_FIND_JS blind-click fallback."""
from types import SimpleNamespace

from automation.config import Config
from automation.pipeline import hybrid
from automation.pipeline import script_compile as sc

from tests.test_heal_promotion import _launch

# The app's hover-hidden pattern in miniature: the wrapper is display:none INLINE, so the
# button has no layout until the injected !important rules override it — the exact
# mechanism the reveal depends on in production.
MARKUP = """
<div class="containerHover">
  <div class="buttons-wrapper" style="display:none">
    <button class="headerButton-12" title="Send NPS survey request"
            onclick="window.__clicked = true">send</button>
  </div>
</div>
"""

NPS_SEL = '[title="Send NPS survey request"]'


# ------------------------------- installer JS on live DOM -------------------------------


async def test_installer_reveals_hidden_control_and_is_idempotent():
    """Before injection the control is invisible (inline display:none wrapper); after, the
    style tag exists, the control is visible, and a REAL Playwright click lands — the same
    visibility property replay's _resolve requires. A second eval adds nothing."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(MARKUP)
        assert not await page.locator(NPS_SEL).is_visible()

        await page.evaluate(sc.REVEAL_CSS_JS)
        assert await page.locator(f"style#{sc.REVEAL_STYLE_ID}").count() == 1
        assert await page.locator(NPS_SEL).is_visible()
        await page.locator(NPS_SEL).click()
        assert await page.evaluate("window.__clicked") is True

        await page.evaluate(sc.REVEAL_CSS_JS)  # idempotent: still exactly one style tag
        assert await page.locator(f"style#{sc.REVEAL_STYLE_ID}").count() == 1


async def test_init_script_covers_new_documents():
    """As a context init script the installer runs at document start (document.head can be
    null); the DOMContentLoaded fallback must land the style with no manual eval — this is
    how login.py covers every document of the main context from birth. Navigation must be
    a REAL document load (data: URL): set_content rewrites the document in place without
    firing init scripts, which production navigations do."""
    import urllib.parse

    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        context = await browser.new_context()
        await context.add_init_script(sc.REVEAL_CSS_JS)
        page = await context.new_page()
        await page.goto("data:text/html," + urllib.parse.quote(MARKUP))
        assert await page.locator(f"style#{sc.REVEAL_STYLE_ID}").count() == 1
        assert await page.locator(NPS_SEL).is_visible()


async def test_run_steps_resolves_revealed_control():
    """End-to-end replay proof: a plain visibility-gated `click` step (what recordings
    authored with the reveal active compile to) resolves and clicks the once-hidden
    control after injection."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(MARKUP)
        await page.evaluate(sc.REVEAL_CSS_JS)

        out = await sc.run_steps(page, [{"action": "click", "selectors": [f"css={NPS_SEL}"]}])
        assert out["executed"] == 1 and out["failed_at"] is None
        assert await page.evaluate("window.__clicked") is True


# ------------------------------- config flag -------------------------------


def test_flag_defaults_on_and_parses_off(monkeypatch, tmp_path):
    # Point from_env at an EMPTY env file: load_dotenv persists the repo's real .env into
    # os.environ across the pytest process, so delenv + explicit path is the robust form.
    empty = tmp_path / "empty.env"
    empty.write_text("")
    monkeypatch.delenv("REVEAL_HIDDEN_CONTROLS", raising=False)
    assert Config.from_env(empty).reveal_hidden_controls is True
    monkeypatch.setenv("REVEAL_HIDDEN_CONTROLS", "false")
    assert Config.from_env(empty).reveal_hidden_controls is False


# ------------------------------- replay-path gating -------------------------------


class _FakePage:
    url = "https://app.example.com/clients"

    def __init__(self):
        self.evals: list[str] = []

    def is_closed(self) -> bool:
        return False

    async def evaluate(self, expr):
        self.evals.append(expr)


def _session(flag: bool) -> tuple[hybrid.HybridSession, _FakePage]:
    runner = SimpleNamespace(config=SimpleNamespace(reveal_hidden_controls=flag))
    hs = hybrid.HybridSession(runner)
    page = _FakePage()
    hs.pw_browser = SimpleNamespace(contexts=[SimpleNamespace(pages=[page])])
    hs.collectors = []
    return hs, page


def _sub():
    return SimpleNamespace(index=0, instantiated_prompt="go to reviews", kind="action")


async def _fake_execute(_skill, _page, **_kw):
    return {"executed": 1, "failed_at": None, "error": None, "log": []}


async def test_replay_segment_injects_reveal_css_when_flag_on(monkeypatch):
    monkeypatch.setattr(hybrid.skills, "execute", _fake_execute)
    hs, page = _session(True)
    seg = await hs.replay_segment(_sub(), "sid", "/clients", None, hybrid.Gate(kind="steps"))
    assert page.evals == [sc.REVEAL_CSS_JS]
    assert seg.ok


async def test_replay_segment_skips_reveal_css_when_flag_off(monkeypatch):
    monkeypatch.setattr(hybrid.skills, "execute", _fake_execute)
    hs, page = _session(False)
    seg = await hs.replay_segment(_sub(), "sid", "/clients", None, hybrid.Gate(kind="steps"))
    assert page.evals == []
    assert seg.ok
