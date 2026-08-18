"""Two false negatives that turned a WORKING Net-to-Gross popup into a failed run.

Run 20260817_163352 seg 4 (video + library/39d62877f6ddd9e7.recording.failed.json): the
Feb-27 pencil opened the "Salary to take home" callout, 4000 went into Net amount, and
Calculate applied it — the row read £6086.28 gross / £4000.00 net on screen. The agent
was told otherwise twice, redid the work, typed 4000 into the grid's Jan-27 pay cell and
called fail_and_stop:

  1. find_by_text('Net amount') answered "0 matches ... not in this page's DOM" for a
     label that was ON SCREEN. 'Net amount' is a LABEL, so RAW_FIND_JS (controls only)
     can never match it — every such query falls into the miss-path scroll hunt, and
     ~22 scrollTop writes later a Fluent Callout (which dismisses on scroll) is gone.
     The proof it fired: the static-text probe, which exists precisely to find labels,
     runs AFTER the sweep and also found nothing. That dismissal is the "popup vanishes"
     the user sees; the re-click that follows dismisses it again.
  2. The Calculate click received "the dialog is STILL OPEN ... likely did NOT go
     through" — a save/submit failure claim about a CLIENT-SIDE calculator that fires no
     write at all, contradicting both the page and the task wording.

A popup owns the screen: its content is never behind a page scroll, so the hunt has
nothing to win there and a dismissal to lose."""
import pytest
from test_agent_tools import _FakeBrowserSession, _FakeDomNode, _dialog_states, \
    _fake_builtin_click, _registered_action

from automation.pipeline import agent_tools


# Distinctive markers of each JS the miss path evaluates, so the stub can answer per
# probe without depending on whitespace.
_MARKERS = {"DOCLICK": "raw_find", "SKIP": "raw_text", "FRAC": "scroll_containers",
            "scrollTop = 0": "scroll_tops", "ms-Callout": "callout_probe"}


def _which(expr: str) -> str:
    for marker, name in _MARKERS.items():
        if marker in expr:
            return name
    return "unknown"


def _stub_eval(monkeypatch, *, popup_open: int, static_hit: bool):
    """Record every probe the miss path runs. RAW_FIND always misses (the live case:
    'Net amount' is a label, not a control)."""
    seen: list[str] = []

    async def fake_eval(_session, expr, **_kw):
        name = _which(expr)
        seen.append(name)
        if name == "callout_probe":
            return {"open": popup_open}
        if name == "raw_text":
            return ({"count": 1, "name": "Net amount"} if static_hit else {"count": 0})
        if name in ("scroll_containers", "scroll_tops"):
            return 1                      # something scrolled — the hunt would go on
        return {"count": 0}               # raw_find: no control matches

    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)
    return seen


async def _find(query="Net amount"):
    fn, _pm = _registered_action("find_by_text")
    return await fn(text=query, click_first=False,
                    browser_session=_FakeBrowserSession({}))


@pytest.mark.asyncio
async def test_miss_path_does_not_scroll_while_a_popup_is_open(monkeypatch):
    seen = _stub_eval(monkeypatch, popup_open=1, static_hit=True)

    res = await _find()

    assert "scroll_containers" not in seen and "scroll_tops" not in seen
    # ...and the label the sweep used to destroy is reported for what it is.
    assert "STATIC text" in res.extracted_content


@pytest.mark.asyncio
async def test_miss_path_still_hunts_when_no_popup_is_open(monkeypatch):
    # The panel hunt (Data Request's employee list, runs 20260814_105247 /
    # 20260817_133135) must keep working: no popup, no guard.
    seen = _stub_eval(monkeypatch, popup_open=0, static_hit=False)

    await _find("Harris Duncan")

    assert "scroll_containers" in seen and "scroll_tops" in seen


@pytest.mark.asyncio
async def test_the_guard_asks_only_about_callouts_not_panels_or_modals(monkeypatch):
    """The probe must not count Panels/Modals. The Add Data Request side panel is a
    fixed [role=dialog] surface whose employee list ONLY container scrolling reveals
    (runs 20260814_105247 / 20260817_110501 / 20260817_133135) — a guard that treated it
    as a popup would delete the hunt those runs exist to protect."""
    seen: list[str] = []

    async def fake_eval(_session, expr, **_kw):
        seen.append(expr)
        return {"open": 0}

    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)
    await agent_tools._callout_open(object())

    probe = next(e for e in seen if "getBoundingClientRect" in e)
    assert ".ms-Callout" in probe
    for panel in ("ms-Panel", "ms-Modal", 'role="dialog"', "fluent-default-layer-host"):
        assert panel not in probe


@pytest.mark.asyncio
async def test_absence_receipt_names_the_open_popup_instead_of_sending_the_agent_scrolling(
        monkeypatch):
    _stub_eval(monkeypatch, popup_open=1, static_hit=False)

    msg = (await _find()).extracted_content

    assert "popup" in msg.lower()
    assert "capped_scroll" not in msg          # scrolling is exactly the wrong advice
    # The re-click that dismissed the callout four times in the live run.
    assert "dismiss" in msg.lower()


@pytest.mark.asyncio
async def test_still_open_without_any_write_does_not_allege_failure(monkeypatch):
    """The Calculate receipt. Network was WATCHED and nothing fired: that is what a
    client-side commit looks like, not evidence it failed."""
    from types import SimpleNamespace
    from test_network_verify import _FakeLiveNetwork, _speed

    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _FakeLiveNetwork([]))

    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Calculate")}))

    msg = res.extracted_content
    assert "likely did NOT go through" not in msg
    assert "STILL OPEN" in msg                 # the observation itself stays
    assert "READ THE PAGE" in msg              # verify by state, then decide
    assert "dismiss" in msg.lower()            # re-clicking the opener closes the popup


@pytest.mark.asyncio
async def test_still_open_with_network_unknown_keeps_the_validation_warning(monkeypatch):
    """_LIVE_NETWORK None means writes were never watched — nothing was learned, so the
    original pessimism must survive (run 20260807_123259's unsaved employee)."""
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)

    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert "STILL OPEN" in res.extracted_content
    assert "validation" in res.extracted_content


"""No movement at all while a callout is open (user report, 2026-08-18): "if you click
on it and then scroll or do something, the popup vanishes".

The 08-17 fix gated ONE of six ways the page can move — find_by_text's miss-path hunt.
These cover the rest: the two scroll tools we register, the two browser-use built-ins
(`scroll`, and `find_text`, whose whole job is "Scroll to text"), PageUp/PageDown through
send_keys, and RAW_FIND_JS's scrollIntoView before a click. A refusal rides the ERROR
channel so the rest of a batched step stops — a queued click must not fire against a
popup the refused scroll would have closed."""

_SCROLLERS = ("scroll", "find_text", "scroll_panels")


# Every registered action is keyword-only and takes its own params model.
_PARAMS = {"scroll": {"down": True, "pages": 1.0}, "find_text": {"text": "anything"},
           "scroll_panels": {"pages": 0.8}}


def _call(name):
    fn, pm = _registered_action(name)
    return fn, {"params": pm(**_PARAMS[name]),
                "browser_session": _FakeBrowserSession({})}


@pytest.mark.parametrize("name", _SCROLLERS)
@pytest.mark.asyncio
async def test_every_scroller_refuses_while_a_popup_is_open(monkeypatch, name):
    seen = _stub_eval(monkeypatch, popup_open=1, static_hit=False)
    fn, kw = _call(name)

    res = await fn(**kw)

    assert res.error, f"{name} must refuse on the ERROR channel"
    low = res.error.lower()
    assert "popup" in low and "dismiss" in low
    assert "escape" in low                      # the way OUT, or the agent deadlocks
    assert (res.metadata or {}).get("no_scroll") is True
    # Nothing moved: the probe is the only JS this call is allowed to run.
    assert seen == ["callout_probe"], seen


@pytest.mark.parametrize("name", ("scroll_panels",))
@pytest.mark.asyncio
async def test_our_scrollers_still_scroll_when_no_popup_is_open(monkeypatch, name):
    # The Data Request panel hunt (runs 20260814_105247 / 20260817_133135) must survive:
    # _callout_open counts .ms-Callout only, so panels never trip this guard.
    seen = _stub_eval(monkeypatch, popup_open=0, static_hit=False)
    fn, kw = _call(name)

    res = await fn(**kw)

    assert not res.error
    assert any(s in seen for s in ("scroll_containers", "unknown")), seen


@pytest.mark.asyncio
async def test_send_keys_refuses_only_the_page_scrolling_keys(monkeypatch):
    """PageUp/PageDown scroll the page. Escape, Tab, Enter, arrows and Space do NOT
    reliably — they are how the agent closes the popup, moves between its fields and
    navigates a combobox INSIDE it. Blocking those would trap the agent in the popup it
    was told to close."""
    fn, pm = _registered_action("send_keys")
    _stub_eval(monkeypatch, popup_open=1, static_hit=False)

    refused = await fn(params=pm(keys="PageDown"),
                       browser_session=_FakeBrowserSession({}))
    assert refused.error and (refused.metadata or {}).get("no_scroll") is True

    for allowed in ("Escape", "Tab", "ArrowDown", "Enter"):
        res = await fn(params=pm(keys=allowed),
                       browser_session=_FakeBrowserSession({}))
        assert not (res.error or "").startswith("REFUSED"), f"{allowed} must pass through"


async def test_raw_find_click_does_not_scroll_while_a_callout_is_open():
    """The sixth mover: RAW_FIND_JS scrollIntoView()s its winner before clicking. With a
    callout on screen that scroll is both fatal and pointless — the click below is
    DISPATCHED, not aimed, so it lands wherever the element sits."""
    import json

    from playwright.async_api import async_playwright

    from automation.pipeline import script_compile as sc
    from tests.test_heal_promotion import _launch

    page_html = """
        <div class="ms-Callout" style="position:fixed;top:0;left:0;width:200px;
             height:100px">Salary to take home</div>
        <div style="height:4000px"></div>
        <button onclick="window.__hit=1">Calculate</button>
    """
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(page_html)
        raw = await page.evaluate(sc.RAW_FIND_JS % (json.dumps(["calculate"]), "true"))

        assert raw["clicked"] is True
        assert await page.evaluate("window.__hit") == 1   # the click still landed
        assert await page.evaluate("window.scrollY") == 0  # ...without moving the page

        # Without a callout the discovery scroll must still happen (the Reviews "View
        # all" icon depends on it).
        await page.set_content(page_html.replace("ms-Callout", "ms-NotACallout"))
        await page.evaluate(sc.RAW_FIND_JS % (json.dumps(["calculate"]), "true"))
        assert await page.evaluate("window.scrollY") > 0


def test_a_refused_scroll_never_compiles_into_a_replay_step(tmp_path):
    """A refusal moved nothing. Compiling it would bake the page-scroll that dismisses
    the popup into every future replay — the phantom-action rule already applied to
    no_click probes and no_fill refusals."""
    from test_compile_coverage import _item, _write

    from automation.pipeline.script_compile import compile_recording

    refused = [{"error": "REFUSED", "metadata": {"no_scroll": True}}]
    history = [
        # capped_scroll is GONE from the tool roster (stock `scroll` does the job),
        # but compile still handles the name so pre-removal recordings replay.
        _item({"capped_scroll": {"down": True, "pages": 0.2}}, result=refused),
        _item({"scroll": {"down": True, "num_pages": 1.0}}, result=refused),
        _item({"scroll_panels": {"pages": 0.8}}, result=refused),
        _item({"scroll_panels": {"pages": 0.8}}),          # the one that actually moved
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert steps == [{"action": "scroll_panels", "pages": 0.8}]
