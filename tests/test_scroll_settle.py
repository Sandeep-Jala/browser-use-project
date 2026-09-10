"""A scroll you read in the same tick tells you what the page looked like BEFORE it.

Run 20260903_110957_989862 subtask 6 ("Scroll down inside the employee list itself … until
the noted employee's name comes into view"). The name WAS in the list. find_by_text swept
past it.

Not because the step was too big — 80% of the container leaves 20% overlap, so nothing is
skipped visually. Because the hunt read the DOM in the same tick as the scroll:

    moved = await _eval_js(session, _SCROLL_CONTAINERS_JS % 0.8)
    if not moved: break
    raw = await _eval_js(session, expr)          # <- no wait

Setting scrollTop fires React's scroll handler, which sets state, which mounts the newly
revealed rows — on a LATER frame. Read immediately, the query sees the PREVIOUS render
window. Twenty rounds of that walks the whole list while checking stale DOM every time.

And in a virtualized list the DOM check cannot compensate: rows outside the render window
are not in the DOM at all, so scrolling and DOM-reading have to be interleaved correctly or
the name is unfindable by either alone.

Every other scroll path in this repo already waits 400ms — `_scroll_containers`,
`_scroll_tops`, `_wheel_scroll` (all script_compile) and the `scroll_panels` tool, whose own
comment states the reason: "Virtualized rows render only after the scroll commits — give
React a beat". The one path that hunts for a NAME in a virtualized list was the one that
skipped it. So the wait now lives in a helper that scrolls and settles together, and both
callers go through it — the knowledge was in the codebase four times and still got dropped
a fifth.
"""
import asyncio

import pytest

from automation.pipeline import agent_tools
from tests.test_agent_tools import _FakeBrowserSession, _registered_action

# Distinctive fragments of the three scripts the hunt runs, so a fake _eval_js can tell
# them apart without depending on the whole body.
_SCROLL_MARK = "scrollTop = before"          # SCROLL_CONTAINERS_JS
_TOPS_MARK = "e.scrollTop = 0; moved++"      # SCROLL_TOPS_JS
_CALLOUT_MARK = "open:"                      # _CALLOUT_OPEN_JS


class _Trace:
    """Records what the hunt did, in order: scrolls, DOM queries and settles."""

    def __init__(self):
        self.calls: list[str] = []

    async def eval_js(self, _session, expr, **_kw):
        if _CALLOUT_MARK in expr and "scrollTop" not in expr:
            return {"open": 0}
        if _SCROLL_MARK in expr:
            self.calls.append("scroll")
            return 1
        if _TOPS_MARK in expr:
            self.calls.append("scroll")
            return 1
        self.calls.append("query")
        return {"count": 0}

    async def sleep(self, seconds):
        self.calls.append(f"settle:{seconds}")


@pytest.fixture
def traced(monkeypatch):
    t = _Trace()
    monkeypatch.setattr(agent_tools, "_eval_js", t.eval_js)
    monkeypatch.setattr(asyncio, "sleep", t.sleep)
    return t


async def _hunt(traced):
    """A find_by_text that misses the snapshot, so the raw-DOM scroll hunt runs."""
    fn, pm = _registered_action("find_by_text")
    return await fn(params=pm(text="Mikey Ross"), browser_session=_FakeBrowserSession({}))


# ------------------------------- the invariant -------------------------------


async def test_every_scroll_settles_before_the_next_dom_query(traced):
    """THE regression. A query that follows a scroll with no settle between them is reading
    the pre-scroll render window — which is what walked past the employee."""
    await _hunt(traced)

    assert "scroll" in traced.calls, traced.calls
    for i, call in enumerate(traced.calls):
        if call != "scroll":
            continue
        after = traced.calls[i + 1:]
        nxt = next((c for c in after if c == "query" or c.startswith("settle")), None)
        assert nxt is not None and nxt.startswith("settle"), (
            f"scroll at {i} was re-queried before settling: {traced.calls}")


async def test_the_settle_is_long_enough_to_matter(traced):
    """400ms, the same beat every other scroll path in the repo already waits."""
    await _hunt(traced)

    settles = {c for c in traced.calls if c.startswith("settle")}
    assert settles == {f"settle:{agent_tools._SCROLL_SETTLE_S}"}
    assert agent_tools._SCROLL_SETTLE_S >= 0.4


async def test_the_hunt_still_sweeps_the_whole_list(traced):
    """The settle must not cost rounds: a name near the bottom of a long virtualized list
    needs every one of them."""
    await _hunt(traced)

    assert traced.calls.count("scroll") >= agent_tools._PANEL_SCROLL_ROUNDS


async def test_a_found_row_stops_the_sweep_immediately(monkeypatch):
    """The settle is only paid for rounds actually traversed — a hit ends the loop, so a
    successful hunt does not wait 20 times."""
    t = _Trace()
    hits = {"n": 0}

    async def eval_js(_session, expr, **kw):
        if _CALLOUT_MARK in expr and "scrollTop" not in expr:
            return {"open": 0}
        if _SCROLL_MARK in expr or _TOPS_MARK in expr:
            t.calls.append("scroll")
            return 1
        t.calls.append("query")
        hits["n"] += 1
        # Miss on the snapshot pass and the reset pass, hit on the first swept round.
        return {"count": 1, "clicked": False, "name": "Mikey Ross",
                "element": {"tag": "span", "attrs": {}}} if hits["n"] >= 3 else {"count": 0}

    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)
    monkeypatch.setattr(asyncio, "sleep", t.sleep)
    await _hunt(t)

    assert t.calls.count("scroll") <= 2, t.calls


# ------------------------------- one definition, both callers -------------------------------


async def test_scroll_panels_goes_through_the_same_helper(monkeypatch):
    """It had its own inline sleep. Two copies of a rule is how the hunt came to be missing
    it; the tool and the hunt now share one."""
    seen = {"n": 0}

    async def fake(_session, fraction=0.8):
        seen["n"] += 1
        return 1

    monkeypatch.setattr(agent_tools, "_scroll_and_settle", fake)
    fn, pm = _registered_action("scroll_panels")
    await fn(params=pm(pages=0.8), browser_session=_FakeBrowserSession({}))

    assert seen["n"] == 1


async def test_the_helper_returns_the_moved_count(monkeypatch):
    """`moved == 0` is how the sweep knows the list is exhausted — the settle must not
    swallow it."""
    async def eval_js(_session, _expr, **_kw):
        return 3
    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)
    monkeypatch.setattr(asyncio, "sleep", _Trace().sleep)

    assert await agent_tools._scroll_and_settle(_FakeBrowserSession({})) == 3


async def test_the_helper_survives_an_unscrollable_page(monkeypatch):
    """Best-effort, like every other scroll path: a failure returns 0 (sweep ends) rather
    than breaking the lookup that called it."""
    async def boom(_session, _expr, **_kw):
        raise RuntimeError("no CDP")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)

    assert await agent_tools._scroll_and_settle(_FakeBrowserSession({})) == 0
