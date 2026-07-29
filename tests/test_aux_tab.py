"""Aux (helper) tab support: HybridSession's explicit page pinning, the open/close/focus
lifecycle of the per-subtask helper tab, and the aux-aware gate normalizer.

The Playwright surface is faked with in-memory doubles; browser-use focus alignment is
asserted through a recording event bus (the real SwitchTabEvent class, so a field rename
upstream fails here and not in production)."""
from types import SimpleNamespace

import pytest

from automation.pipeline import hybrid
from automation.pipeline.hybrid import Gate, evaluate_gate


class FakeCdp:
    def __init__(self, tid):
        self.tid = tid

    async def send(self, method):
        assert method == "Target.getTargetInfo"
        return {"targetInfo": {"targetId": self.tid}}

    async def detach(self):
        pass


class FakePage:
    _n = 0

    def __init__(self, url="about:blank", context=None):
        FakePage._n += 1
        self.tid = f"T{FakePage._n}"
        self.url = url
        self.context = context
        self.closed = False
        self.fronted = 0
        self.gotos = []
        self.evals = []
        self.goto_error = None

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True
        if self.context is not None and self in self.context.pages:
            self.context.pages.remove(self)

    async def bring_to_front(self):
        self.fronted += 1

    async def goto(self, url, **_kw):
        if self.goto_error:
            raise RuntimeError(self.goto_error)
        self.gotos.append(url)
        self.url = url

    async def evaluate(self, expr):
        self.evals.append(expr)
        # window.open spawns a sibling-tab popup in the SAME context, like real Chrome.
        if "window.open" in str(expr) and self.context is not None:
            popup = FakePage("about:blank", self.context)
            popup.goto_error = self.context.new_page_goto_error
            self.context.pages.append(popup)
            self.context._popup = popup


class FakeContext:
    def __init__(self):
        self.pages = []
        self.new_page_goto_error = None   # inherited by pages this context creates
        self.new_page_calls = 0           # the CDP path (separate window in headful Chrome)
        self.expect_page_error = None     # set to make the popup path fail
        self._popup = None

    async def new_page(self):
        self.new_page_calls += 1
        page = FakePage("about:blank", self)
        page.goto_error = self.new_page_goto_error
        self.pages.append(page)
        return page

    def expect_page(self, timeout=None):
        if self.expect_page_error:
            raise RuntimeError(self.expect_page_error)
        ctx = self

        class _Info:
            @property
            def value(self):
                async def _get():
                    assert ctx._popup is not None, "no popup appeared inside expect_page"
                    return ctx._popup
                return _get()

        class _CM:
            async def __aenter__(self):
                ctx._popup = None
                return _Info()

            async def __aexit__(self, *_exc):
                return False

        return _CM()

    async def new_cdp_session(self, page):
        return FakeCdp(page.tid)


class FakeBus:
    def __init__(self):
        self.events = []

    def dispatch(self, event):
        self.events.append(event)

        async def _ack():
            return None
        return _ack()


def _session():
    hs = hybrid.HybridSession(
        SimpleNamespace(config=SimpleNamespace(reveal_hidden_controls=False)))
    ctx = FakeContext()
    main = FakePage("https://app.example.com/books", ctx)
    ctx.pages.append(main)
    hs.pw_browser = SimpleNamespace(contexts=[ctx])
    hs.session = SimpleNamespace(event_bus=FakeBus())
    hs.collectors = []
    hs._main_page = main
    return hs, ctx, main


# ------------------------------- open / close / focus -------------------------------


async def test_open_aux_tab_creates_focuses_and_is_idempotent():
    hs, ctx, main = _session()
    page = await hs.open_aux_tab("https://duckduckgo.com")

    assert page is not main and page in ctx.pages
    # Opened BY the main page (window.open -> same-window sibling tab), never via the
    # CDP path that headful Chrome materializes as a separate window.
    assert any("window.open" in e for e in main.evals)
    assert ctx.new_page_calls == 0
    assert page.gotos == ["https://duckduckgo.com"]
    assert page.fronted == 1
    assert hs.current_page() is page                 # the aux tab wins while open
    # browser-use's agent focus was aligned to the aux tab's CDP target.
    events = hs.session.event_bus.events
    assert events and events[-1].target_id == page.tid

    again = await hs.open_aux_tab("https://duckduckgo.com")
    assert again is page
    assert page.gotos == ["https://duckduckgo.com"]  # NO re-goto: dirty state preserved
    assert len(ctx.pages) == 2


async def test_open_aux_tab_falls_back_to_new_page_when_popup_fails():
    hs, ctx, main = _session()
    ctx.expect_page_error = "popup path unavailable"
    page = await hs.open_aux_tab("https://duckduckgo.com")

    assert ctx.new_page_calls == 1                   # CDP fallback carried the subtask
    assert page in ctx.pages and page.gotos == ["https://duckduckgo.com"]
    assert hs.current_page() is page


async def test_open_aux_tab_goto_failure_closes_page_and_raises():
    hs, ctx, main = _session()
    ctx.new_page_goto_error = "net::ERR_NAME_NOT_RESOLVED"
    with pytest.raises(RuntimeError, match="ERR_NAME_NOT_RESOLVED"):
        await hs.open_aux_tab("https://bad.invalid")
    assert hs._aux_page is None
    assert ctx.pages == [main]                       # the dead page was closed
    assert hs.current_page() is main


async def test_close_aux_tab_refocuses_main_and_is_idempotent():
    hs, ctx, main = _session()
    aux = await hs.open_aux_tab("https://duckduckgo.com")

    await hs.close_aux_tab()
    assert aux.closed and hs._aux_page is None
    assert hs.current_page() is main
    assert main.fronted == 1
    assert hs.session.event_bus.events[-1].target_id == main.tid

    await hs.close_aux_tab()                          # no aux open: a clean no-op
    assert hs.current_page() is main


# ------------------------------- page pinning -------------------------------


async def test_current_page_repins_when_the_pinned_main_dies():
    hs, ctx, main = _session()
    await main.close()
    fresh = FakePage("https://app.example.com/books/x", ctx)
    ctx.pages.append(fresh)
    assert hs.current_page() is fresh                 # re-pinned via the scan fallback
    assert hs._main_page is fresh


async def test_close_extra_tabs_keeps_main_and_aux_kills_popups():
    hs, ctx, main = _session()
    aux = await hs.open_aux_tab("https://duckduckgo.com")
    junk = FakePage("https://popup.example.com", ctx)
    ctx.pages.append(junk)

    await hs.close_extra_tabs()

    assert junk.closed
    assert not main.closed and not aux.closed
    assert hs.current_page() is aux


# ------------------------------- aux-aware gate normalizer -------------------------------


async def test_end_context_gate_picks_normalizer_by_prefix(monkeypatch):
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    # Aux contexts are host-qualified (no leading "/") -> normalize_aux_context.
    gate = Gate(kind="postcondition", end_context="duckduckgo.com/")
    page = SimpleNamespace(url="https://duckduckgo.com/?q=acting+office")
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is True and detail["reached"] == "duckduckgo.com/"
    # Main contexts (leading "/") keep the origin-stripping normalizer.
    gate = Gate(kind="postcondition", end_context="/x/inputs/sales")
    page = SimpleNamespace(url="https://app/x/inputs/sales?tab=1")
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is True and detail["reached"] == "/x/inputs/sales"


# ------------------------------- extraction -> findings -------------------------------


async def test_replay_segment_folds_extracted_into_finding(monkeypatch):
    """A zero-LLM replay whose extract steps re-read the live DOM must surface the fresh
    values as the segment's finding — the whole point of caching an aux-tab subtask."""
    async def fake_execute(_skill, _page, **_kw):
        return {"executed": 2, "failed_at": None, "error": None, "log": [],
                "extracted": {"top_result_title": "Acting Office — Accounting"}}
    monkeypatch.setattr(hybrid.skills, "execute", fake_execute)

    hs, _ctx, _main = _session()
    sub = SimpleNamespace(index=0, instantiated_prompt="search and note", kind="action")
    seg = await hs.replay_segment(sub, "sid", "duckduckgo.com/", None, Gate(kind="steps"))

    assert seg.ok
    assert seg.extracted == {"top_result_title": "Acting Office — Accounting"}
    assert seg.finding == "top_result_title = Acting Office — Accounting"
    assert seg.as_dict()["extracted"] == seg.extracted


def test_history_extracts_harvests_labels_collision_safe():
    """A re-labeled SECOND fact must not clobber the first (the live loss: the address
    extract reused 'identity_block' and erased the generated name) — colliding labels
    keep both values under suffixed keys."""
    history = SimpleNamespace(history=[
        SimpleNamespace(result=[
            SimpleNamespace(metadata={"extract": {"label": "title", "value": "First"}}),
            SimpleNamespace(metadata=None),
        ]),
        SimpleNamespace(result=[
            SimpleNamespace(metadata={"extract": {"label": "title", "value": "Fresh"}}),
            SimpleNamespace(metadata={"other": 1}),
        ]),
    ])
    assert hybrid._history_extracts(history) == {"title": "First", "title_2": "Fresh"}
    assert hybrid._history_extracts(SimpleNamespace(history=[])) == {}
