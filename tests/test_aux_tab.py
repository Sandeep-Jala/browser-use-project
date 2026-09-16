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
        self.routes = []                  # (pattern, handler) from page.route
        self.ops = []                     # ("route"|"goto", arg) in call order
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
        self.ops.append(("goto", url))
        self.url = url

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))
        self.ops.append(("route", pattern))

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


# ------------------------------- aux-tab ad blocking -------------------------------


async def test_open_aux_tab_registers_ad_blocking_before_goto():
    """The route must exist BEFORE the goto (the initial ad barrage is the expensive one)
    and only on the helper tab — the app tab never gets a route."""
    hs, ctx, main = _session()
    page = await hs.open_aux_tab("https://www.fakenamegenerator.com/gen-male-gd-uk.php")

    kinds = [op[0] for op in page.ops]
    assert "route" in kinds and "goto" in kinds
    assert kinds.index("route") < kinds.index("goto")
    assert page.routes and page.routes[0][1] is hybrid._abort_ad_requests
    assert main.routes == []


def test_blocked_ad_host_is_suffix_matched():
    assert hybrid._is_blocked_ad_host("https://x.doubleclick.net/instream/ad.js")
    assert hybrid._is_blocked_ad_host("https://taboola.com/widget")
    # Suffix match, not substring: a lookalike registrable domain is NOT blocked.
    assert not hybrid._is_blocked_ad_host("https://evildoubleclick.net/x")
    # The aux site itself and CMP/consent hosts are never blocked (the recording clicks
    # the consent banner — see the _AUX_BLOCKED_HOSTS comment).
    assert not hybrid._is_blocked_ad_host("https://www.fakenamegenerator.com/gen-male-gd-uk.php")
    assert not hybrid._is_blocked_ad_host("https://cdn.cookielaw.org/consent.js")
    assert not hybrid._is_blocked_ad_host("not a url")


async def test_abort_ad_requests_aborts_only_blocked_hosts():
    class FakeRoute:
        def __init__(self, url):
            self.request = SimpleNamespace(url=url)
            self.aborted = False
            self.continued = False

        async def abort(self):
            self.aborted = True

        async def continue_(self):
            self.continued = True

    ad = FakeRoute("https://securepubads.googlesyndication.com/tag.js")
    await hybrid._abort_ad_requests(ad)
    assert ad.aborted and not ad.continued

    page_req = FakeRoute("https://www.fakenamegenerator.com/gen-male-gd-uk.php")
    await hybrid._abort_ad_requests(page_req)
    assert page_req.continued and not page_req.aborted


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


# --------------------------- adopting an app-opened tab ---------------------------
# Run 20260824_163152: subtask 3 clicked the app's external-link button, did the whole
# OTP handshake in the tab it opened, and reported "Proceed Securely submitted
# successfully in the new tab" — then close_extra_tabs swept that tab as a misclick
# popup on the way out of the segment, and subtask 4 (the edits that must happen INSIDE
# the portal) had nowhere to run. The tab carries a per-request signed URL, so the
# wording cannot name it and `tab_url` is never set.

_ANNOUNCES = ("Now click on the external link button next to the ref. no, A new tab will "
              "be open click Already have an OTP, enter the OTP got it from the previous "
              "step and click proceed Securely.")
_SILENT = ("Now click the + button next to payment, enter 4000 in the amount field, "
           "click Save. Close this tab.")


def _sub(template_prompt, tab_url=None):
    return SimpleNamespace(template_prompt=template_prompt, tab_url=tab_url,
                           instantiated_prompt=template_prompt, index=3)


def test_announces_new_tab_matches_the_opener_only():
    """The URL-free half of the aux test, against the live part2 slices."""
    from automation.pipeline.decompose import announces_new_tab, _implied_tab_url

    assert announces_new_tab(_ANNOUNCES) is True
    assert _implied_tab_url(_ANNOUNCES) is None     # the APP supplies the address
    assert announces_new_tab(_SILENT) is False
    assert announces_new_tab("Go to Data Request and click Get OTP, note the OTP.") is False


async def test_adopted_tab_survives_the_sweep_and_becomes_current():
    hs, ctx, main = _session()
    portal = FakePage("https://test.actingoffice.com/employeeapproval/abc123", ctx)
    ctx.pages.append(portal)

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))
    await hs.close_extra_tabs()

    assert not portal.closed and not main.closed
    assert hs.current_page() is portal          # the next subtask runs INSIDE it
    # browser-use focus follows, or the next segment acts on (and records) the app tab.
    assert [e.target_id for e in hs.session.event_bus.events] == [portal.tid]


async def test_two_extra_tabs_are_ambiguous_and_none_is_adopted():
    """One of them IS a misclick and nothing here can say which — today's sweep stands."""
    hs, ctx, main = _session()
    portal = FakePage("https://test.actingoffice.com/employeeapproval/abc123", ctx)
    junk = FakePage("https://ads.example.com", ctx)
    ctx.pages.extend([portal, junk])

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))
    await hs.close_extra_tabs()

    assert hs._aux_page is None
    assert portal.closed and junk.closed
    assert hs.current_page() is main


async def test_silent_subtask_still_gets_its_popup_swept():
    """The invariant this must not break: a tab nobody announced is a misclick."""
    hs, ctx, main = _session()
    junk = FakePage("https://popup.example.com", ctx)
    ctx.pages.append(junk)

    await hs.adopt_announced_tab(_sub(_SILENT))
    await hs.close_extra_tabs()

    assert junk.closed and hs._aux_page is None
    assert hs.current_page() is main


async def test_adoption_never_displaces_a_declared_helper_tab():
    hs, ctx, main = _session()
    aux = await hs.open_aux_tab("https://duckduckgo.com")
    stray = FakePage("https://popup.example.com", ctx)
    ctx.pages.append(stray)

    await hs.adopt_announced_tab(_sub(_ANNOUNCES, tab_url="https://duckduckgo.com"))

    assert hs._aux_page is aux
    await hs.close_extra_tabs()
    assert stray.closed and not aux.closed


async def test_closing_the_adopted_tab_hands_the_run_back_to_main():
    """The task's own last instruction is "Close this tab." — after that current_page()
    must fall back to the pinned app page rather than a dead handle."""
    hs, ctx, main = _session()
    portal = FakePage("https://test.actingoffice.com/employeeapproval/abc123", ctx)
    ctx.pages.append(portal)
    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    await portal.close()

    assert hs.current_page() is main
    await hs.close_extra_tabs()                 # a stale _aux_page ref stays harmless
    assert not main.closed


# ------------- which tab: the agent's focus, not the tab count (2026-08-26) -------------
# Run 20260825_105115 seg 3: the OTP handshake succeeded in the portal tab, but a replay's
# force-retry had opened that SAME tab twice, so two extras existed, `len(extras) != 1`
# adopted nothing, close_extra_tabs took both, and the postcondition gate then measured the
# app tab and failed a segment whose work had landed. Two tabs to one destination are not
# ambiguous, and the agent had explicitly switched into the one it finished in — so the
# question "which tab" is answered by the agent's focus, with the count rule kept only as
# the fallback for when focus cannot be resolved.


def _focus(hs, page):
    """Point the fake browser-use session's agent focus at `page`."""
    hs.session.get_focused_target = lambda: SimpleNamespace(target_id=page.tid)


async def test_two_tabs_to_the_same_url_adopt_the_focused_one():
    hs, ctx, main = _session()
    first = FakePage("https://test.actingoffice.com/links/10/c/a/r/b/calcdatarequest", ctx)
    second = FakePage("https://test.actingoffice.com/links/10/c/a/r/b/calcdatarequest", ctx)
    ctx.pages += [first, second]
    _focus(hs, second)                     # the tab the agent actually finished in

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    assert hs._aux_page is second
    assert hs.current_page() is second     # so every page-dependent measurement follows
    await hs.close_extra_tabs()
    assert not second.closed and first.closed


async def test_unresolvable_focus_with_two_extras_still_adopts_none():
    """The count rule survives as the fallback: with no focus to ask and two genuinely
    different tabs, one of them may be a misclick and nothing here can say which."""
    hs, ctx, main = _session()
    a = FakePage("https://portal.example.com/one", ctx)
    b = FakePage("https://elsewhere.example.com/two", ctx)
    ctx.pages += [a, b]
    assert not hasattr(hs.session, "get_focused_target")   # nothing to ask

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    assert hs._aux_page is None


async def test_unresolvable_focus_with_one_extra_adopts_it():
    hs, ctx, main = _session()
    only = FakePage("https://portal.example.com/one", ctx)
    ctx.pages.append(only)

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    assert hs._aux_page is only


async def test_focus_on_a_tab_is_ignored_when_the_subtask_announced_nothing():
    """The wording gate still leads: a tab the task never mentioned is a misclick popup,
    however firmly the agent is focused on it."""
    hs, ctx, main = _session()
    popup = FakePage("https://ads.example.com/popup", ctx)
    ctx.pages.append(popup)
    _focus(hs, popup)

    await hs.adopt_announced_tab(_sub(_SILENT))

    assert hs._aux_page is None
    await hs.close_extra_tabs()
    assert popup.closed


async def test_focus_resolution_failure_degrades_to_the_count_rule():
    """A raising focus probe must not break adoption."""
    hs, ctx, main = _session()
    only = FakePage("https://portal.example.com/one", ctx)
    ctx.pages.append(only)

    def _boom():
        raise RuntimeError("session manager gone")
    hs.session.get_focused_target = _boom

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    assert hs._aux_page is only


async def test_no_adoption_when_the_pinned_main_page_has_died():
    """current_page() re-pins a survivor as MAIN; adopting that same page as AUX would
    give one tab both roles."""
    hs, ctx, main = _session()
    await main.close()
    survivor = FakePage("https://app.example.com/books/x", ctx)
    ctx.pages.append(survivor)

    await hs.adopt_announced_tab(_sub(_ANNOUNCES))

    assert hs._aux_page is None
    assert hs.current_page() is survivor        # re-pinned as MAIN, not adopted as AUX
    assert hs._main_page is survivor
