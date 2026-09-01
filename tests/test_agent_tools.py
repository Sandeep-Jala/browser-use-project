"""agent_tools: descendant icon-hint harvesting.

The failure that motivated this: Fluent UI icon buttons (`<button class="ms-Button--icon">`)
carry their meaning only in a child glyph, which browser-use serializes away — so the send,
edit, and delete icons all reach the agent as a nameless `<button/>`. `_descendant_icon_hints`
recovers that meaning from the child so nameless icon buttons become findable."""

import pathlib

from automation.pipeline import agent_tools
from automation.pipeline.agent_tools import (
    _descendant_icon_hints,
    ensure_callout_scroll_pin,
    ensure_reveal_css,
    read_new_notifications,
)


class Node:
    """Minimal stand-in for browser-use's EnhancedDOMTreeNode (only what the harvester reads)."""

    def __init__(self, node_name="DIV", attributes=None, node_value="", children=None):
        self.node_name = node_name
        self.attributes = attributes or {}
        self.node_value = node_value
        self.children = children or []


def _button(*children):
    return Node("BUTTON", {"class": "ms-Button ms-Button--icon", "type": "button"},
                children=list(children))


def test_fluent_data_icon_name_is_recovered():
    btn = _button(Node("I", {"data-icon-name": "MailReminder", "class": "ms-Icon"}))
    assert _descendant_icon_hints(btn) == "MailReminder"


def test_child_title_and_aria_label():
    btn = _button(Node("SPAN", {"title": "Send survey"}),
                  Node("I", {"aria-label": "envelope"}))
    hints = _descendant_icon_hints(btn).lower()
    assert "send survey" in hints and "envelope" in hints


def test_svg_title_text_and_use_href():
    svg = Node("SVG", children=[
        Node("TITLE", node_value="Send NPS survey request"),
        Node("USE", {"href": "#icon-send"}),
    ])
    hints = _descendant_icon_hints(_button(svg))
    assert "Send NPS survey request" in hints
    assert "icon-send" in hints


def test_icon_font_class_token():
    btn = _button(Node("I", {"class": "fa fa-envelope"}))
    assert _descendant_icon_hints(btn) == "envelope"
    btn2 = _button(Node("I", {"class": "ms-Icon ms-Icon--Send"}))
    assert _descendant_icon_hints(btn2) == "Send"


def test_dedup_and_empty():
    # Same hint from two children collapses to one.
    btn = _button(Node("I", {"data-icon-name": "Send"}), Node("I", {"title": "send"}))
    assert _descendant_icon_hints(btn).lower().split() == ["send"]
    # A button with no semantic children yields nothing (not a crash).
    assert _descendant_icon_hints(_button(Node("SPAN", node_value="   "))) == ""
    assert _descendant_icon_hints(Node("BUTTON")) == ""


def test_depth_is_bounded():
    # A hint buried deeper than the walk limit is not harvested (keeps the scan cheap).
    deep = Node("I", {"data-icon-name": "TooDeep"})
    for _ in range(6):
        deep = Node("DIV", children=[deep])
    assert "TooDeep" not in _descendant_icon_hints(_button(deep), depth=4)


# ------------------------------- read_new_notifications -------------------------------


async def test_notifications_strip_icon_glyphs_and_dedup(monkeypatch):
    """PUA icon glyphs are stripped, whitespace collapsed, batch de-duplicated, and an
    icon-only 'notification' drops out. (The JS observer needs a browser; the Python
    cleaning path is faked here by stubbing _eval_js.)"""
    G = "\ue7ba"  # a Fluent private-use icon glyph
    raw = [
        G + " Client has unpaid invoice.  Please add a payment link",  # icon + real text
        "\uea39",                                                      # icon only -> dropped
        "Client has unpaid invoice. Please add a payment link",         # dup of #1 cleaned
        "Saved successfully",
    ]

    async def fake_eval(_session, _expr):
        return raw
    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)

    out = await read_new_notifications(object())  # non-None session
    assert out == ["Client has unpaid invoice. Please add a payment link", "Saved successfully"]


async def test_notifications_none_session_and_bad_result(monkeypatch):
    assert await read_new_notifications(None) == []

    async def boom(_s, _e):
        raise RuntimeError("cdp down")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)
    assert await read_new_notifications(object()) == []  # never raises

    async def not_a_list(_s, _e):
        return "oops"
    monkeypatch.setattr(agent_tools, "_eval_js", not_a_list)
    assert await read_new_notifications(object()) == []


# ------------------------------- ensure_reveal_css -------------------------------


async def test_reveal_css_evals_the_shared_installer(monkeypatch):
    """The helper must inject EXACTLY the shared script_compile constant — identity, not a
    lookalike — so authoring and replay install one and the same stylesheet."""
    calls = []

    async def fake_eval(_session, expr):
        calls.append(expr)
    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)

    await ensure_reveal_css(object())
    assert calls == [agent_tools._REVEAL_CSS_JS]


async def test_reveal_css_never_raises_and_skips_none_session(monkeypatch):
    calls = []

    async def boom(_s, _e):
        calls.append(1)
        raise RuntimeError("cdp down")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)

    await ensure_reveal_css(object())  # swallowed
    assert calls == [1]
    await ensure_reveal_css(None)      # no session -> no eval
    assert calls == [1]


# --------------------------- callout scroll pin (2026-08-24) ---------------------------
# A Fluent Callout dismisses on any OUTSIDE scroll, and the scroll that kills it is usually
# not ours: a control PARTLY outside the viewport bounds is still visible and clickable, so
# the click scrolls it into view to reach it and the popup that click opened dies on that
# movement (user's own mechanism, 2026-08-24). The pin reverts the scroll AND swallows the
# event, verified in a real browser: scrollIntoView on a below-the-fold target moved the
# page to y=317, the pin put it back to 0, and a Fluent-shaped dismisser registered AFTER
# the pin never fired. In-callout scrolling stayed live (innerScrollTop 50).


async def test_scroll_pin_evals_the_shared_installer(monkeypatch):
    """Identity, not a lookalike: the pin the agent installs per step must be the same
    constant login.py and HybridSession.open init-script, or a document could carry a
    different pin than the one the run was verified with."""
    calls = []

    async def fake_eval(_session, expr):
        calls.append(expr)
    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)

    await ensure_callout_scroll_pin(object())
    assert calls == [agent_tools._CALLOUT_SCROLL_PIN_JS]


async def test_scroll_pin_never_raises_and_skips_none_session(monkeypatch):
    calls = []

    async def boom(_s, _e):
        calls.append(1)
        raise RuntimeError("cdp down")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)

    await ensure_callout_scroll_pin(object())   # swallowed: never fails a step
    assert calls == [1]
    await ensure_callout_scroll_pin(None)       # no session -> no eval
    assert calls == [1]


def test_scroll_pin_is_idempotent_and_self_contained():
    """The pin guards on a window flag so repeated installs are no-ops, probes `.ms-Callout`
    ONLY (Panels and Modals scroll freely — the Data Request employee list depends on it),
    and swallows the event as well as reverting: Fluent dismisses on the EVENT, so reverting
    alone would leave the popup already closed."""
    js = agent_tools._CALLOUT_SCROLL_PIN_JS
    assert "if (window[FLAG]) return { already: true }" in js
    assert ".ms-Callout" in js and ".ms-Panel" not in js and ".ms-Modal" not in js
    assert "stopImmediatePropagation" in js          # the load-bearing half
    assert "callout.contains(t)) return" in js       # in-popup scrolling stays live
    # Registered in the CAPTURE phase on both window and document, so it runs before the
    # dismisser Fluent attaches when the callout opens.
    assert js.count("addEventListener('scroll', onScroll, true)") == 2


def test_field_focus_never_scrolls_the_page():
    """Focusing a field the browser thinks is off-screen scrolls the page to the caret, and
    that movement dismisses an open Callout. Every acting focus() must pass preventScroll —
    the pin would revert it, but not moving is cheaper and covers documents the pin missed."""
    from automation.pipeline import script_compile as sc
    for mod in (agent_tools, sc):
        offenders = [
            (n, line.strip())
            for n, line in enumerate(pathlib.Path(mod.__file__).read_text().splitlines(), 1)
            # Comments may DISCUSS a bare focus() (agent_tools documents that react-select
            # ignores one); only real calls matter.
            if ".focus()" in line and not line.lstrip().startswith(("#", "//"))
        ]
        assert not offenders, f"{mod.__name__} bare focus() call(s): {offenders}"


# ------------------------------- extract_data -------------------------------


def _registered_action(name):
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions[name]
    return action.function, action.param_model


class _FakeDomNode:
    """Snapshot node double for _matching_nodes/_captured_element (the full
    DOMInteractedElement loader raises on it, exercising the manual-capture path)."""

    def __init__(self, text, attributes=None):
        self._text = text
        self.attributes = attributes or {}
        self.node_name = "H2"
        self.ax_node = None
        self.backend_node_id = 7
        self.children = []

    def get_all_children_text(self, max_depth=5):
        return self._text

    @property
    def xpath(self):
        raise RuntimeError("no layout")


class _FakeBrowserSession:
    def __init__(self, nodes):
        self._nodes = nodes

    async def get_browser_state_summary(self, include_screenshot=False):
        from types import SimpleNamespace

        return SimpleNamespace(dom_state=SimpleNamespace(selector_map=self._nodes),
                               url="https://duckduckgo.com")

    async def get_element_by_index(self, index):
        return self._nodes.get(index)


async def test_extract_data_captures_value_and_replayable_element_metadata():
    fn, pm = _registered_action("extract_data")
    node = _FakeDomNode("Acting Office — Cloud Accounting",
                        attributes={"data-testid": "result-title"})
    res = await fn(params=pm(text="acting office", label="Top Result Title!"),
                   browser_session=_FakeBrowserSession({3: node}))

    ext = (res.metadata or {}).get("extract")
    assert ext is not None
    assert ext["label"] == "top_result_title"           # slugified
    assert ext["value"] == "Acting Office — Cloud Accounting"
    assert ext["query"] == "acting office"
    assert ext["interacted_element"]["node_name"] == "H2"
    assert ext["interacted_element"]["attributes"] == {"data-testid": "result-title"}
    assert "top_result_title = 'Acting Office" in res.extracted_content


async def test_extract_data_no_match_returns_guidance_without_metadata(monkeypatch):
    async def no_raw(_session, _expr):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", no_raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="ghost value", label="g"),
                   browser_session=_FakeBrowserSession({}))
    assert res.metadata is None                          # compiles to NOTHING
    assert "NOT captured" in res.extracted_content


async def test_extract_data_raw_dom_fallback_for_non_interactive_text(monkeypatch):
    async def raw(_session, _expr):
        return {"count": 1, "clicked": False, "name": "Acting Office result",
                "element": {"tag": "h2", "attrs": {"id": "r1"}}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="acting office", label="t"),
                   browser_session=_FakeBrowserSession({}))
    ext = res.metadata["extract"]
    assert ext["value"] == "Acting Office result"
    assert ext["interacted_element"] == {"node_name": "h2", "attributes": {"id": "r1"},
                                         "ax_name": "Acting Office result"}


async def test_verify_download_reports_session_downloads():
    """The download ground-truth probe: NONE with an empty session, CONFIRMED with the
    downloaded files' basenames — the receipt that stops an agent from re-clicking a
    download control whose click receipt lied with a timeout."""
    from types import SimpleNamespace

    fn, _pm = _registered_action("verify_download")

    res = await fn(browser_session=SimpleNamespace(downloaded_files=[]))
    assert "NONE" in res.extracted_content

    res = await fn(browser_session=SimpleNamespace(
        downloaded_files=["/tmp/x/FOOD LIMITED_Bailey Stevenson_Forecast report_27.pdf"]))
    assert "CONFIRMED" in res.extracted_content
    assert "Forecast report_27.pdf" in res.extracted_content
    assert "do NOT click" in res.extracted_content


async def test_extract_data_defers_form_controls_to_raw_dom_finders(monkeypatch):
    """A <select> snapshot match must NOT be read via children text — that concatenates
    every option label ('Random Male Female', the observed live failure). The tool defers
    to the raw-DOM finders, whose select branch reports the SELECTED option."""
    async def raw(_session, _expr):
        return {"count": 1, "clicked": False, "name": "Male",
                "element": {"tag": "select", "attrs": {"id": "gender"},
                            "xpath": "/html/body/select"}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    node = _FakeDomNode("Random Male Female", attributes={"id": "gender"})
    node.node_name = "SELECT"
    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="male", label="gender"),
                   browser_session=_FakeBrowserSession({6: node}))
    ext = res.metadata["extract"]
    assert ext["value"] == "Male"                        # not the option blob
    assert ext["interacted_element"]["node_name"] == "select"
    assert ext["interacted_element"]["x_path"] == "/html/body/select"


async def test_extract_data_control_only_match_with_no_raw_hit_is_honest_miss(monkeypatch):
    """When the only snapshot match is a form control and the raw-DOM finders find
    nothing either (an <input>'s current value has no text presence), the tool must say
    so instead of reporting a wrong blob."""
    async def no_raw(_session, _expr):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", no_raw)

    node = _FakeDomNode("", attributes={"id": "email", "value": "a@b.com"})
    node.node_name = "INPUT"
    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="a@b.com", label="email"),
                   browser_session=_FakeBrowserSession({4: node}))
    assert res.metadata is None
    assert "NOT captured" in res.extracted_content


async def test_extract_data_falls_through_to_static_text_finder(monkeypatch):
    """The live failure this guards: a value in a bare <h3> (fakenamegenerator's generated
    identity) is invisible to the interactive snapshot AND to RAW_FIND_JS's control
    selector — extract_data must fall through to the static-text finder instead of
    reporting 'nothing matched' forever."""
    exprs = []

    async def raw(_session, expr):
        exprs.append(expr)
        if len(exprs) == 1:                       # RAW_FIND_JS: no control matches
            return {"count": 0}
        return {"count": 1, "name": "Felix MacDonald",
                "element": {"tag": "h3", "attrs": {}, "xpath": "/html/body/div/h3"}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="felix macdonald", label="generated_name"),
                   browser_session=_FakeBrowserSession({}))

    assert len(exprs) == 2
    assert exprs[0] != exprs[1]                   # second call is the TEXT finder
    ext = res.metadata["extract"]
    assert ext["label"] == "generated_name"
    assert ext["value"] == "Felix MacDonald"
    assert ext["interacted_element"]["node_name"] == "h3"
    # The positional anchor rides along — without it the compiled extract would anchor
    # ONLY on text="Felix MacDonald", dead the moment the page regenerates its value.
    assert ext["interacted_element"]["x_path"] == "/html/body/div/h3"


# ------------------------------- element capture resilience -------------------------------


def test_captured_element_manual_fallback_for_offscreen_nodes():
    """The full DOMInteractedElement loader throws for off-screen/zero-size nodes (the
    exact nodes find_by_text exists to reach); the capture must degrade to a manual
    identity carrying the fields compile anchors on — never to None."""
    from types import SimpleNamespace

    from automation.pipeline.agent_tools import _captured_element

    class _Node(SimpleNamespace):
        @property
        def xpath(self):  # traversal fails off-screen, exactly like 0.13.3
            raise RuntimeError("no layout")

    node = _Node(node_name="BUTTON", attributes={"title": "View all"},
                 ax_node=None, backend_node_id=123)
    el = _captured_element(node, "View all")
    assert el is not None
    assert el["node_name"] == "BUTTON"
    assert el["attributes"] == {"title": "View all"}
    assert el["ax_name"] == "View all"          # matched label fills in for the ax name
    assert "x_path" not in el                   # unreachable xpath is simply omitted


# --------------------- find_by_text truthfulness on unclickables ---------------------
# The live failure these guard: browser-use REFUSES a click on a native <select> by
# RETURNING {'validation_error': ...} (never raising), and find_by_text used to discard
# that result and report "clicked the single match" anyway — two phantom receipts on the
# country <select> convinced the agent its already-applied selection kept failing.


class _FakeEvent:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def __await__(self):
        async def _dispatched():
            return None
        return _dispatched().__await__()

    async def event_result(self, **_kw):
        if self._exc:
            raise self._exc
        return self._result


class _FakeClickSession(_FakeBrowserSession):
    def __init__(self, nodes, result=None, exc=None):
        super().__init__(nodes)
        from types import SimpleNamespace

        self.event_bus = SimpleNamespace(dispatch=lambda _e: _FakeEvent(result, exc))


class _FakeSelectDomNode(_FakeDomNode):
    def __init__(self):
        super().__init__("Australia Austria United Kingdom", attributes={"id": "c"})
        self.node_name = "SELECT"
        self.tag_name = "select"


async def test_find_by_text_refuses_to_click_native_select():
    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="united kingdom", click_first=True),
                   browser_session=_FakeBrowserSession({9: _FakeSelectDomNode()}))

    # error channel: multi_act stops the step's remaining queued actions on it, so
    # nothing executes on top of a click that never happened.
    assert "did NOT click" in res.error
    assert "select_dropdown(index=9" in res.error
    assert res.metadata == {"no_click": True}    # compiles to NOTHING, never a phantom click


async def test_find_by_text_refuses_to_click_file_input():
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("", attributes={"type": "file", "name": "csv upload"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="csv upload", click_first=True),
                   browser_session=_FakeBrowserSession({4: node}))

    assert "did NOT click" in res.error
    assert "upload_file" in res.error
    assert res.metadata == {"no_click": True}


async def test_find_by_text_reports_refused_click_instead_of_lying(monkeypatch):
    """A returned validation_error dict is a REFUSAL, not a click — the receipt must say
    so and must not stamp interacted_element (which would compile a dead click step)."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("Continue", attributes={"id": "go"})
    node.tag_name = "h2"
    refusal = {"validation_error": "Cannot click on <select> elements. "
                                   "Use dropdown_options(index=...) action instead."}
    res = await fn(params=pm(text="continue", click_first=True),
                   browser_session=_FakeClickSession({4: node}, result=refusal))

    assert "REFUSED" in res.error
    assert "Nothing was clicked" in res.error
    assert res.metadata == {"no_click": True}


async def test_find_by_text_refuses_self_match_click_on_text_input():
    """The live failure this guards (run 20260803_152301_114498): the agent typed the
    employee name into the WRONG react-select (value/placeholder invisible in the
    snapshot), then find_by_text(name, click_first=True) matched ONLY its own typed text
    inside that input and "clicked" it — a no-op wearing a success receipt, so the agent
    believed the employee was selected. A single match that is a text-entry field whose
    identity (aria-label/title/placeholder/name/id) does NOT carry the query must refuse."""
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("Calum Findlay",
                        attributes={"id": "react-select-2-input", "type": "text",
                                    "role": "combobox"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="Calum Findlay", click_first=True),
                   browser_session=_FakeBrowserSession({18958: node}))

    assert "did NOT click" in res.error
    assert res.metadata == {"no_click": True}    # compiles to NOTHING, never a phantom click


async def test_find_by_text_still_clicks_input_matched_by_placeholder(monkeypatch):
    """The guard must NOT block the recovery path it steers toward: clicking an input
    found by its placeholder/label (e.g. find_by_text('Select Employee')) is the
    legitimate way to focus/open a combobox."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("", attributes={"placeholder": "Select Employee", "type": "text"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="select employee", click_first=True),
                   browser_session=_FakeClickSession({4: node}, result=None))

    assert res.error is None
    assert "clicked the single match" in res.extracted_content
    assert res.metadata and "interacted_element" in res.metadata


# ------------------------- dialog-outcome click receipts -------------------------
# The live failure these guard (run 20260805_131827_339055, the duplicate-add loop): Save
# silently closes the modal and appends a row; with no receipt saying so, the agent
# re-opened the dialog and re-added the same benefit ten times. Clicks on elements inside
# a dialog now report whether the dialog closed — the missing did-it-actually-save signal.


def _dialog_states(monkeypatch, pre, post):
    """Feed _click_with_dialog_outcome its pre/post probes in order."""
    seq = [pre, post]

    async def fake_state(_session, node=None):
        return seq.pop(0) if seq else None

    monkeypatch.setattr(agent_tools, "_dialog_state", fake_state)
    monkeypatch.setattr(agent_tools, "_DIALOG_SETTLE_S", 0)


async def _fake_builtin_click(params=None, browser_session=None):
    from browser_use.agent.views import ActionResult

    return ActionResult(extracted_content='Clicked button "Save"')


async def test_click_inside_dialog_reports_dialog_closed(monkeypatch):
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert 'Clicked button "Save"' in res.extracted_content
    assert "dialog CLOSED" in res.extracted_content
    # Without an accepted write the close alone is NOT proof of acceptance.
    assert "VERIFY the record" in res.extracted_content


async def test_click_stamps_the_element_it_acted_on(monkeypatch):
    """Run 20260825_090938: browser-use fills state.interacted_element from the snapshot it
    takes AFTER the action, so the external-link click that opened the OTP tab recorded
    [null, null, null] and compile dropped the one load-bearing click of that segment
    without a word. The override stamps the PRE-click element instead."""
    from types import SimpleNamespace

    _dialog_states(monkeypatch, None, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    monkeypatch.setattr(agent_tools, "_captured_element",
                        lambda node, _label: {"node_name": "a", "ax_name": "Open review"})
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Open review")}))

    assert res.metadata["interacted_element"] == {"node_name": "a", "ax_name": "Open review"}


def test_the_click_stamp_never_clobbers_the_write_outcome(monkeypatch):
    """Both stamps ride the same metadata dict, and the click override applies them in
    sequence; a merge that replaced the dict wholesale would cost the gate its structured
    write verdict (or the compiler its element)."""
    monkeypatch.setattr(agent_tools, "_captured_element",
                        lambda node, _label: {"node_name": "button"})
    stamped = agent_tools._stamp_interacted({"row_label": "Harris Duncan"}, object())
    merged = agent_tools._with_write_outcome(stamped, {"fired": True, "accepted": True})

    assert merged["interacted_element"] == {"node_name": "button"}
    assert merged["write_outcome"] == {"fired": True, "accepted": True}
    assert merged["row_label"] == "Harris Duncan"


def test_the_stamp_is_a_no_op_without_an_element(monkeypatch):
    # None-in-None-out: an unstamped result stays byte-identical, so nothing about the
    # existing receipts changes for clicks browser-use already captured.
    monkeypatch.setattr(agent_tools, "_captured_element", lambda node, _label: None)
    assert agent_tools._stamp_interacted(None, None) is None
    assert agent_tools._stamp_interacted(None, object()) is None


async def test_a_click_that_already_names_its_element_keeps_that_one(monkeypatch):
    """find_by_text has already named the exact node it clicked — an existing stamp wins."""
    from types import SimpleNamespace
    from browser_use.agent.views import ActionResult

    async def stamped_click(params=None, browser_session=None):
        return ActionResult(extracted_content='Clicked button "Save"',
                            metadata={"interacted_element": {"node_name": "input"}})

    _dialog_states(monkeypatch, None, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    monkeypatch.setattr(agent_tools, "_captured_element",
                        lambda node, _label: {"node_name": "WRONG"})
    res = await agent_tools._click_with_dialog_outcome(
        stamped_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert res.metadata["interacted_element"] == {"node_name": "input"}


async def test_click_inside_dialog_reports_still_open(monkeypatch):
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert "STILL OPEN" in res.extracted_content
    assert "validation" in res.extracted_content


async def test_anonymous_checkbox_click_receipt_names_its_row(monkeypatch):
    # Run 20260817_124339 seg 6: the ticked row checkbox's receipt read
    # 'Clicked div role=checkbox ""' — nameless, so nothing contradicted the agent's
    # belief it had ticked Harris Duncan; the request saved for Aaran Duncan and the
    # segment false-passed. An anonymous checkbox click now names the ROW it sits in.
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    probed = []

    async def fake_row_label(_session, node):
        probed.append(node)
        return "Aaran Duncan"

    monkeypatch.setattr(agent_tools, "_row_context_label", fake_row_label)
    node = _FakeDomNode("", attributes={"role": "checkbox",
                                        "id": "row1569-40-checkbox"})

    async def builtin(params=None, browser_session=None):
        from browser_use.agent.views import ActionResult
        return ActionResult(extracted_content='Clicked div role=checkbox ""')

    res = await agent_tools._click_with_dialog_outcome(
        builtin, SimpleNamespace(index=4), _FakeBrowserSession({4: node}))

    assert probed, "row probe never ran for an anonymous checkbox"
    assert 'in row "Aaran Duncan"' in res.extracted_content


async def test_named_button_click_gets_no_row_probe(monkeypatch):
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    probed = []

    async def fake_row_label(_session, node):
        probed.append(node)
        return "SHOULD NOT APPEAR"

    monkeypatch.setattr(agent_tools, "_row_context_label", fake_row_label)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert not probed                      # named non-checkbox: no probe, no suffix
    assert "SHOULD NOT APPEAR" not in res.extracted_content


async def test_click_outside_dialog_keeps_receipt_unchanged(monkeypatch):
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert res.extracted_content == 'Clicked button "Save"'


async def test_click_error_result_gets_no_dialog_suffix(monkeypatch):
    from types import SimpleNamespace

    from browser_use.agent.views import ActionResult

    async def failing_builtin(params=None, browser_session=None):
        return ActionResult(error="element vanished")

    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": False, "open": 0})
    res = await agent_tools._click_with_dialog_outcome(
        failing_builtin, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert res.error == "element vanished"
    assert not res.extracted_content


def test_click_action_is_overridden_with_dialog_outcome_wrapper():
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions["click"]
    assert action.function.__module__ == "automation.pipeline.agent_tools"


def _stamped_dialog(monkeypatch, probe):
    """Pre-probe says the clicked node's dialog was STAMPED (pre count 1); the post poll
    then asks the stamped probe for {'present': that dialog still up, 'open': count}."""
    async def fake_pre(_session, node=None):
        return {"in_dialog": True, "open": 1, "stamped": True}

    async def fake_open(_session):
        return dict(probe)

    monkeypatch.setattr(agent_tools, "_dialog_state", fake_pre)
    monkeypatch.setattr(agent_tools, "_stamped_dialog_open", fake_open)
    monkeypatch.setattr(agent_tools, "_DIALOG_SETTLE_S", 0)
    monkeypatch.setattr(agent_tools, "_DIALOG_WRITE_SETTLE_S", 0)


class _AcceptedNet:
    def writes_since(self, t0):
        record = {"url": "https://api.app/DataRequest?yearId=27", "method": "POST",
                  "status": 200, "failed": False, "body": '{"success": true}',
                  "response_headers": {"content-type": "application/json"}}
        return [{"record": record, "started": 0.0, "settled": True}]


async def test_chained_panel_with_accepted_write_says_do_not_redo(monkeypatch):
    """The seg6 chain (run 20260807_095537): Save closed its dialog and the NEXT panel
    opened, so the count never dropped — by DOM shape alone indistinguishable from a
    re-render that ate the stamp (run 20260807_123259). The accepted write carries the
    truth: succeeded, do NOT redo, the open panel may be the follow-up."""
    from types import SimpleNamespace

    _stamped_dialog(monkeypatch, {"present": False, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _AcceptedNet())
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    msg = res.extracted_content
    assert "SUCCEEDED" in msg and "do NOT redo" in msg and "FOLLOW-UP" in msg
    assert "dialog CLOSED" not in msg


async def test_rerendered_stamp_loss_is_not_reported_closed(monkeypatch):
    """Run 20260807_123259: loan-toggle clicks re-rendered the panel, the stamp died
    with the replaced node, and 'stamp gone' was read as 'closed' — three false
    ACCEPTED receipts in a row and an unsaved employee sailed through. Stamp gone
    with the count unchanged must NOT claim closure."""
    from types import SimpleNamespace

    _stamped_dialog(monkeypatch, {"present": False, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert "dialog CLOSED" not in res.extracted_content
    assert "STILL OPEN" in res.extracted_content


async def test_closed_without_write_demands_verification(monkeypatch):
    """Run 20260807_123259's Save: the panel truly closed but NO create POST fired —
    the old unconditional 'form was ACCEPTED' advisory waved the unsaved employee
    through. A close without an accepted write now demands the agent verify the
    record exists before calling the step done."""
    from types import SimpleNamespace

    _stamped_dialog(monkeypatch, {"present": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    msg = res.extracted_content
    assert "dialog CLOSED" in msg
    assert "VERIFY the record" in msg
    assert "ACCEPTED" not in msg


async def test_closed_with_accepted_write_confirms_done(monkeypatch):
    from types import SimpleNamespace

    _stamped_dialog(monkeypatch, {"present": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _AcceptedNet())
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    msg = res.extracted_content
    assert "dialog CLOSED" in msg and "ACCEPTED" in msg and "DONE" in msg
    assert "VERIFY the record" not in msg


async def test_stamped_still_open_without_write_keeps_the_warning(monkeypatch):
    from types import SimpleNamespace

    _stamped_dialog(monkeypatch, {"present": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert "STILL OPEN" in res.extracted_content
    assert "validation" in res.extracted_content


async def test_find_by_text_click_carries_outcome_receipts(monkeypatch):
    """A Save clicked via find_by_text used to fire its POST with NO receipt (run
    20260807_095537 seg6: the unreceipted DataRequest create was re-clicked into a
    duplicate). The click branch now appends the same network+dialog suffix as the
    click override."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    _stamped_dialog(monkeypatch, {"present": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _AcceptedNet())
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("Save", attributes={"id": "btn-save-1"})
    node.node_name = "BUTTON"
    node.tag_name = "button"
    res = await fn(params=pm(text="save", click_first=True),
                   browser_session=_FakeClickSession({4: node}, result=None))

    msg = res.extracted_content
    assert "clicked the single match" in msg
    assert "this click fired POST" in msg and "DataRequest" in msg and "200" in msg
    assert "dialog CLOSED" in msg


async def test_find_by_text_miss_reports_static_text_instead_of_not_in_dom(monkeypatch):
    """A label that exists as STATIC text (the shadow-DOM modal's 'Period to', run
    20260805_123407_334719) used to get 'the element is not in this page's DOM' — a lie
    that sent the agent hunting other pages and guessing combobox indexes. The miss
    branch now probes the static-text finder and reports what actually exists."""
    async def raw(_session, expr):
        if "DOCLICK" in expr:                      # control finder: nothing control-shaped
            return {"count": 0}
        return {"count": 1, "name": "Period to",   # static-text finder: the label
                "element": {"tag": "div", "attrs": {}, "xpath": ""}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Period to"),
                   browser_session=_FakeBrowserSession({}))

    msg = res.extracted_content
    assert "not in this page's DOM" not in msg
    assert "STATIC text" in msg
    assert "Period to" in msg
    assert (res.metadata or {}).get("no_click") is True   # a probe — compiles to NOTHING


async def test_find_by_text_miss_sweeps_from_top_and_resets_after(monkeypatch):
    # Run 20260817_133135: the fallback swept DOWN-only from wherever the page stood
    # (never finding targets ABOVE it) and a failed hunt left every container parked
    # at the bottom — the user had to scroll back up by hand. The hunt must reset to
    # the top BEFORE sweeping (full coverage) and again AFTER a failed sweep.
    calls = []

    async def eval_js(_session, expr, **kw):
        if "FRAC" in expr:                        # SCROLL_CONTAINERS_JS
            calls.append("sweep")
            return 1                              # containers keep moving
        if "scrollTop = 0" in expr:               # SCROLL_TOPS_JS
            calls.append("top")
            return 2
        calls.append("find")
        return {"count": 0}                       # a true miss, every probe

    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)
    fn, pm = _registered_action("find_by_text")
    await fn(params=pm(text="ghost button"), browser_session=_FakeBrowserSession({}))

    assert calls.count("top") >= 2
    assert calls.index("top") < calls.index("sweep")          # reset BEFORE the sweep
    last_top = max(i for i, c in enumerate(calls) if c == "top")
    last_sweep = max(i for i, c in enumerate(calls) if c == "sweep")
    assert last_top > last_sweep                              # and again after failing


async def test_find_by_text_true_miss_still_says_not_in_dom(monkeypatch):
    async def raw(_session, _expr):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Ghost Section"),
                   browser_session=_FakeBrowserSession({}))

    assert "not in this page's DOM" in res.extracted_content
    assert (res.metadata or {}).get("no_click") is True


# ------------------------- select_dropdown read-back override -------------------------
# The built-in trusts the picker's self-report; on an ad-heavy page the confirmation can
# time out AFTER the selection took, and the empty failure receipt made the agent re-set
# the same option repeatedly. The override reads the element back and reports what the
# select actually shows.


class _FakeSelectSession:
    def __init__(self, result=None, exc=None):
        from types import SimpleNamespace

        # tag_name matters since the custom-combobox branch: only a native <select>
        # takes the SelectDropdownOptionEvent path these tests exercise.
        self._node = SimpleNamespace(tag_name="select")
        self.event_bus = SimpleNamespace(dispatch=lambda _e: _FakeEvent(result, exc))

    async def get_element_by_index(self, _index):
        return self._node


def _patch_select_readback(monkeypatch, state):
    monkeypatch.setattr(agent_tools, "SelectDropdownOptionEvent", lambda **kw: kw)

    async def field_handle(_session, _node):
        return ("cdp", "obj")

    async def select_state(_handle):
        return state

    monkeypatch.setattr(agent_tools, "_field_handle", field_handle)
    monkeypatch.setattr(agent_tools, "_select_state", select_state)


def test_select_dropdown_override_is_registered():
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions["select_dropdown"]
    assert action.function.__module__ == "automation.pipeline.agent_tools"


async def test_select_dropdown_timeout_is_rescued_by_readback(monkeypatch):
    _patch_select_readback(monkeypatch, ("uk", "United Kingdom"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(exc=TimeoutError("no confirmation")))

    assert res.error is None
    assert "read-back confirms" in res.extracted_content
    assert "do NOT set it again" in res.extracted_content


async def test_select_dropdown_mismatch_is_reported_truthfully(monkeypatch):
    _patch_select_readback(monkeypatch, ("au", "Australia"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(result={"success": "true"}))

    assert res.error is not None
    assert "did NOT take" in res.error
    assert "Australia" in res.error
    assert "dropdown_options(index=9)" in res.error


async def test_select_dropdown_success_echoes_readback(monkeypatch):
    _patch_select_readback(monkeypatch, ("uk", "United Kingdom"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(result={"success": "true"}))

    assert res.error is None
    assert "now reads 'United Kingdom'" in res.extracted_content



# ------------------------------- custom-combobox picks -------------------------------
# Motivating failure (payroll run 20260803_112915): react-select comboboxes have no
# native-<select> path, so the agent hand-rolled open/type/click across batched steps and
# died after 17 steps of dropdown thrashing on "Join with". select_dropdown's non-native
# branch owns the whole transaction; these tests cover its option matcher, the in-page
# JS contract (against a real react-select-like widget), and the decision ladder.

_RS_PAGE = """
<div class="field">
  <label>Frequency</label>
  <div class="rs" id="c1">
    <div class="rs-control">
      <div class="rs-value" id="v1">Monthly</div>
      <input id="react-select-3-input" role="combobox">
    </div>
  </div>
</div>
<div class="field">
  <label>Join with</label>
  <div class="rs" id="c2">
    <div class="rs-control">
      <div class="rs-value" id="v2">Select...</div>
      <input id="react-select-4-input" role="combobox">
    </div>
  </div>
</div>
<script>
  // Faithful to react-select where it matters: the menu mounts on the CONTROL's
  // mousedown, options select on the OPTION's mousedown (never on click), and the
  // menu unmounts on selection.
  document.querySelectorAll('.rs-control').forEach(function (ctl) {
    ctl.addEventListener('mousedown', function () {
      var container = ctl.parentElement;
      if (container.querySelector('.rs-menu')) return;
      var menu = document.createElement('div');
      menu.className = 'rs-menu';
      var instance = ctl.querySelector('input').id.replace('-input', '');
      ['P45', 'P46', 'Existing employee'].forEach(function (t, i) {
        var o = document.createElement('div');
        o.id = instance + '-option-' + i;
        o.textContent = t;
        o.addEventListener('mousedown', function () {
          container.querySelector('.rs-value').textContent = t;
          menu.remove();
          window.__picked = t;
        });
        menu.appendChild(o);
      });
      container.appendChild(menu);
    });
  });
</script>
"""


def _cb_expr(op, input_id, wanted=None):
    import json as _json
    return agent_tools._CB_OPS_JS % {
        "id": _json.dumps(input_id), "op": _json.dumps(op), "wanted": _json.dumps(wanted)}


def _resolve_expr():
    return "(el) => (" + agent_tools._CB_RESOLVE_JS + ").call(el)"


def test_choose_option_exact_beats_partial():
    opts = [{"id": "a", "text": "Existing employee record"},
            {"id": "b", "text": "Existing employee"}]
    assert agent_tools._choose_option(opts, "existing employee")["id"] == "b"


def test_choose_option_unique_partial_matches_padded_cards():
    # react-select option cards pad the label with detail lines (tax code, id, pay).
    opts = [{"id": "a", "text": "AM Aaran Macleod Tax Code : 1257L Cum Employee ID : 5302"},
            {"id": "b", "text": "HS Haiden Stevenson Tax Code : 1257L Cum"}]
    assert agent_tools._choose_option(opts, "Haiden Stevenson")["id"] == "b"


def test_choose_option_ambiguous_or_absent_returns_none():
    opts = [{"id": "a", "text": "Existing employee A"},
            {"id": "b", "text": "Existing employee B"}]
    assert agent_tools._choose_option(opts, "Existing employee") is None
    assert agent_tools._choose_option(opts, "New starter") is None
    assert agent_tools._choose_option([], "anything") is None


def test_choose_option_matches_across_separator_differences():
    # Run 20260817_124339 seg 7: the task says "select the no-reply option"; the real
    # option is a full address WITHOUT the hyphen, so both the fabricated full-address
    # call and a faithful 'no-reply' call missed. A separator-squashed comparison lets
    # the faithful fragment pick the UNIQUE containing option; ambiguity still refuses,
    # and short squashes never match (guard against 2-char accidents).
    opts = [{"id": "a", "text": "anirban.manna@actingoffice.com"},
            {"id": "b", "text": "noreply@actingoffice.com"}]
    assert agent_tools._choose_option(opts, "no-reply")["id"] == "b"
    # A word-aligned spelling ('no.reply@…') still wins via the existing partial tier.
    assert agent_tools._choose_option(
        opts + [{"id": "c", "text": "no.reply@other.com"}], "no-reply")["id"] == "c"
    # Two squash-only candidates: ambiguous, refuse (the receipt lists the options).
    assert agent_tools._choose_option(
        [{"id": "b", "text": "noreply@actingoffice.com"},
         {"id": "c", "text": "noreply@other.com"}], "no-reply") is None
    assert agent_tools._choose_option(opts, "an") is None


async def test_cb_resolve_from_input_placeholder_and_container():
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        # From the combobox input itself.
        got = await page.eval_on_selector("#react-select-4-input", _resolve_expr())
        assert got == {"input_id": "react-select-4-input"}
        # From the visible placeholder/value div — the label-to-input association the
        # snapshot can't provide (inputs render nameless).
        got = await page.eval_on_selector("#v2", _resolve_expr())
        assert got == {"input_id": "react-select-4-input"}
        # From the widget container.
        got = await page.eval_on_selector("#c1", _resolve_expr())
        assert got == {"input_id": "react-select-3-input"}
        # The root stamp lands on the resolved widget, not the page.
        assert await page.eval_on_selector(
            "#c1", "el => el.hasAttribute('data-ao-cb-root')")


async def test_cb_resolve_refuses_multi_combobox_wrappers():
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        got = await page.eval_on_selector("body", _resolve_expr())
        assert got["error"] == "ambiguous"
        assert got["count"] == 2
        got = await page.eval_on_selector("label", _resolve_expr())
        # The 'Frequency' label's parent .field contains exactly one combobox — resolves.
        assert got == {"input_id": "react-select-3-input"}


async def test_cb_open_list_pick_verify_against_react_select_semantics():
    """The full transaction the tool drives: open fires the control's mousedown, options
    are read by instance prefix, pick fires the option's mousedown (el.click() would be
    a no-op here), and state shows the committed value with the menu closed."""
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        await page.eval_on_selector("#v2", _resolve_expr())

        assert (await page.evaluate(_cb_expr("open", "react-select-4-input")))["ok"]
        got = await page.evaluate(_cb_expr("options", "react-select-4-input"))
        assert [o["text"] for o in got["options"]] == ["P45", "P46", "Existing employee"]

        chosen = agent_tools._choose_option(got["options"], "existing employee")
        picked = await page.evaluate(
            _cb_expr("pick", "react-select-4-input", chosen["id"]))
        assert picked["clicked"] is True
        assert picked["attrs"]["id"] == "react-select-4-option-2"
        assert await page.evaluate("window.__picked") == "Existing employee"

        state = await page.evaluate(_cb_expr("state", "react-select-4-input"))
        assert state["menu_open"] is False
        assert "Existing employee" in state["display"]


def _scripted_combobox(monkeypatch, resolve, ops):
    """Wire _combobox_select's collaborators to canned responses. `ops` maps op-name to
    a value or a list consumed in order."""
    async def call_on_field(_handle, _decl, args=None):
        return resolve
    monkeypatch.setattr(agent_tools, "_call_on_field", call_on_field)

    async def eval_js(_session, expr, **_kw):
        for op in ("open", "options", "pick", "state"):
            if f'"{op}"' in expr.partition("var input")[0]:
                v = ops.get(op)
                if isinstance(v, list):
                    return v.pop(0) if len(v) > 1 else v[0]
                return v
        return {"ok": True}  # the focus() expression
    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)


class _CbNode:
    tag_name = "div"
    backend_node_id = 1
    attributes: dict = {}


class _CbSession:
    async def get_element_by_index(self, _index):
        return _CbNode()

    async def get_or_create_cdp_session(self):
        raise RuntimeError("no cdp in unit test")  # filter typing becomes a no-op


def _cb_action():
    fn, pm = _registered_action("select_dropdown")

    async def run(monkeypatch, text="Existing employee"):
        async def field_handle(_s, _n):
            return ("cdp", "obj")
        monkeypatch.setattr(agent_tools, "_field_handle", field_handle)
        return await fn(params=pm(index=7, text=text), browser_session=_CbSession())
    return run


async def test_combobox_select_success_receipt_and_replayable_metadata(monkeypatch):
    opts = {"options": [{"id": "react-select-4-option-2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts,
         "pick": {"clicked": True, "tag": "div",
                  "attrs": {"id": "react-select-4-option-2", "role": "option"}},
         "state": {"menu_open": False, "display": "Join with Existing employee"}})
    res = await _cb_action()(monkeypatch)
    assert res.error is None
    assert "Selected 'Existing employee'" in res.extracted_content
    assert "Do NOT set it again" in res.extracted_content
    el = res.metadata["interacted_element"]
    assert el["attributes"]["id"] == "react-select-4-option-2"
    assert el["ax_name"] == "Existing employee"


async def test_combobox_select_lists_real_options_on_a_miss(monkeypatch):
    # THE receipt that would have saved the live run: the agent hunted 'Aaran Macleod'
    # while the menu only ever offered P45/P46/Existing employee.
    opts = {"options": [{"id": "o0", "text": "P45"}, {"id": "o1", "text": "P46"},
                        {"id": "o2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts})
    res = await _cb_action()(monkeypatch, text="Aaran Macleod")
    assert res.error is not None
    assert "no such option" in res.error
    assert "'P45', 'P46', 'Existing employee'" in res.error
    assert "Do NOT hunt the page for 'Aaran Macleod'" in res.error


async def test_combobox_select_read_back_mismatch_fails_honestly(monkeypatch):
    opts = {"options": [{"id": "o2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts,
         "pick": {"clicked": True, "tag": "div", "attrs": {"id": "o2"}},
         "state": {"menu_open": False, "display": "Select..."}})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "does not show it" in res.error
    assert "Do not report it as set" in res.error


async def test_combobox_select_ambiguous_container_guides_the_agent(monkeypatch):
    _scripted_combobox(monkeypatch, {"error": "ambiguous", "count": 2}, {})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "2 different comboboxes" in res.error
    assert "find_by_text" in res.error


async def test_combobox_select_non_combobox_element_says_so(monkeypatch):
    _scripted_combobox(monkeypatch, {"error": "none"}, {})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "no combobox input" in res.error


async def _instant_sleep(_seconds):
    return None


async def test_combobox_select_zero_options_recovers_via_escape_reopen(monkeypatch):
    # Run 20260807_110530: after a full page reload the employee combobox showed
    # 'No options' for every retyped query. The zero-options branch now runs one
    # Escape (clear stuck filter) + reopen cycle and looks at the UNFILTERED list.
    polls = [[], [], [{"id": "o7", "text": "Daniel Bruce"}]]

    async def poll(_s, _id, timeout):
        return polls.pop(0) if polls else [{"id": "o7", "text": "Daniel Bruce"}]
    monkeypatch.setattr(agent_tools, "_cb_poll_options", poll)
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True},
         "pick": {"clicked": True, "tag": "div",
                  "attrs": {"id": "o7", "role": "option"}},
         "state": {"menu_open": False, "display": "Daniel Bruce"}})
    res = await _cb_action()(monkeypatch, text="Daniel Bruce")
    assert res.error is None
    assert "Selected 'Daniel Bruce'" in res.extracted_content


async def test_combobox_select_dead_source_receipt_prescribes_reload_and_wait(monkeypatch):
    # When the option list never renders even after both recovery cycles, the receipt
    # must defer to the task's own recovery (refresh-and-retry wording) or a reload—
    # NOT steer away from reloading (run 20260807_120553: an anti-reload receipt made
    # the agent wander to the frequency dropdown and then type into the unloaded
    # template grid instead of following the task's refresh loop).
    async def poll(_s, _id, timeout):
        return []
    monkeypatch.setattr(agent_tools, "_cb_poll_options", poll)
    monkeypatch.setattr(agent_tools.asyncio, "sleep", _instant_sleep)
    _scripted_combobox(monkeypatch, {"input_id": "react-select-4-input"},
                       {"open": {"ok": True}})
    res = await _cb_action()(monkeypatch, text="Daniel Bruce")
    assert res.error is not None
    assert "NEVER rendered" in res.error
    assert "did not load on this page view" in res.error
    assert "the task prescribes" in res.error
    assert "WAIT for the page to finish loading" in res.error
    assert "WITHOUT a full browser reload" not in res.error
    assert "wait a moment" not in res.error


# --------------------- find_by_text: wrapper duplicates and refusals ---------------------
# Run 20260824_155123 seg 2 (the OTP segment). The app renders "Get OTP" as a
# cursor:pointer <div> holding a same-text child — captured live in the recording's
# interacted_element: the clicked control's xpath is
# .../form/div[1]/div/div and the OTP number that replaces it is .../form/div[1]/div/div/span.
# find_by_text('Get OTP', click_first=true) therefore saw 2 candidates on EVERY attempt
# and refused; the OTP was never fetched. Two defects, both guarded below.


class _NestedDomNode(_FakeDomNode):
    """A snapshot node that knows its parent — the shape _collapse_nested_duplicates
    walks. _FakeDomNode hardcodes one backend id, so these carry distinct ones."""

    def __init__(self, text, attributes=None, backend_node_id=0, parent_node=None,
                 tag_name="div"):
        super().__init__(text, attributes)
        self.backend_node_id = backend_node_id
        self.parent_node = parent_node
        self.node_name = tag_name.upper()
        self.tag_name = tag_name


def _get_otp_pair():
    """The live pair: an outer cursor:pointer <div> and its same-text inner <div>."""
    outer = _NestedDomNode("Get OTP", attributes={"style": "cursor: pointer"},
                           backend_node_id=19426)
    inner = _NestedDomNode("Get OTP", backend_node_id=19427, parent_node=outer)
    return outer, inner


async def test_find_by_text_clicks_through_same_text_wrapper(monkeypatch):
    """A control wrapped in a same-text div is ONE candidate, not an ambiguity."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    outer, inner = _get_otp_pair()
    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Get OTP", click_first=True),
                   browser_session=_FakeClickSession({19426: outer, 19427: inner},
                                                     result=None))

    assert res.error is None
    assert "clicked the single match" in res.extracted_content
    # the DEEPEST node: a click there bubbles to every ancestor's handler, the reverse
    # does not hold.
    assert "index=19427" in res.extracted_content


def test_collapse_keeps_siblings_and_differently_labelled_ancestors():
    """The collapse must only ever eat WRAPPERS. Two separate controls with the same
    text stay two candidates, and a container holding the control plus other text
    (a grid row) is labelled differently and is left alone."""
    from automation.pipeline.agent_tools import _collapse_nested_duplicates

    row = _NestedDomNode("1 Get OTP FOOD LIMITED", backend_node_id=100)
    button = _NestedDomNode("Get OTP", backend_node_id=101, parent_node=row)
    other = _NestedDomNode("Get OTP", backend_node_id=102)
    matches = [(1, row, "1 Get OTP FOOD LIMITED"), (2, button, "Get OTP"),
               (3, other, "Get OTP")]

    kept = _collapse_nested_duplicates(matches)

    assert [idx for idx, _n, _l in kept] == [1, 2, 3]


def test_collapse_survives_nodes_without_a_parent_chain():
    """Snapshot nodes that expose no parent_node (0-size/off-screen captures) must pass
    straight through rather than crash the lookup."""
    from automation.pipeline.agent_tools import _collapse_nested_duplicates

    a, b = _FakeDomNode("Get OTP"), _FakeDomNode("Get OTP")
    matches = [(1, a, "Get OTP"), (2, b, "Get OTP")]

    assert _collapse_nested_duplicates(matches) == matches


async def test_find_by_text_ambiguous_click_first_refuses_on_the_error_channel():
    """click_first that clicked NOTHING is a REFUSAL: it must ride the error channel so
    multi_act drops the step's remaining queued actions. On the success channel it did
    not — run 20260824_155123 seg 2 batched this refusal with a click(index) picked from
    the PREVIOUS snapshot, the stale click fired onto the panel's nameless close button,
    and the Payroll Review panel was lost (twice in one run)."""
    fn, pm = _registered_action("find_by_text")
    left = _NestedDomNode("Save draft", backend_node_id=1)
    right = _NestedDomNode("Save and send", backend_node_id=2)
    res = await fn(params=pm(text="Save", click_first=True),
                   browser_session=_FakeBrowserSession({4: left, 9: right}))

    assert res.extracted_content is None
    assert "NOTHING WAS CLICKED" in res.error
    assert "index=4" in res.error and "index=9" in res.error   # candidates still listed
    assert res.metadata == {"no_click": True}


async def test_find_by_text_listing_without_click_first_stays_a_result():
    """A plain listing clicked nothing because it was never asked to — that is not a
    failed action, and erroring it would break every batch that legitimately looks
    before it clicks."""
    fn, pm = _registered_action("find_by_text")
    left = _NestedDomNode("Save draft", backend_node_id=1)
    right = _NestedDomNode("Save and send", backend_node_id=2)
    res = await fn(params=pm(text="Save", click_first=False),
                   browser_session=_FakeBrowserSession({4: left, 9: right}))

    assert res.error is None
    assert "2 match(es)" in res.extracted_content
    assert "click(index) NOW" in res.extracted_content


# --- select_dropdown addressed by its neighbouring label -------------------------------
# select_dropdown was the one tool that always worked on the Send Email "From" field
# (4/4 in run 20260827_012631) and the only one that demanded an INDEX. near_text removes
# the index so the task's own wording maps to a single call. These exercise the resolver
# JS against a DOM shaped like the live one: the label is a BARE TEXT NODE sharing a block
# with the widget's current value, and react-select nests its input four levels deep.

import json

from automation.pipeline.agent_tools import _CB_RESOLVE_BY_TEXT_JS
from tests.test_heal_promotion import _launch

_EMAIL_PANEL = """
  <div class='email-panel'>
    <span>Send email</span>
    <div class='form-row'>
      From
      <div class='rs-container'><div class='rs-control'><div class='rs-valueContainer'>
        <div class='rs-singleValue'>Me</div>
        <div class='rs-inputContainer'>
          <input class='rs-input' id='react-select-20-input' type='text' role='combobox'
                 aria-autocomplete='list' aria-expanded='false' aria-haspopup='true'>
        </div>
      </div></div></div>
      <label><input type='checkbox'> Include signature</label>
    </div>
    <button id='mailbtn'>Send</button>
  </div>
"""


async def _resolve(html, near_text):
    from playwright.async_api import async_playwright

    tokens = [t for t in near_text.lower().replace("-", " ").split() if t]
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(html)
        try:
            return await page.evaluate(_CB_RESOLVE_BY_TEXT_JS % json.dumps(tokens))
        finally:
            await browser.close()


async def test_from_resolves_to_the_nameless_react_select_input():
    got = await _resolve(_EMAIL_PANEL, "From")
    assert len(got["hits"]) == 1
    assert got["hits"][0]["id"] == "react-select-20-input"
    assert got["hits"][0]["native"] is False


async def test_two_dropdowns_are_told_apart_by_their_labels():
    """Three anonymous comboboxes on one page is the live shape — each must resolve to
    itself, and never to a neighbour."""
    html = """<div>
        <div class='row'>Tax year<div><div><input role='combobox' id='cb-a'></div></div></div>
        <div class='row'>Period<div><div><input role='combobox' id='cb-b'></div></div></div>
      </div>"""
    a = await _resolve(html, "Tax year")
    b = await _resolve(html, "Period")
    assert [h["id"] for h in a["hits"]] == ["cb-a"]
    assert [h["id"] for h in b["hits"]] == ["cb-b"]


async def test_an_ambiguous_label_reports_every_match():
    """Two boxes under the same word must REFUSE, not silently take the first — this page
    has comboboxes with no distinguishing attributes at all."""
    html = """<div>
        <div class='row'>Amount<div><input role='combobox' id='cb-a'></div></div>
        <div class='row'>Amount<div><input role='combobox' id='cb-b'></div></div>
      </div>"""
    got = await _resolve(html, "Amount")
    assert len(got["hits"]) == 2


async def test_a_miss_reports_the_labels_that_do_exist():
    got = await _resolve(_EMAIL_PANEL, "Subject")
    assert got["hits"] == []
    assert any("From" in lbl for lbl in got["labels"])


async def test_a_native_select_is_found_and_flagged_native():
    html = """<div class='row'>Country<select id='sel-a'>
                <option>United Kingdom</option><option>Australia</option></select></div>"""
    got = await _resolve(html, "Country")
    assert len(got["hits"]) == 1
    assert got["hits"][0]["native"] is True


async def test_a_real_label_element_and_aria_label_both_win():
    """The wired-up rungs still take priority over the neighbour text."""
    html = """<div class='row'>Ignore me
                <input role='combobox' id='cb-a' aria-label='Tax year'></div>"""
    got = await _resolve(html, "Tax year")
    assert [h["id"] for h in got["hits"]] == ["cb-a"]


async def test_a_hidden_dropdown_is_not_a_match():
    """A collapsed panel's twin must not shadow the visible control the user named."""
    html = """<div>
        <div class='row' style='display:none'>From<input role='combobox' id='cb-hidden'></div>
        <div class='row'>From<input role='combobox' id='cb-live'></div>
      </div>"""
    got = await _resolve(html, "From")
    assert [h["id"] for h in got["hits"]] == ["cb-live"]


async def test_native_select_by_label_stamps_a_replayable_identity():
    """compile keys the native-select step on metadata.interacted_element; without it the
    pick is dropped and replay submits the form with its defaults. The resolver's synthetic
    'ao-cb-N' id must NOT reach that identity — it does not exist on the next run."""
    from playwright.async_api import async_playwright

    from automation.pipeline.agent_tools import _NATIVE_SELECT_BY_ID_JS

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(
            "<div>Country<select name='country' id='ao-cb-1'>"
            "<option>United Kingdom</option><option>Australia</option></select></div>")
        try:
            got = await page.evaluate(_NATIVE_SELECT_BY_ID_JS % {
                "id": json.dumps("ao-cb-1"), "want": json.dumps("Australia")})
        finally:
            await browser.close()
    assert got["ok"] is True
    assert got["shows"] == "Australia"
    assert got["attrs"]["name"] == "country"
    assert "id" not in got["attrs"], "synthetic resolver id must not become a selector"


def test_near_text_takes_precedence_over_a_supplied_index():
    """Live run 20260827_0152: the agent supplies BOTH (near_text='From', index=348). An
    `index < 0` guard made the label dead weight and left the pick riding a guessed index.
    The label is the robust address; the index is only the fallback."""
    src = pathlib.Path(agent_tools.__file__).read_text()
    body = src[src.index("async def select_dropdown("):]
    body = body[:body.index("\n    @tools.action")]
    assert "if near_text.strip():" in body, "near_text must be tried first, unconditionally"
    assert "if not by_label.error or index < 0:" in body, "index is the fallback only"


async def test_the_label_resolver_stamps_the_widget_root():
    """The 'state' read-back reads the widget root via [data-ao-cb-root]. Without the same
    stamp the index path sets, the label path clicked the right option and then failed its
    OWN verification: "the combobox does not show it (reads: '')" (live run 20260827_0159).
    A unique hit must leave exactly one stamped root, and it must contain the input."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_EMAIL_PANEL)
        try:
            await page.evaluate(_CB_RESOLVE_BY_TEXT_JS % json.dumps(["from"]))
            roots = await page.eval_on_selector_all(
                "[data-ao-cb-root]",
                "els => els.map(e => !!e.querySelector('#react-select-20-input'))")
            # An ambiguous resolve must NOT leave a stamp pointing at an arbitrary box.
            await page.evaluate(_CB_RESOLVE_BY_TEXT_JS % json.dumps(["nosuchlabel"]))
            after_miss = await page.eval_on_selector_all("[data-ao-cb-root]", "els => els.length")
        finally:
            await browser.close()
    assert roots == [True], f"expected exactly one stamped root holding the input, got {roots}"
    assert after_miss == 1, "a miss must leave the previous stamp alone, not re-point it"


# --------------------------- close: last-tab guard ---------------------------
#
# Run 20260827_091313 (subtask 8, "… click submit, and then close this tab"): the agent
# closed the aux client tab correctly, then closed the app's own /datarequests tab one step
# later on an invented memory of a third tab. Zero tabs remained, browser-use spawned a
# fresh about:blank, and the next subtask had no app page and no history to go back to — it
# guessed a URL, hit ERR_NAME_NOT_RESOLVED, and the run stopped. `close` is stock
# browser-use with no wrapper, so nothing stood between that second call and the workflow's
# only page.


class _FakeTabSession:
    """browser_session double exposing just get_tabs() — what the guard reads."""

    def __init__(self, tab_ids, raises=False):
        from types import SimpleNamespace

        self._tabs = [SimpleNamespace(target_id="0000" + t, url="https://test.actingoffice.com/x")
                      for t in tab_ids]
        self._raises = raises

    async def get_tabs(self):
        if self._raises:
            raise RuntimeError("browser not connected")
        return self._tabs


async def test_close_refuses_the_last_open_tab():
    fn, pm = _registered_action("close")
    session = _FakeTabSession(["85A6"])

    res = await fn(params=pm(tab_id="85A6"), browser_session=session)

    # error channel: multi_act must stop, and a queued `done` must not ride on a close
    # that never happened.
    assert res.error, f"expected a refusal, got {res!r}"
    assert "85A6" in res.error
    assert "last" in res.error.lower()
    assert not res.extracted_content or "Closed tab" not in res.extracted_content


async def test_close_proceeds_when_another_tab_would_remain():
    fn, pm = _registered_action("close")
    session = _FakeTabSession(["85A6", "9CD7"])

    res = await fn(params=pm(tab_id="9CD7"), browser_session=session)

    assert not res.error, f"a non-final close must be delegated, got {res.error!r}"


async def test_close_fails_open_when_the_tab_list_is_unreadable():
    """Never break the close path over a diagnosis: an unreadable tab list delegates."""
    fn, pm = _registered_action("close")
    session = _FakeTabSession(["85A6"], raises=True)

    res = await fn(params=pm(tab_id="85A6"), browser_session=session)

    assert not res.error, f"unreadable tab list must fail open, got {res.error!r}"


# ---- a click_first miss is a REFUSAL, not a probe (run 20260827_104331) --------------
# The OTP segment batched find_by_text('Close', click_first) -> click(index). The find
# missed, said so on the SUCCESS channel, and multi_act ran the queued click anyway: the
# Payroll Review panel was never closed, the click fired against an unchanged page, and
# the half-trace was committed as if the close had happened (entry 07044b6a0dbf7988).

async def test_a_click_first_miss_rides_the_error_channel(monkeypatch):
    async def raw(_session, _expr, **kw):
        return {"count": 0}                       # nothing, control-shaped or static
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Close", click_first=True),
                   browser_session=_FakeBrowserSession({}))

    assert res.error and "0 matches" in res.error
    assert not res.extracted_content                 # nothing on the success channel
    assert (res.metadata or {}).get("no_click") is True


async def test_a_probe_miss_still_rides_the_success_channel(monkeypatch):
    """click_first=False IS a question — "is this text here?" — and no is a valid answer."""
    async def raw(_session, _expr, **kw):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Close"), browser_session=_FakeBrowserSession({}))

    assert not res.error
    assert "0 matches" in res.extracted_content
    assert (res.metadata or {}).get("no_click") is True


async def test_a_click_first_miss_on_static_text_also_refuses(monkeypatch):
    """The static-text branch is the same fact — nothing was clicked — so a click_first
    call must stop the batch there too, while a probe keeps its helpful receipt."""
    async def raw(_session, expr, **kw):
        if "DOCLICK" in expr:
            return {"count": 0}
        return {"count": 1, "name": "Period to",
                "element": {"tag": "div", "attrs": {}, "xpath": ""}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="Period to", click_first=True),
                   browser_session=_FakeBrowserSession({}))

    assert res.error and "STATIC text" in res.error
    assert (res.metadata or {}).get("no_click") is True


async def test_a_nameless_click_stamps_the_row_it_happened_in(monkeypatch):
    # The recording side of the wrong-row fix (run 20260827_104331): a click with no name
    # of its own compiles to a positional row path unless the row travels with it.
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)

    async def fake_row_cells(_session, _node):
        return {"scope": '[role="row"]', "cells": ["PR/01797494/27/CDR072"]}

    monkeypatch.setattr(agent_tools, "_row_cells", fake_row_cells)
    node = _FakeDomNode("", attributes={"title": "Open payroll review request as client"})

    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4), _FakeBrowserSession({4: node}))

    row = ((res.metadata or {}).get("interacted_element") or {}).get("row")
    assert row and row["cells"] == ["PR/01797494/27/CDR072"]


async def test_a_named_click_stamps_no_row(monkeypatch):
    """A named control is guarded by expect_text at replay; the probe must not run."""
    from types import SimpleNamespace

    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    probed = []

    async def fake_row_cells(_session, node):
        probed.append(node)
        return {"scope": "tr", "cells": ["nope"]}

    monkeypatch.setattr(agent_tools, "_row_cells", fake_row_cells)
    node = _FakeDomNode("", attributes={"aria-label": "Delete row"})

    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4), _FakeBrowserSession({4: node}))

    assert not probed
    assert not ((res.metadata or {}).get("interacted_element") or {}).get("row")


# ---------------- repeat_click: the live counter tool (2026-08-28) ----------------
# Repetition had no first-class expression: the agent issued N clicks and the COMPILER
# guessed from adjacency whether they were iterations or slow-app retries — a guess keyed on
# `kind: loop`, itself inferred from wording. One counted call states the fact instead.


def _ready(monkeypatch, verdicts):
    """Script the between-clicks readiness probe. Each entry is (ready, why).

    Also stubs ClickElementEvent: browser-use validates a real DOM node on construction, and
    what is under test here is the COUNTING — how many clicks the tool issues and what it
    reports — not browser-use's event model."""
    seq = list(verdicts)

    async def fake(_session, _node):
        return seq.pop(0) if seq else (False, "the control left the page")
    monkeypatch.setattr(agent_tools, "_repeat_ready", fake)
    monkeypatch.setattr(agent_tools, "_REPEAT_SETTLE_S", 0)
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda **kw: object())


async def test_counted_mode_clicks_exactly_n_times(monkeypatch):
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _FakeDomNode("Save & Next")})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]
    _ready(monkeypatch, [(True, "")] * 10)

    res = await fn(index=4, times=6, browser_session=session)

    assert len(clicks) == 6                          # exactly, not 5 and not 7
    assert res.metadata["repeat"] == {"count": 6, "until_done": False,
                                      "wait_s": agent_tools._REPEAT_SETTLE_S}
    assert "6 of 6" in res.extracted_content


async def test_until_done_stops_when_the_control_stops_advancing(monkeypatch):
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _FakeDomNode("Next")})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]
    # ready after clicks 1 and 2, gone after the third — a three-row list.
    _ready(monkeypatch, [(True, ""), (True, ""), (False, "the control became disabled")])

    res = await fn(index=4, times=0, browser_session=session)

    assert len(clicks) == 3
    assert res.metadata["repeat"]["until_done"] is True
    assert res.metadata["repeat"]["count"] == 3      # the REAL number, reported
    assert "stopped advancing" in res.extracted_content


async def test_a_shortfall_is_an_error_and_never_compiles(monkeypatch):
    """Asked for 6, the control died after 2. The clicks that landed are real, but a wrong
    count must never become a cached step — so it rides the error channel with no_click."""
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _FakeDomNode("Save & Next")})
    session.event_bus.dispatch = lambda _e: _FakeEvent(None)
    _ready(monkeypatch, [(True, ""), (False, "the control left the page")])

    res = await fn(index=4, times=6, browser_session=session)

    assert res.error and "only 2 of the 6" in res.error
    assert res.metadata == {"no_click": True}
    assert "repeat" not in (res.metadata or {})


async def test_a_bad_index_refuses_without_clicking(monkeypatch):
    _ready(monkeypatch, [])
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({})
    res = await fn(index=99, times=3, browser_session=session)
    assert res.error and "not available" in res.error
    assert res.metadata == {"no_click": True}


async def test_still_advancing_at_the_cap_is_a_failure_not_a_finished_list(monkeypatch):
    """The until-done cap is a runaway bound. Reaching it means the list never ended, which
    must not be reported as a complete run."""
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _FakeDomNode("Next")})
    session.event_bus.dispatch = lambda _e: _FakeEvent(None)
    monkeypatch.setattr(agent_tools, "_REPEAT_HARD_CAP", 4)
    _ready(monkeypatch, [(True, "")] * 10)

    res = await fn(index=4, times=0, browser_session=session)

    assert res.error and "safety cap" in res.error
    assert res.metadata == {"no_click": True}


# ---- the declared budget stops a redundant repeat (2026-09-01) ----
# Run 20260901_122209 subtask 15 ("exactly 5 more clicks"): the agent clicked Save & Next
# once by hand, then called repeat_click(times=5) THREE times — each eval saying the previous
# repeat had succeeded, then re-forming the same goal — for 16 clicks and 26 payroll writes
# where the task wanted 11. The prompt already told it to call repeat_click ONCE, so the
# budget is enforced rather than advised.


def _budget(monkeypatch, n, verdicts=None):
    _ready(monkeypatch, verdicts if verdicts is not None else [(True, "")] * 40)
    monkeypatch.setattr(agent_tools, "_CLICK_LEDGER", {})
    monkeypatch.setattr(agent_tools, "_REPEAT_BUDGET", n)


def _save_next(name="Save & Next"):
    node = _FakeDomNode(name, attributes={"id": "btnSave"})
    node.ax_node = type("_Ax", (), {"name": name})()
    return node


async def test_the_receipt_names_the_control(monkeypatch):
    """It read node.ax_name — which does not exist on a live node — and so reported
    "Clicked element 1063 5 times" for a button called Save & Next."""
    _budget(monkeypatch, None)
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _save_next()})
    session.event_bus.dispatch = lambda _e: _FakeEvent(None)

    res = await fn(index=4, times=3, browser_session=session)

    assert "Save & Next" in res.extracted_content
    assert "element 4" not in res.extracted_content


async def test_a_second_repeat_is_refused_once_the_budget_is_spent(monkeypatch):
    _budget(monkeypatch, 5)
    fn, _pm = _registered_action("repeat_click")
    node = _save_next()
    session = _FakeClickSession({4: node})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]

    first = await fn(index=4, times=5, browser_session=session)
    assert first.error is None and len(clicks) == 5
    assert first.metadata["repeat"]["count"] == 5

    second = await fn(index=4, times=5, browser_session=session)
    assert second.error and "COMPLETE" in second.error
    assert second.metadata == {"no_click": True}      # never compiles
    assert len(clicks) == 5, "the refused call must not click at all"


async def test_a_manual_click_counts_against_the_budget(monkeypatch):
    """The agent clicked Save & Next once before repeating; a budget that ignored plain
    clicks would still overshoot by one."""
    _budget(monkeypatch, 5)
    node = _save_next()
    agent_tools._note_clicks(node, 1)                  # what the `click` override records

    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: node})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]

    res = await fn(index=4, times=5, browser_session=session)

    assert len(clicks) == 4, "clamped to the 4 remaining of 5"
    assert res.metadata["repeat"]["count"] == 4


async def test_no_declared_budget_leaves_the_tool_unconstrained(monkeypatch):
    _budget(monkeypatch, None)
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _save_next()})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]

    for _ in range(2):
        res = await fn(index=4, times=3, browser_session=session)
        assert res.error is None
    assert len(clicks) == 6                            # unconstrained, as before
    assert "6 click(s) on it in this step" in res.extracted_content


def test_the_ledger_is_per_segment(monkeypatch):
    monkeypatch.setattr(agent_tools, "_CLICK_LEDGER", {"id=btnSave": 5})
    agent_tools.set_live_network(object())
    try:
        assert agent_tools._CLICK_LEDGER == {}, "a new segment starts from zero"
    finally:
        agent_tools.clear_live_network()

