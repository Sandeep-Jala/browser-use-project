"""agent_tools: descendant icon-hint harvesting.

The failure that motivated this: Fluent UI icon buttons (`<button class="ms-Button--icon">`)
carry their meaning only in a child glyph, which browser-use serializes away — so the send,
edit, and delete icons all reach the agent as a nameless `<button/>`. `_descendant_icon_hints`
recovers that meaning from the child so nameless icon buttons become findable."""

from automation.pipeline import agent_tools
from automation.pipeline.agent_tools import (
    _descendant_icon_hints,
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

    assert res.error is None
    assert "did NOT click" in res.extracted_content
    assert "select_dropdown(index=9" in res.extracted_content
    assert res.metadata == {"no_click": True}    # compiles to NOTHING, never a phantom click


async def test_find_by_text_refuses_to_click_file_input():
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("", attributes={"type": "file", "name": "csv upload"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="csv upload", click_first=True),
                   browser_session=_FakeBrowserSession({4: node}))

    assert res.error is None
    assert "did NOT click" in res.extracted_content
    assert "upload_file" in res.extracted_content
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

    assert res.error is None
    assert "REFUSED" in res.extracted_content
    assert "Nothing was clicked" in res.extracted_content
    assert res.metadata == {"no_click": True}


# ------------------------- select_dropdown read-back override -------------------------
# The built-in trusts the picker's self-report; on an ad-heavy page the confirmation can
# time out AFTER the selection took, and the empty failure receipt made the agent re-set
# the same option repeatedly. The override reads the element back and reports what the
# select actually shows.


class _FakeSelectSession:
    def __init__(self, result=None, exc=None):
        from types import SimpleNamespace

        self._node = object()
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

