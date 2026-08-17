"""Compile coverage fixes surfaced by live NPS runs: scrolls must compile, a
metadata-less find_by_text click must not vanish, save_history's metadata drop must be
repaired, and a zero-step script must never be committed."""
import json
from types import SimpleNamespace

import pytest

from automation.pipeline import subtask_store as ss
from automation.pipeline.runner import restore_result_metadata
from automation.pipeline.script_compile import compile_recording
from automation.skills.codegen import lint_code, transpile


def _item(action, url="http://app/x", result=None, element=None, state_message=None):
    item = {"state": {"url": url, "interacted_element": [element] if element else []},
            "model_output": {"action": [action]},
            "result": result if result is not None else []}
    if state_message is not None:
        item["state_message"] = state_message
    return item


def _write(tmp_path, history):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"history": history}))
    return p


def test_scrolls_compile(tmp_path):
    history = [
        _item({"capped_scroll": {"down": True, "pages": 0.2}}),
        _item({"scroll": {"down": False, "num_pages": 2.0}}),   # built-in shape, capped
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [
        {"action": "scroll", "down": True, "pages": 0.2},
        {"action": "scroll", "down": False, "pages": 1.0},
    ]


def test_find_by_text_click_without_metadata_is_unanchorable(tmp_path):
    """No recorded element identity -> the click cannot be tied to a location. Recording
    the tool's TEXT SEARCH instead is what made replays hunt tokens and land on the
    wrong element, so compile marks the segment unanchorable (the commit is refused and
    it authors live) rather than baking a search step."""
    history = [_item({"find_by_text": {"text": "View all", "click_first": True}},
                     result=[{"extracted_content": "clicked"}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["unanchorable"]
    assert "View all" in steps[0]["why"]
    # A non-clicking find_by_text still compiles to nothing.
    history = [_item({"find_by_text": {"text": "View all"}})]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_unexecuted_actions_are_never_compiled(tmp_path):
    """multi_act stops the queue on a refusal: the later actions of that step never ran
    and have no result slot. Compiling them baked phantom steps — a find_by_text that
    never happened became find_click('Ayaan Campbell') and failed every replay of the
    pay-forecast segment (run 20260813_132507)."""
    history = [{
        "state": {"url": "http://app/x", "interacted_element": []},
        "model_output": {"action": [
            {"input": {"index": 7447, "text": "Ayaan Campbell", "clear": True}},
            {"wait": {"seconds": 2}},
            {"find_by_text": {"text": "Ayaan Campbell", "click_first": True}},
            {"wait": {"seconds": 2}},
        ]},
        "result": [{"error": "REFUSED — did NOT type 'Ayaan Campbell': …",
                    "metadata": {"no_fill": True}}],
    }]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_restore_result_metadata_reinjects_dropped_field(tmp_path):
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"history": [
        {"state": {"url": "u"}, "model_output": {"action": [{"find_by_text": {}}]},
         "result": [{"extracted_content": "clicked"}]},
    ]}))
    element = {"node_name": "button", "ax_name": "View all", "attributes": {}}
    in_memory = SimpleNamespace(history=[
        SimpleNamespace(result=[
            SimpleNamespace(metadata={"interacted_element": element}),
        ]),
    ])
    assert restore_result_metadata(in_memory, rec) is True
    saved = json.loads(rec.read_text())
    assert saved["history"][0]["result"][0]["metadata"]["interacted_element"] == element
    # Idempotent: a second pass changes nothing.
    assert restore_result_metadata(in_memory, rec) is False


def test_transpiler_maps_scroll():
    steps = [{"action": "scroll", "down": True, "pages": 0.2},
             {"action": "scroll", "down": False, "pages": 0.5}]
    code, anchors = transpile("s", steps)
    assert "await api.scroll(0.2)" in code
    assert "await api.scroll(0.5, down=False)" in code
    assert anchors == {} and lint_code(code) == []


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    (tmp_path / "library").mkdir()
    return tmp_path


async def test_zero_step_recording_is_never_committed(stores, monkeypatch):
    """An agent that acted only through non-compiling tools must leave NO library entry —
    an empty script would replay as a hollow no-op pass."""
    from automation.pipeline import hybrid
    from tests.test_hybrid import FakeSession, _runner, _seg, _run

    empty_recording = {"history": [
        {"state": {"url": "http://app/section", "interacted_element": []},
         "model_output": {"action": [{"search_page": {"pattern": "Reviews"}}]},
         "result": []},
    ]}
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    fake.recording = empty_recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True          # the segments themselves passed
    assert ss.load_manifest() == {}              # but nothing hollow was committed
    for sid_file in (ss.LIBRARY_DIR).glob("*.steps.json"):
        raise AssertionError(f"unexpected committed steps: {sid_file}")


async def test_zero_step_noop_authoring_logs_info_not_dropped_tools(
        tmp_path, monkeypatch, caplog, stores):
    """The two zero-step compiles read differently: a done-only history is a conditional
    guard that was already satisfied (expected — INFO), while real actions that compiled
    away point at a compiler coverage gap (WARNING). The live misread: 'if the send
    email section is not closed, click save again' — the agent verified, took no action,
    and the log blamed dropped tools."""
    import logging

    from automation.pipeline import hybrid
    from tests.test_hybrid import FakeSession, _runner, _seg, _run

    # Pure-helper contract (the branch the message hangs on).
    p = tmp_path / "r.json"
    noop_recording = {"history": [
        {"state": {"url": "http://app/section", "interacted_element": []},
         "model_output": {"action": [{"done": {"success": True, "text": "closed"}}]},
         "result": []},
    ]}
    p.write_text(json.dumps(noop_recording))
    assert hybrid._recording_had_page_actions(p) is False
    p.write_text(json.dumps({"history": [
        {"model_output": {"action": [{"search_page": {"pattern": "x"}}]}}]}))
    assert hybrid._recording_had_page_actions(p) is True
    assert hybrid._recording_had_page_actions(tmp_path / "missing.json") is True

    # And through the commit path: a done-only authoring stays quiet (INFO, no WARNING).
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    fake.recording = noop_recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    with caplog.at_level(logging.INFO, logger="framework.hybrid"):
        result = await _run(fake)
    assert result.is_successful is True
    assert ss.load_manifest() == {}
    zero_msgs = [r for r in caplog.records if "ZERO steps" in r.getMessage()]
    noop_msgs = [r for r in caplog.records
                 if "without any page action" in r.getMessage()]
    assert not zero_msgs and noop_msgs
    assert all(r.levelno == logging.INFO for r in noop_msgs)


def test_find_by_text_no_click_results_compile_to_nothing(tmp_path):
    """The tool stamps no_click on probes and listings — outcomes that touched no
    element. Without the stamp+skip, a conditional guard's closed-panel probe
    (find_by_text('Save') -> 0 matches -> done) compiled to find_click('save') and
    failed every replay on the healthy page; a multi-match listing compiled to a
    phantom duplicate click. Metadata-LESS click_first results keep the legacy
    semantic-find_click fallback (browser-use's save_history used to drop metadata)."""
    probe = [_item({"find_by_text": {"text": "Save", "click_first": True}},
                   result=[{"metadata": {"no_click": True}}])]
    assert compile_recording(_write(tmp_path, probe), emit_start_goto=False) == []

    listing_then_click = [
        _item({"find_by_text": {"text": "Sent", "click_first": True}},
              result=[{"metadata": {"no_click": True}}]),
        _item({"click": {"index": 7}},
              element={"node_name": "span", "ax_name": "Sent",
                       "attributes": {}, "x_path": "html/body/span[1]"}),
    ]
    steps = compile_recording(_write(tmp_path, listing_then_click),
                              emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click"]   # no phantom find_click


def test_find_by_text_hidden_clicks_replay_the_tool_normal_ones_keep_selectors(tmp_path):
    """A hidden-control click (hover-revealed/0-size — the Reviews 'View all' icon) is
    anchored like any other click when the capture carries identity; hidden_ok keeps the
    hover/dispatch recovery as its safety net. A normal find_by_text click (snapshot
    path, full element identity) keeps the proven selector replay."""
    hidden = [_item({"find_by_text": {"text": "View all", "click_first": True}},
                    result=[{"metadata": {"interacted_element": {
                        "node_name": "button", "ax_name": "View all",
                        "attributes": {"title": "View all", "role": "button"},
                        "x_path": "html/body/button[2]",
                        "hidden_click": True}}}])]
    steps = compile_recording(_write(tmp_path, hidden), emit_start_goto=False)
    assert steps[0]["action"] == "click" and steps[0]["hidden_ok"] is True
    assert steps[0]["selectors"][0] == "xpath=/html/body/button[2]"

    normal = [_item({"find_by_text": {"text": "btnInvoice", "click_first": True}},
                    result=[{"metadata": {"interacted_element": {
                        "node_name": "button", "ax_name": "btnInvoice",
                        "attributes": {"aria-label": "btnInvoice"},
                        "x_path": "//button[1]"}}}])]
    steps = compile_recording(_write(tmp_path, normal), emit_start_goto=False)
    assert steps[0]["action"] == "click" and steps[0]["hidden_ok"] is True
    assert any("btnInvoice" in s for s in steps[0]["selectors"])


def test_find_click_transpiles_and_parameterizes():
    from automation.skills.codegen import lint_code, transpile

    steps = [{"action": "scroll", "down": True, "pages": 0.5},
             {"action": "find_click", "text": "{{label}}"},
             {"action": "find_click", "text": "View all"}]
    code, anchors = transpile("s", steps, params={"label": "Reviews"})
    assert "await api.find_click(label)" in code
    assert "await api.find_click('View all')" in code
    assert anchors == {} and lint_code(code) == []


def test_labelless_click_recovers_text_from_state_message(tmp_path):
    """The live 'Sent' status-chip failure: browser-use records the chip <span> with
    ax_name null and only class/style attributes, so compile collapsed it to a bare
    positional xpath — which resolved into a DIFFERENT element next run and broke the
    segment. The chip's text sits in the DOM listing right below its [backend_id] line;
    compile must recover it into a text= candidate, the fingerprint, and a landed-click
    expect_text guard."""
    sm = ("\t\t\t\t\t[17899]<span />\n"
          "\t\t\t\t\t\t[17898]<span />\n"
          "\t\t\t\t\t\t\tSent\n"
          "\t\t\t[18239]<div id=row1630-1 role=row />\n")
    chip = {"node_name": "SPAN", "backend_node_id": 17898, "ax_name": None,
            "attributes": {"class": "label label-148", "style": "display: block;"},
            "x_path": "html/body/div[1]/span/span"}
    history = [_item({"click": {"index": 17898}}, element=chip, state_message=sm)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    (step,) = steps
    # The chip carries no identity attribute, so the recorded location IS the anchor
    # (2026-08-13: text= candidates are gone — a replay locates, it does not search).
    # The recovered text still guards the landing and feeds heal scoring.
    assert step["selectors"] == ["xpath=/html/body/div[1]/span/span"]
    assert step["expect_text"] == "Sent"                # wrong-element clicks refuse
    assert step["fingerprint"]["text"] == "Sent"        # heal gets the text signal
    # And the DOM-listing text also names the codegen handle (not 'click').
    code, anchors = transpile("s", steps)
    assert "await api.click('sent')" in code and "sent" in anchors


def test_state_message_text_recovery_edges(tmp_path):
    """Recovery must not invent labels: a real ax_name wins, an element's own text is
    strictly DEEPER-indented than its line (same-indent text is a sibling's — the live
    'Select file' pickup under the Notes textarea), editable elements are exempt
    entirely, page-edge markers never count, and a >60-char blob stays out."""
    from automation.pipeline.script_compile import _sm_child_text, _with_recovered_text

    assert _sm_child_text("[7]<span />\n\tSent\n", 7) == "Sent"
    assert _sm_child_text("[7]<span />\nSelect file\n", 7) == ""       # sibling indent
    assert _sm_child_text("\t\t[7]<span />\n\t\tSelect file\n", 7) == ""
    assert _sm_child_text("[7]<span />\n\t... 1215 pixels below - scroll ...\n", 7) == ""
    assert _sm_child_text("[7]<span />\n\t" + "x" * 80 + "\n", 7) == ""
    assert _sm_child_text("[7]<span />\n\t[8]<div />\n\tSent\n", 7) == ""  # next element's
    assert _sm_child_text("", 7) == "" and _sm_child_text("[7]<span />\nSent", None) == ""

    named = {"ax_name": "Real Label", "backend_node_id": 7}
    assert _with_recovered_text(named, "[7]<span />\n\tOther\n") is named
    textarea = {"node_name": "TEXTAREA", "ax_name": None, "backend_node_id": 7}
    assert _with_recovered_text(textarea, "[7]<textarea />\n\tNotes\n") is textarea


def test_named_click_compiles_with_expect_text_blob_does_not(tmp_path):
    """Every click whose element carries a short accessible name is stamped expect_text
    (the landed-element guard, previously value-parameterized clicks only). A >60-char
    announcement blob must stamp nothing — it is a container, not a name."""
    named = {"node_name": "button", "ax_name": "Verify",
             "attributes": {"id": "btn-Verify", "type": "button"},
             "x_path": "html/body/button[1]"}
    blob = {"node_name": "a", "ax_name": "1 PR/01797494/27/DR018 FOOD LIMITED 2026-27 "
                                         "Payroll review 1 24/07/2026 Sent",
            "attributes": {"href": "/x"}, "x_path": "html/body/a[1]"}
    history = [_item({"click": {"index": 1}}, element=named),
               _item({"click": {"index": 2}}, element=blob)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps[0]["expect_text"] == "Verify"
    assert "expect_text" not in steps[1]


def test_compile_expect_text_flows_through_anchor_to_api_click():
    """A compile-time landed-name guard must survive the tier-1 transpile: into the
    anchor bundle, through _substituted_anchors untouched (no params), and into the
    step api.click builds."""
    from automation.skills.api import SkillApi
    from automation.skills.base import _substituted_anchors
    from automation.skills.codegen import _anchor

    step = {"action": "click", "selectors": ['text="Sent"', "xpath=/html/body/span"],
            "expect_text": "Sent"}
    anchor = _anchor(step)
    assert anchor["expect_text"] == "Sent"

    concrete = _substituted_anchors({"chip": anchor}, {})
    api = SkillApi(page=None, anchors=concrete)
    assert api._step_for("chip", expect=True)["expect_text"] == "Sent"
    assert "expect_text" not in api._step_for("chip")   # fill/select path unchanged

    # Tokenized guard (the parameterizer lifted the literal): substituted like a value.
    tokenized = {"chip": {"selectors": ['text="{{status}}"'],
                          "expect_text": "{{status}}"}}
    concrete = _substituted_anchors(tokenized, {"status": "Draft"})
    assert concrete["chip"]["expect_text"] == "Draft"


def test_hidden_ok_flows_through_anchors_and_api():
    from automation.skills.api import SkillApi
    from automation.skills.codegen import _anchor

    step = {"action": "click", "selectors": ['css=[title="View all"]'],
            "fingerprint": {"tag": "button"}, "hidden_ok": True}
    anchor = _anchor(step)
    assert anchor["hidden_ok"] is True

    api = SkillApi(page=None, anchors={"view-all": anchor})
    rebuilt = api._step_for("view-all")
    assert rebuilt["hidden_ok"] is True
    assert rebuilt["selectors"] == ['css=[title="View all"]']

    # A normal anchor carries no such permission.
    assert "hidden_ok" not in _anchor({"action": "click", "selectors": ["x"]})


def test_extract_data_with_element_compiles_to_selector_extract(tmp_path):
    """The recorded value is provenance only — the compiled step carries the LOCATOR (and
    the query as semantic fallback) so replay re-reads whatever the page shows then."""
    history = [_item({"extract_data": {"text": "result title", "label": "top_result_title"}},
                     result=[{"metadata": {"extract": {
                         "label": "top_result_title", "value": "Acting Office",
                         "query": "result title",
                         "interacted_element": {
                             "node_name": "h2", "ax_name": "Acting Office",
                             "attributes": {"data-testid": "result-title"},
                             "x_path": "//h2[1]"}}}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    step = steps[0]
    assert step["action"] == "extract" and step["label"] == "top_result_title"
    assert step["query"] == "result title"
    assert step["selectors"] and step.get("fingerprint")


def test_extract_data_element_less_and_valueless_forms(tmp_path):
    # No stable element identity -> a query-only extract (semantic re-find at replay).
    history = [_item({"extract_data": {"text": "result title", "label": "t"}},
                     result=[{"metadata": {"extract": {
                         "label": "t", "value": "x", "query": "result title",
                         "interacted_element": None}}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [{"action": "extract", "label": "t", "query": "result title"}]
    # A valueless extract_data records NO metadata and must compile to NOTHING.
    history = [_item({"extract_data": {"text": "ghost", "label": "g"}},
                     result=[{"extracted_content": "nothing matched"}])]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_transpiler_maps_extract_and_query_only_stays_tier0():
    steps = [{"action": "extract", "label": "top_result_title", "query": "result title",
              "selectors": ['css=[data-testid="result-title"]'],
              "fingerprint": {"tag": "h2"}}]
    code, anchors = transpile("s", steps)
    assert "await api.extract(" in code and "'top_result_title'" in code
    assert lint_code(code) == []
    (_handle, anchor), = anchors.items()
    assert anchor["selectors"] == ['css=[data-testid="result-title"]']
    assert anchor["query"] == "result title"      # the semantic fallback rides along

    with pytest.raises(ValueError, match="query-only extract"):
        transpile("s", [{"action": "extract", "label": "t", "query": "x"}])


def test_offsite_exit_click_is_dropped_but_recovery_and_reads_survive(tmp_path):
    """Site-boundary invariant (from the live fakenamegenerator failure): a segment is
    single-site by construction, so a recorded click whose consequence was leaving the
    site (stray ad/SSO overlay hit) compiles to NOTHING — otherwise replay requires the
    overlay to exist. The recorded recovery navigate still compiles (goto), and read-only
    actions at the boundary are never dropped (they did not cause the exit and may carry
    a needed observation)."""
    site = "https://www.fakenamegenerator.com/gen-random-gd-uk.php"
    off = "https://accounts.google.com/v3/signin"
    exit_el = {"node_name": "button", "ax_name": "Reload",
               "attributes": {"id": "reload-button"}}
    ok_el = {"node_name": "input", "ax_name": "Generate",
             "attributes": {"id": "genbtn", "type": "submit"}}
    history = [
        # Read on the site, right before a spontaneous redirect: KEPT.
        _item({"extract_data": {"text": "kirsty", "label": "n"}}, url=site,
              result=[{"metadata": {"extract": {"label": "n", "value": "Kirsty",
                                                "query": "kirsty",
                                                "interacted_element": None}}}]),
        # The exit click (next state is off-site): DROPPED.
        _item({"click": {"index": 4}}, url=site, element=exit_el),
        # The agent's recovery, taken while off-site: compiles to goto.
        _item({"navigate": {"url": "https://www.fakenamegenerator.com"}}, url=off),
        # Back on the site: compiles normally.
        _item({"click": {"index": 9}}, url="https://www.fakenamegenerator.com/",
              element=ok_el),
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["extract", "goto", "click"]
    assert steps[1]["url"] == "https://www.fakenamegenerator.com"
    assert not any("reload" in str(s).lower() for s in steps)


def test_select_dropdown_compiles_to_select_step(tmp_path):
    """Native <select> picks must compile — dropping them replayed the surrounding flow
    with the form's DEFAULTS (observed live: fakenamegenerator ended on gen-random-us-us
    instead of gd-uk because Scottish/United Kingdom were never selected)."""
    element = {"node_name": "select", "ax_name": "Name set",
               "attributes": {"id": "nameset", "name": "n"}}
    history = [_item({"select_dropdown": {"index": 34, "text": "Scottish"}},
                     element=element)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    step = steps[0]
    assert step["action"] == "select" and step["value"] == "Scottish"
    assert any("nameset" in s for s in step["selectors"])
    assert step.get("fingerprint")

    # Without a <select> element identity there is nothing stable to anchor on: the step
    # drops (as before) and the segment gate catches a wrong end state.
    history = [_item({"select_dropdown": {"index": 34, "text": "Scottish"}})]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_transpiler_maps_select():
    steps = [{"action": "select", "selectors": ['css=[id="nameset"]'],
              "value": "{{name_set}}", "fingerprint": {"tag": "select"}},
             {"action": "select", "selectors": ['css=[id="country"]'],
              "value": "United Kingdom", "fingerprint": {"tag": "select"}}]
    code, anchors = transpile("s", steps, params={"name_set": "Scottish"})
    assert "await api.select('nameset', name_set)" in code
    assert "await api.select('country', 'United Kingdom')" in code
    assert lint_code(code) == []
    assert anchors["nameset"]["selectors"] == ['css=[id="nameset"]']


async def test_run_steps_select_picks_by_label_and_fires_change():
    """The replay pick must go through select_option (label first, option-value fallback)
    so the page's own change listeners fire — the difference between the form actually
    switching and it submitting its defaults."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    html = """
      <select id="c">
        <option value="us" selected>United States</option>
        <option value="uk">United Kingdom</option>
      </select>
      <span id="out"></span>
      <script>
        document.getElementById('c').addEventListener('change',
          e => document.getElementById('out').textContent = e.target.value);
      </script>
    """
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()

        await page.set_content(html)
        out = await run_steps(page, [{"action": "select", "selectors": ["css=#c"],
                                      "value": "United Kingdom"}], timeout_ms=3000)
        assert out["failed_at"] is None
        assert await page.locator("#c").input_value() == "uk"
        assert await page.locator("#out").text_content() == "uk"   # change event fired

        # Label drifted -> the recorded text still matches via the option's VALUE attr.
        await page.set_content(html.replace(">United Kingdom<", ">United Kingdom (GB)<"))
        out = await run_steps(page, [{"action": "select", "selectors": ["css=#c"],
                                      "value": "uk"}], timeout_ms=3000)
        assert out["failed_at"] is None
        assert await page.locator("#c").input_value() == "uk"


async def test_run_steps_extract_query_reads_static_text():
    """A query-only extract must find values living in PLAIN text (no control wraps the
    fakenamegenerator identity block) — the static-text finder half of the live fix."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("""
          <div class="address">
            <h3>Felix MacDonald</h3>
            <div class="adr">93 Mounthoolie Lane<br>SUNNYSIDE<br>AB1 5AW</div>
          </div>
        """)
        steps = [
            {"action": "extract", "label": "generated_name", "query": "felix macdonald"},
            {"action": "extract", "label": "generated_address", "query": "93 mounthoolie lane"},
        ]
        out = await run_steps(page, steps, timeout_ms=3000)
        assert out["failed_at"] is None
        # The name alone in its <h3> is EXACTLY the query — zero information gained — so
        # the finder expands to the enclosing card (2026-07-29): the whole identity block
        # is the value, and the consuming step parses the facts out of it.
        # LINE STRUCTURE IS PRESERVED (2026-08-13): bindings slice fields out of a block
        # by line position, so the block must read back the same way it was captured.
        assert out["extracted"]["generated_name"].splitlines() == [
            "Felix MacDonald", "93 Mounthoolie Lane", "SUNNYSIDE", "AB1 5AW"]
        # The address capture already gains beyond its query (city and postcode ride
        # along), so it is NOT expanded — the tightest element stays the anchor.
        assert out["extracted"]["generated_address"].splitlines() == [
            "93 Mounthoolie Lane", "SUNNYSIDE", "AB1 5AW"]

        # A value that is truly absent still fails honestly.
        out = await run_steps(page, [{"action": "extract", "label": "g",
                                      "query": "ghost value"}], timeout_ms=1200)
        assert out["failed_at"] == 0 and "no visible text" in out["error"]

        # The finder hands back a positional xpath — the anchor that makes fresh-data
        # extracts replayable when the value's own text is the only other identity.
        # With the zero-gain expansion the anchor is the CARD, not the bare <h3>: a
        # replay re-reads the whole block (fresh name AND address) from this slot.
        import json as _json

        from automation.pipeline.script_compile import RAW_TEXT_FIND_JS
        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["felix", "macdonald"]))
        assert raw["expanded"] is True
        assert raw["element"]["xpath"] == "/html/body/div"
        # A query that already fills its element is untouched: no expansion flag.
        raw = await page.evaluate(
            RAW_TEXT_FIND_JS % _json.dumps(["93", "mounthoolie", "lane"]))
        assert raw["expanded"] is False
        assert raw["element"]["xpath"] == "/html/body/div/div"


async def test_extract_zero_gain_expansion_guards():
    """Expansion never fires when it cannot help: a first-gaining ancestor bigger than
    the 1000-char value cap keeps the tight capture (never a page blob), and a <select>
    (whose reported text is already the SELECTED option, not page prose) never expands."""
    import json as _json

    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import RAW_TEXT_FIND_JS
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        filler = "lorem ipsum dolor sit amet " * 60          # ~1600 chars of sibling text
        await page.set_content(f"""
          <div class="page">
            <h3>Rio Kerr</h3>
            <p>{filler}</p>
          </div>
        """)
        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["rio", "kerr"]))
        assert raw["expanded"] is False
        assert raw["name"] == "Rio Kerr"

        await page.set_content("""
          <select id="c"><option selected>United Kingdom</option>
          <option>France</option></select>
        """)
        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["united", "kingdom"]))
        assert raw["expanded"] is False
        assert raw["name"] == "United Kingdom"


def test_upload_file_action_compiles_to_basename_step(tmp_path):
    """A recorded upload_file compiles to an `upload` step carrying only the BASENAME
    (files.UPLOADS_DIR is the implied folder — portable, and identical to the string
    the prompt spells so parameterize can lift it) with hidden_ok set (upload inputs
    hide behind styled drop zones)."""
    history = [_item({"upload_file": {"index": 7,
                                      "path": "/Users/x/Automation/automation/uploads/"
                                              "New_Employees_List_-_WI_LTD.csv"}},
                     element={"node_name": "input",
                              "attributes": {"type": "file", "id": "fileInput"},
                              "x_path": "//input[@type='file']"})]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    step = steps[0]
    assert step["action"] == "upload"
    assert step["value"] == "New_Employees_List_-_WI_LTD.csv"
    assert "dir" not in step and "path" not in step
    assert step["hidden_ok"] is True
    assert step["selectors"] and step.get("fingerprint")


async def test_run_steps_upload_attaches_file_and_fails_honestly(tmp_path, monkeypatch):
    """The upload executor attaches UPLOADS_DIR/<value> to the real <input type=file>
    behind the recorded drop zone — and a missing file RAISES instead of ghost-uploading
    (a ghost upload crashes the tab at save time; observed live)."""
    from playwright.async_api import async_playwright

    from automation.pipeline import files as pfiles
    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    up = tmp_path / "uploads"
    up.mkdir()
    (up / "list.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(pfiles, "UPLOADS_DIR", up)

    html = """
      <div id="zone"><span>Select or drop file</span>
        <input id="fi" type="file" style="display:none">
      </div>
    """
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(html)

        steps = [{"action": "upload", "selectors": ["css=#zone"], "value": "list.csv"}]
        out = await run_steps(page, steps, timeout_ms=3000)
        assert out["failed_at"] is None
        assert await page.evaluate(
            "document.getElementById('fi').files[0].name") == "list.csv"

        out = await run_steps(page, [{"action": "upload", "selectors": ["css=#zone"],
                                      "value": "ghost.csv"}], timeout_ms=1500)
        assert out["failed_at"] == 0
        assert "ghost.csv" in out["error"]


async def test_parameterize_lifts_upload_basename(monkeypatch):
    """The upload step's value (the basename the prompt spells) parameterizes like a
    fill value, so 'import THIS OTHER csv' replays with the new name at zero LLM."""
    from types import SimpleNamespace

    from automation.pipeline.adapt import parameterize

    class _LLM:
        async def ainvoke(self, _msgs):
            return SimpleNamespace(
                completion='{"bindings": [{"step": 0, "param": "file"}]}')

    steps = [{"action": "upload", "selectors": ["css=#zone"], "value": "list.csv",
              "hidden_ok": True}]
    template = await parameterize("Click Import, upload the file list.csv and save",
                                  steps, _LLM())
    assert template["params"] == {"file": "list.csv"}
    assert template["steps"][0]["value"] == "{{file}}"


async def test_run_steps_extract_reads_select_value_not_option_blob():
    """A <select>-anchored extract must report the SELECTED option ('Male'), never the
    concatenated option labels ('Random Male Female' — the observed live failure), on
    BOTH the selector path and the query re-find path."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    html = """
      <label for="gender">Gender</label>
      <select id="gender" name="gender">
        <option value="r">Random</option>
        <option value="m" selected>Male</option>
        <option value="f">Female</option>
      </select>
    """
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(html)

        # Selector-anchored: the reader chain prefers the control's current value.
        out = await run_steps(page, [{"action": "extract", "label": "gender",
                                      "selectors": ["css=#gender"]}], timeout_ms=3000)
        assert out["failed_at"] is None
        assert out["extracted"]["gender"] == "Male"

        # Query path (selectors dead): the static-text finder reaches the select through
        # its option text — OPTION children must not knock it out of the deepest-only
        # filter — and still reports the selected value.
        out = await run_steps(page, [{"action": "extract", "label": "gender",
                                      "selectors": ["css=#gone"],
                                      "query": "male"}], timeout_ms=1500)
        assert out["failed_at"] is None
        assert out["extracted"]["gender"] == "Male"


async def test_run_steps_extract_reads_fresh_value_through_stale_text_anchor():
    """The fresh-data regression from the live run: an extract recorded on a generated
    value anchors on text=\"<old value>\" — dead the moment the page regenerates. The
    positional xpath fallback must re-read whatever the same slot shows NOW."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content('<div class="address"><h3>Peyton Mackay</h3></div>')
        steps = [{"action": "extract", "label": "generated_name",
                  "selectors": ['text="Kirsty Crawford"',      # last run's value: dead
                                "xpath=/html/body/div/h3"],    # the positional anchor
                  "query": "kirsty crawford"}]                  # also last run's value
        out = await run_steps(page, steps, timeout_ms=2500)
        assert out["failed_at"] is None
        assert out["extracted"] == {"generated_name": "Peyton Mackay"}


async def test_run_steps_extract_reads_fresh_text_and_fails_on_empty():
    """The fresh-data contract: an extract step re-reads the element's CURRENT text every
    run, and an empty read fails the segment instead of reporting a hollow pass."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        steps = [{"action": "extract", "selectors": ["css=#t"], "label": "title"}]

        await page.set_content('<h2 id="t">First Title</h2>')
        out = await run_steps(page, steps, timeout_ms=3000)
        assert out["failed_at"] is None
        assert out["extracted"] == {"title": "First Title"}
        assert out["log"][0]["value"] == "First Title"

        await page.set_content('<h2 id="t">Fresh Title</h2>')
        out = await run_steps(page, steps, timeout_ms=3000)
        assert out["extracted"] == {"title": "Fresh Title"}     # re-read, not replayed

        await page.set_content('<h2 id="t" style="min-height:20px"></h2>')
        out = await run_steps(page, steps, timeout_ms=3000)
        assert out["failed_at"] == 0 and out["extracted"] == {}
        assert "no visible text" in out["error"]

        # Selector gone but the recorded query survives -> semantic re-find (RAW_FIND_JS).
        await page.set_content(
            '<button title="Acting Office result">Acting Office result</button>')
        out = await run_steps(page, [{"action": "extract", "label": "t",
                                      "selectors": ["css=#gone"],
                                      "query": "acting office"}], timeout_ms=1200)
        assert out["failed_at"] is None
        assert out["extracted"]["t"].startswith("Acting Office")


async def test_api_extract_records_ledger_and_execute_code_returns_extracted(monkeypatch):
    from automation.skills import api as api_mod
    from automation.skills.base import Skill, _execute_code

    async def fake_extract(_page, step, _timeout):
        assert step["selectors"] == ["css=#t"]
        assert step["query"] == "result title"    # anchor query flows into the step
        assert step["label"] == "title"
        return "Fresh Value", "css=#t", None
    monkeypatch.setattr(api_mod, "_extract_value", fake_extract)

    api = api_mod.SkillApi(page=None, anchors={
        "t": {"selectors": ["css=#t"], "query": "result title"}})
    value = await api.extract("t", "title")
    assert value == "Fresh Value"
    assert api.extracted == {"title": "Fresh Value"}
    assert api.log[-1]["action"] == "extract" and api.log[-1]["handle"] == "t"
    assert api.log[-1]["value"] == "Fresh Value"

    code = "async def run(api):\n    await api.extract('t', 'title')\n"
    skill = Skill(sid="s", body="code", code=code,
                  anchors={"t": {"selectors": ["css=#t"], "query": "result title"}})
    out = await _execute_code(skill, page=None, timeout_ms=1000)
    assert out["failed_at"] is None
    assert out["extracted"] == {"title": "Fresh Value"}


_SCOPED_PICK = 'css=[id$="-listbox"] >> text="existing employee"'


def _select_option_page(typed):
    """A minimal page double for select_option: keyboard + settle, no locator — menu
    detection then reports None (unprobeable) and the legacy type-first behavior holds."""
    from types import SimpleNamespace

    class _Keyboard:
        async def type(self, text, delay=0):
            typed.append(text)

    async def _wait(_ms):
        return None

    return SimpleNamespace(keyboard=_Keyboard(), wait_for_timeout=_wait)


def test_merge_extract_keeps_colliding_labels_and_dedups_retries():
    """Label collisions must keep BOTH values (the live loss: the generated NAME was
    extracted as 'identity_block', the ADDRESS extract reused the label and clobbered
    it, and every replay fed consumers a nameless identity). Re-reads of an
    already-stored value stay single (agent retries must not multiply keys)."""
    from automation.pipeline.script_compile import merge_extract

    store: dict = {}
    assert merge_extract(store, "identity_block", "Haiden Christie") == "identity_block"
    assert merge_extract(store, "identity_block", "28 Caerfai Bay Road") == "identity_block_2"
    assert merge_extract(store, "identity_block", "PE34 3QZ") == "identity_block_3"
    assert merge_extract(store, "identity_block", "Haiden Christie") == "identity_block"
    assert merge_extract(store, "identity_block", " 28 Caerfai Bay Road ") == "identity_block_2"
    assert store == {"identity_block": "Haiden Christie",
                     "identity_block_2": "28 Caerfai Bay Road",
                     "identity_block_3": "PE34 3QZ"}


async def test_api_extract_label_collision_keeps_both_values(monkeypatch):
    """The committed fakenamegenerator skill's exact shape: extract(name) and
    extract(address) share the label 'identity_block'. Replay must surface BOTH — this
    heals the existing entry in place, no re-authoring needed."""
    from automation.skills import api as api_mod

    values = iter(["Haiden Christie", "28 Caerfai Bay Road TERRINGTON", "April 23, 1963"])

    async def fake_extract(_page, step, _timeout):
        return next(values), step["selectors"][0], None
    monkeypatch.setattr(api_mod, "_extract_value", fake_extract)

    api = api_mod.SkillApi(page=None, anchors={
        "name": {"selectors": ["css=#n"]}, "addr": {"selectors": ["css=#a"]},
        "dob": {"selectors": ["css=#d"]}})
    await api.extract("name", "identity_block")
    await api.extract("addr", "identity_block")
    await api.extract("dob", "date_of_birth")
    assert api.extracted == {"identity_block": "Haiden Christie",
                             "identity_block_2": "28 Caerfai Bay Road TERRINGTON",
                             "date_of_birth": "April 23, 1963"}


async def test_api_select_option_reopens_closed_menu(monkeypatch):
    """When no option resolves (the menu closed under the type — the observed
    'no unique candidate matched: role=option[...]' replay failure), select_option must
    re-click the opener recorded by the previous api.click and retry the type+pick pair
    once. Every pick candidate is menu-scoped — a bare page-wide text match must never
    appear in the ladder (it would click a same-worded row cell when the menu is gone)."""
    from automation.skills import api as api_mod

    typed, clicks, fails = [], [], {"n": 0}

    async def fake_click(_page, step, _timeout):
        sel = (step.get("selectors") or [""])[0]
        clicks.append(sel)
        if step.get("expect_text") and fails["n"] == 0:   # the option pick, first try
            fails["n"] = 1
            raise RuntimeError("no unique candidate matched")
        if step.get("expect_text"):
            bare = f'text="{step["expect_text"]}"'
            assert bare not in (step.get("selectors") or []), \
                "page-wide text candidate must not be in the pick ladder"
        return sel, None
    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)

    page = _select_option_page(typed)
    api = api_mod.SkillApi(page=page, anchors={"opener": {"selectors": ["css=#opener"]}})
    await api.click("opener")                       # records the opener as _last_click
    await api.select_option("existing employee")

    # opener click -> option click FAILS -> opener re-click -> re-type -> option click OK.
    assert clicks == ["css=#opener", _SCOPED_PICK, "css=#opener", _SCOPED_PICK]
    assert typed == ["existing employee", "existing employee"]
    assert api.log[-1]["action"] == "select_option"
    assert api.log[-1]["used"] == _SCOPED_PICK


async def test_api_select_option_preopens_menu_and_skips_blind_type(monkeypatch):
    """A detectably-CLOSED menu (the observed replay failure: the recorded opener click
    landed elsewhere, so typing went into whatever had focus) makes select_option
    re-click the opener BEFORE typing — and when the menu still is not open, skip the
    type-to-filter entirely instead of typing blind."""
    from types import SimpleNamespace

    from automation.skills import api as api_mod

    typed, clicks = [], []

    async def fake_click(_page, step, _timeout):
        sel = (step.get("selectors") or [""])[0]
        clicks.append(sel)
        return sel, None
    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)

    class _Keyboard:
        async def type(self, text, delay=0):
            typed.append(text)

    class _NoMenuLocator:
        async def count(self):
            return 0

        def nth(self, _n):
            raise AssertionError("nth() on an empty locator")

    async def _wait(_ms):
        return None

    page = SimpleNamespace(keyboard=_Keyboard(), wait_for_timeout=_wait,
                           locator=lambda _sel: _NoMenuLocator())
    api = api_mod.SkillApi(page=page, anchors={"opener": {"selectors": ["css=#opener"]}})
    await api.click("opener")
    await api.select_option("Submitted")

    # opener (api.click) -> menu closed: opener PRE-click -> still closed: NO typing ->
    # pick resolves via its menu-scoped candidates.
    assert clicks == ["css=#opener", "css=#opener",
                      'css=[id$="-listbox"] >> text="Submitted"']
    assert typed == []


# --------------------- value-anchored clicks (wrong-row guard) ---------------------
# The live failure: replaying "search and select {{business}}" with business=FOOD LIMITED
# clicked FUNFOOD LIMITED — the results listed FUNFOOD first, "FOOD LIMITED" is a literal
# substring of "FUNFOOD LIMITED", and the ladder's fallbacks (stale recorded href,
# positional row xpath, last-candidate first-visible concession) know nothing about the
# value. A click whose selectors carried a {{param}} now verifies every acted-on
# candidate is actually NAMED that value.


def test_names_value_token_subsequence():
    from automation.pipeline.script_compile import _names_value

    assert _names_value("FOOD LIMITED", "FOOD LIMITED")
    assert _names_value("FOOD LIMITED 0123 Monthly", "food limited")   # row concatenation
    assert not _names_value("FUNFOOD LIMITED", "FOOD LIMITED")         # the live bug
    assert not _names_value("FUNFOOD LIMITED 0123", "FOOD LIMITED")
    assert _names_value("  Food   Limited  ", "FOOD LIMITED")          # whitespace noise
    assert not _names_value("LIMITED FOOD", "FOOD LIMITED")            # order matters
    assert _names_value("anything", "")                                # no value: no gate


def test_instantiate_stamps_expect_text_on_value_clicks():
    from automation.pipeline.adapt import instantiate

    template = {
        "params": {"business": "WI LTD"},
        "steps": [
            {"action": "fill", "selectors": ['css=[placeholder="Search"]'],
             "value": "{{business}}"},
            {"action": "click", "selectors": ['role=link[name="{{business}}"]',
                                              'css=a[href="/paye/clients/old-id/x"]',
                                              'xpath=/html/body/div[1]/a']},
            {"action": "click", "selectors": ['role=link[name="Employees"]']},
            {"action": "find_click", "text": "{{business}}"},
        ],
    }
    steps = instantiate(template, {"business": "FOOD LIMITED"})
    assert steps[0].get("expect_text") is None          # fill: value ≠ target name
    assert steps[1]["expect_text"] == "FOOD LIMITED"    # value-anchored click
    assert steps[2].get("expect_text") is None          # tokenless click untouched
    assert steps[3]["verify_name"] is True              # semantic click by value


def test_anchor_expect_text_flows_to_click_steps_only(monkeypatch):
    from automation.skills.api import SkillApi
    from automation.skills.base import _substituted_anchors

    anchors = {
        "business-row": {"selectors": ['role=link[name="{{business}}"]',
                                       "xpath=/html/body/div[1]/a"]},
        "save": {"selectors": ['css=[id="btn-save-1"]']},
    }
    concrete = _substituted_anchors(anchors, {"business": "FOOD LIMITED"})
    assert concrete["business-row"]["expect_text"] == "FOOD LIMITED"
    assert "expect_text" not in concrete["save"]

    api = SkillApi(page=None, anchors=concrete)
    assert api._step_for("business-row", expect=True)["expect_text"] == "FOOD LIMITED"
    assert "expect_text" not in api._step_for("business-row")          # fill/select path
    assert "expect_text" not in api._step_for("save", expect=True)


async def test_value_anchored_click_picks_named_row_not_first(tmp_path):
    """Two search-result rows, wrong one first: every fallback that resolves (stale href,
    row-1 xpath, ambiguous substring) must be rejected in favor of the row actually
    named the value — and when the named row is absent, the step must FAIL rather than
    click a confident wrong match."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    html = """
      <div id="results">
        <a href="/clients/fun/x" onclick="window.__picked='FUNFOOD'; return false">
          <div>FUNFOOD LIMITED</div><div>0456</div></a>
        <a href="/clients/food/x" onclick="window.__picked='FOOD'; return false">
          <div>FOOD LIMITED</div><div>0123</div></a>
      </div>
    """
    step = {"action": "click",
            "selectors": ['css=a[href="/clients/old-recorded-id/x"]',   # stale href
                          'xpath=/html/body/div/a[1]',                  # positional row 1
                          "css=#results a"],                            # ambiguous last
            "expect_text": "FOOD LIMITED"}
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(html)
        out = await run_steps(page, [step], timeout_ms=2500)
        assert out["failed_at"] is None
        assert await page.evaluate("window.__picked") == "FOOD"

        # Without the stamp, the same ladder documents today's wrong-row behavior.
        await page.set_content(html)
        bare = {k: v for k, v in step.items() if k != "expect_text"}
        out = await run_steps(page, [bare], timeout_ms=2500)
        assert await page.evaluate("window.__picked") == "FUNFOOD"

        # Named row absent -> the step fails honestly (agent recovery), never a
        # confident wrong click. (set_content keeps the window object; reset the probe.)
        await page.set_content("""
          <div id="results">
            <a href="/clients/fun/x" onclick="window.__picked='FUNFOOD'; return false">
              <div>FUNFOOD LIMITED</div><div>0456</div></a>
          </div>
        """)
        await page.evaluate("window.__picked = null")
        out = await run_steps(page, [dict(step)], timeout_ms=1500)
        assert out["failed_at"] == 0
        assert 'none named "FOOD LIMITED"' in out["error"] or "no unique candidate" in out["error"]
        assert await page.evaluate("window.__picked") is None


async def test_find_click_verify_name_refuses_wrong_named_match(tmp_path):
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        # Only a wrong-named candidate exists: verified find_click must refuse it...
        await page.set_content("""
          <button onclick="window.__picked='FUNFOOD'">FUNFOOD LIMITED</button>
        """)
        out = await run_steps(page, [{"action": "find_click", "text": "FOOD LIMITED",
                                      "verify_name": True}], timeout_ms=1500)
        assert out["failed_at"] == 0 and "not named" in out["error"]
        assert await page.evaluate("window.__picked") is None
        # ...while the exact-ranked right candidate clicks normally.
        await page.set_content("""
          <button onclick="window.__picked='FUN'">FUNFOOD LIMITED</button>
          <button onclick="window.__picked='FOOD'">FOOD LIMITED</button>
        """)
        out = await run_steps(page, [{"action": "find_click", "text": "FOOD LIMITED",
                                      "verify_name": True}], timeout_ms=2500)
        assert out["failed_at"] is None
        assert await page.evaluate("window.__picked") == "FOOD"


def test_discovery_loop_notice_fires_on_pure_discovery_streaks():
    from automation.pipeline.runner import discovery_loop_notice

    def acts(*names):
        return [{n: {}, "interacted_element": None} for n in names]

    # A streak of 4 read-only discovery actions -> nudge; re-fires at 8, silent between.
    assert discovery_loop_notice(acts("click", "list_actions", "search_page",
                                      "capped_scroll", "find_elements")) is not None
    assert discovery_loop_notice(acts("list_actions", "search_page", "capped_scroll")) is None
    assert discovery_loop_notice(
        acts("list_actions", "search_page", "capped_scroll", "find_elements",
             "list_actions")) is None                       # streak 5: between multiples
    assert discovery_loop_notice(
        acts(*(["list_actions"] * 8))) is not None          # streak 8: re-fires
    # Any real interaction resets the streak.
    assert discovery_loop_notice(acts("list_actions", "search_page", "click",
                                      "list_actions", "search_page")) is None
    notice = discovery_loop_notice(acts(*(["search_page"] * 4)))
    assert "ALREADY CLICKED" in notice and "find_by_text" in notice


# ------------------------- custom-combobox select_dropdown picks -------------------------
# The combobox branch of select_dropdown records the OPTION element it clicked; compile
# must turn that into the same replayable by-label steps as a recorded option click
# instead of dropping the pick (the old warning path replayed the form with defaults).


def test_custom_combobox_pick_compiles_to_type_and_by_label_click(tmp_path):
    element = {"node_name": "DIV", "ax_name": "Existing employee",
               "attributes": {"id": "react-select-22-option-5", "role": "option"}}
    history = [_item(
        {"select_dropdown": {"index": 7, "text": "Existing employee"}},
        result=[{"extracted_content": "Selected 'Existing employee'"}],
        element=element,
    )]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [
        {"action": "type", "text": "Existing employee",
         "field_id": "react-select-22"},
        {"action": "click", "selectors": [
            'role=option[name="Existing employee"]',
            'text="Existing employee"',
            'css=[id$="-option-0"]',
        ], "expect_text": "Existing employee"},
    ]


def test_generic_role_option_pick_compiles_to_by_label_click(tmp_path):
    element = {"node_name": "DIV", "ax_name": "Weekly",
               "attributes": {"id": "freq-ao-opt-1", "role": "option"}}
    history = [_item(
        {"select_dropdown": {"index": 3, "text": "Weekly"}},
        result=[{"extracted_content": "Selected 'Weekly'"}],
        element=element,
    )]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    # type-then-pick (collapsed by codegen into api.select_option): the menu's list is
    # filtered into existence by the typing, so the option is there to be clicked.
    assert steps == [
        {"action": "type", "text": "Weekly"},
        {"action": "click", "selectors": ['role=option[name="Weekly"]', 'text="Weekly"'],
         "expect_text": "Weekly"},
    ]


def test_native_select_pick_still_compiles_to_select_step(tmp_path):
    element = {"node_name": "SELECT", "ax_name": "Country",
               "attributes": {"id": "country"}}
    history = [_item(
        {"select_dropdown": {"index": 2, "text": "United Kingdom"}},
        result=[{"extracted_content": "Selected"}],
        element=element,
    )]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    assert steps[0]["action"] == "select"
    assert steps[0]["value"] == "United Kingdom"


# ---------------------- repeat-aware collapse + indexed runs ----------------------
# (2026-08-12: _push_step used to DROP wait-separated repeat clicks — the recorded
# 5x/14x Save & Next counters compiled to ONE click. Repeats now absorb into a count;
# id-indexed grid runs collapse to one click_indexed template step.)


def _btn(text="Save & Next"):
    return {"node_name": "button", "ax_name": text,
            "attributes": {"id": "btnSave"}, "x_path": "html/body/div[1]/button"}


def _chk(idv):
    return {"node_name": "div", "ax_name": None,
            "attributes": {"id": idv, "role": "checkbox",
                           "data-automationid": "DetailsRowCheck"},
            "x_path": "html/body/div[2]/div/div"}


def test_wait_separated_repeat_clicks_absorb_into_count(tmp_path):
    history = []
    for _ in range(5):
        history.append(_item({"click": {"index": 1}}, element=_btn()))
        history.append(_item({"wait": {"seconds": 1}}))
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "wait"]  # trailing settle survives
    assert steps[0]["count"] == 5
    assert steps[0]["repeat_wait_s"] == 1.0


def test_adjacent_same_target_click_still_collapses_as_retry(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_btn()),
               _item({"click": {"index": 1}}, element=_btn())]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click"]
    assert "count" not in steps[0]


def test_repeat_hint_pins_single_cluster_count(tmp_path):
    history = []
    for _ in range(4):
        history.append(_item({"click": {"index": 1}}, element=_btn()))
        history.append(_item({"wait": {"seconds": 2}}))
    from automation.pipeline.script_compile import save_steps
    out = tmp_path / "steps.json"
    steps = save_steps(_write(tmp_path, history), out, emit_start_goto=False,
                       repeat_hint=5)
    assert steps[0]["count"] == 5          # wording "exactly 5 clicks" wins
    assert steps[0]["repeat_wait_s"] == 2.0


def test_repeat_hint_never_invents_a_cluster(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_btn())]
    from automation.pipeline.script_compile import save_steps
    out = tmp_path / "steps.json"
    steps = save_steps(_write(tmp_path, history), out, emit_start_goto=False,
                       repeat_hint=5)
    assert "count" not in steps[0]


def test_repeat_hint_from_wording():
    from automation.pipeline.script_compile import repeat_hint_from_wording
    assert repeat_hint_from_wording(
        "Click Save & Next for the next 5 employees: exactly 5 clicks, waiting") == 5
    assert repeat_hint_from_wording("exactly 14 more clicks, waiting for") == 14
    assert repeat_hint_from_wording("click Save & Next exactly 3 times") == 3
    assert repeat_hint_from_wording("click Save and continue") is None


def test_indexed_click_run_collapses_and_normalizes(tmp_path):
    # The agent skipped index 7 and ticked 20 to compensate (observed live 2026-08-12):
    # normalization yields the contiguous span 0..19.
    observed = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    history = [_item({"click": {"index": 1}},
                     element=_chk(f"row{26102 + 13 * n}-{n}-checkbox"))
               for n in observed]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    s = steps[0]
    assert s["action"] == "click_indexed"
    assert s["selector_template"] == 'css=[id$="-{n}-checkbox"]'
    assert s["start"] == 0 and s["count"] == 20
    assert s["fingerprint"]["attrs"]["data-automationid"] == "DetailsRowCheck"


def test_indexed_run_swallows_interleaved_scrolls_and_waits(tmp_path):
    history = []
    for n in range(6):
        history.append(_item({"click": {"index": 1}},
                             element=_chk(f"row{100 + n}00-{n}-checkbox")))
        if n == 2:
            history.append(_item({"capped_scroll": {"down": True, "pages": 0.5}}))
        if n == 4:
            history.append(_item({"wait": {"seconds": 1}}))
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click_indexed"]
    assert steps[0]["start"] == 0 and steps[0]["count"] == 6


def test_indexed_run_preserves_nonzero_start(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_chk(f"row9{n}1-{n}-checkbox"))
               for n in range(5, 16)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps[0]["action"] == "click_indexed"
    assert steps[0]["start"] == 5 and steps[0]["count"] == 11


def test_short_or_shapeless_runs_stay_individual_clicks(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_chk(f"row111-{n}-checkbox"))
               for n in range(2)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "click"]
    history = [_item({"click": {"index": 1}}, element=_btn()),
               _item({"click": {"index": 1}}, element=_chk("plainid")),
               _item({"click": {"index": 1}}, element=_btn("Other"))]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert all(s["action"] == "click" for s in steps)


def test_volatile_prefix_id_gets_stable_suffix_selector():
    from automation.pipeline.script_compile import _selectors
    sels = _selectors(_chk("row26102-3-checkbox"))
    assert 'css=[id$="-3-checkbox"]' in sels
    assert not any("row26102" in s for s in sels)
    assert 'css=[data-automationid="DetailsRowCheck"]' in sels


# ---------------------- repeat-aware collapse + indexed runs ----------------------
# (2026-08-12: _push_step used to DROP wait-separated repeat clicks — the recorded
# 5x/14x Save & Next counters compiled to ONE click. Repeats now absorb into a count;
# id-indexed grid runs collapse to one click_indexed template step.)


def _btn(text="Save & Next"):
    return {"node_name": "button", "ax_name": text,
            "attributes": {"id": "btnSave"}, "x_path": "html/body/div[1]/button"}


def _chk(idv):
    return {"node_name": "div", "ax_name": None,
            "attributes": {"id": idv, "role": "checkbox",
                           "data-automationid": "DetailsRowCheck"},
            "x_path": "html/body/div[2]/div/div"}


def test_wait_separated_repeat_clicks_absorb_into_count(tmp_path):
    history = []
    for _ in range(5):
        history.append(_item({"click": {"index": 1}}, element=_btn()))
        history.append(_item({"wait": {"seconds": 1}}))
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "wait"]  # trailing settle survives
    assert steps[0]["count"] == 5
    assert steps[0]["repeat_wait_s"] == 1.0


def test_adjacent_same_target_click_still_collapses_as_retry(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_btn()),
               _item({"click": {"index": 1}}, element=_btn())]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click"]
    assert "count" not in steps[0]


def test_repeat_hint_pins_single_cluster_count(tmp_path):
    history = []
    for _ in range(4):
        history.append(_item({"click": {"index": 1}}, element=_btn()))
        history.append(_item({"wait": {"seconds": 2}}))
    from automation.pipeline.script_compile import save_steps
    out = tmp_path / "steps.json"
    steps = save_steps(_write(tmp_path, history), out, emit_start_goto=False,
                       repeat_hint=5)
    assert steps[0]["count"] == 5          # wording "exactly 5 clicks" wins
    assert steps[0]["repeat_wait_s"] == 2.0


def test_repeat_hint_never_invents_a_cluster(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_btn())]
    from automation.pipeline.script_compile import save_steps
    out = tmp_path / "steps.json"
    steps = save_steps(_write(tmp_path, history), out, emit_start_goto=False,
                       repeat_hint=5)
    assert "count" not in steps[0]


def test_repeat_hint_from_wording():
    from automation.pipeline.script_compile import repeat_hint_from_wording
    assert repeat_hint_from_wording(
        "Click Save & Next for the next 5 employees: exactly 5 clicks, waiting") == 5
    assert repeat_hint_from_wording("exactly 14 more clicks, waiting for") == 14
    assert repeat_hint_from_wording("click Save & Next exactly 3 times") == 3
    assert repeat_hint_from_wording("click Save and continue") is None


def test_indexed_click_run_collapses_and_normalizes(tmp_path):
    # The agent skipped index 7 and ticked 20 to compensate (observed live 2026-08-12):
    # normalization yields the contiguous span 0..19.
    observed = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    history = [_item({"click": {"index": 1}},
                     element=_chk(f"row{26102 + 13 * n}-{n}-checkbox"))
               for n in observed]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    s = steps[0]
    assert s["action"] == "click_indexed"
    assert s["selector_template"] == 'css=[id$="-{n}-checkbox"]'
    assert s["start"] == 0 and s["count"] == 20
    assert s["fingerprint"]["attrs"]["data-automationid"] == "DetailsRowCheck"


def test_indexed_run_swallows_interleaved_scrolls_and_waits(tmp_path):
    history = []
    for n in range(6):
        history.append(_item({"click": {"index": 1}},
                             element=_chk(f"row{100 + n}00-{n}-checkbox")))
        if n == 2:
            history.append(_item({"capped_scroll": {"down": True, "pages": 0.5}}))
        if n == 4:
            history.append(_item({"wait": {"seconds": 1}}))
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click_indexed"]
    assert steps[0]["start"] == 0 and steps[0]["count"] == 6


def test_indexed_run_preserves_nonzero_start(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_chk(f"row9{n}1-{n}-checkbox"))
               for n in range(5, 16)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps[0]["action"] == "click_indexed"
    assert steps[0]["start"] == 5 and steps[0]["count"] == 11


def test_short_or_shapeless_runs_stay_individual_clicks(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_chk(f"row111-{n}-checkbox"))
               for n in range(2)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "click"]
    history = [_item({"click": {"index": 1}}, element=_btn()),
               _item({"click": {"index": 1}}, element=_chk("plainid")),
               _item({"click": {"index": 1}}, element=_btn("Other"))]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert all(s["action"] == "click" for s in steps)


def test_volatile_prefix_id_gets_stable_suffix_selector():
    from automation.pipeline.script_compile import _selectors
    sels = _selectors(_chk("row26102-3-checkbox"))
    assert 'css=[id$="-3-checkbox"]' in sels
    assert not any("row26102" in s for s in sels)
    assert 'css=[data-automationid="DetailsRowCheck"]' in sels


# ------------------- extract values stay WHOLE (2026-08-13) -------------------
# Auto-splitting labeled lines into per-field keys was removed: the keys came from page
# CONTENT ("PILTON" from an address block), so a binding keyed on one could never
# resolve on the next identity. Values are stored whole and sliced by binding
# transforms (line index / date reformat) instead.


def test_merge_extract_stores_blocks_whole_without_content_derived_keys():
    from automation.pipeline.script_compile import merge_extract
    store = {}
    block = "Ayaan Campbell\n73 Tadcaster Rd\nPILTON\nPE8 9ZS"
    merge_extract(store, "identity_block", block)
    assert list(store) == ["identity_block"]
    assert store["identity_block"] == block          # line structure preserved
    merge_extract(store, "dob_block", "Birthday\nMarch 2, 1979")
    assert list(store) == ["identity_block", "dob_block"]


def test_extract_normalizes_identically_when_authored_and_replayed():
    """Authoring and replay must produce the SAME shape of value: bindings slice fields
    by LINE POSITION, so a replay that flattened the block made every line-transform
    binding unresolvable and the form-fill segment re-authored with the LLM every run
    (runs 20260813_153838 vs _155450 — authored kept the lines, replay lost them)."""
    from automation.pipeline.script_compile import normalize_block_text
    raw = "  Arron McIntosh \n\n 41 Gloucester   Road \n CILRHEDYN \n SA35 5WS  "
    assert normalize_block_text(raw) == (
        "Arron McIntosh\n41 Gloucester Road\nCILRHEDYN\nSA35 5WS")
    # Single-line values are unaffected; empties stay empty.
    assert normalize_block_text("  one   line  ") == "one line"
    assert normalize_block_text("") == ""


async def test_replayed_extract_keeps_block_lines(monkeypatch):
    """The replay reader itself — a block read back through _extract_value keeps its
    line structure (this is the value bindings resolve against)."""
    from automation.pipeline import script_compile as sc

    class _Loc:
        async def evaluate(self, expr):
            return ""                       # not a form control

        async def inner_text(self):
            return "Luis Stevenson\n67 Wressle Road\nPLESHEY\nCM3 2SE"

        async def text_content(self):
            return ""

        async def input_value(self):
            return ""

    async def fake_resolve(page, step, timeout_ms, require_editable=False):
        return _Loc(), "xpath=/html/body/div[2]", None

    monkeypatch.setattr(sc, "_resolve", fake_resolve)
    value, used, _ = await sc._extract_value(
        None, {"selectors": ["xpath=/html/body/div[2]"], "label": "name"}, 500)
    assert value.splitlines() == ["Luis Stevenson", "67 Wressle Road", "PLESHEY",
                                  "CM3 2SE"]
    assert used.startswith("xpath=")


def test_custom_combobox_pick_replays_by_value_not_a_bare_option_click(tmp_path):
    """select_dropdown on a non-react-select combobox used to compile to a BARE option
    click. The list is only populated once the filter text is typed, so replay clicked
    an option that did not exist yet — the pay-forecast employee pick failed on two
    consecutive runs with `role=option[name="Abdullah Reilly"] -> no match`
    (20260813_161952, 20260814_103913). Emitting the type+option pair makes codegen
    collapse it into api.select_option(value): open the menu, filter, pick, verify."""
    option_el = {"node_name": "div", "ax_name": "Riley Moore",
                 "attributes": {"role": "option", "id": "emp-opt-3"},
                 "x_path": "html/body/div[3]/div[2]"}
    history = [_item({"select_dropdown": {"index": 15201, "text": "Riley Moore"}},
                     result=[{"extracted_content": "Selected"}], element=option_el)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert [s["action"] for s in steps] == ["type", "click"]
    assert steps[0]["text"] == "Riley Moore"
    # Exact location leads (name-guarded below); the by-name candidates back it up.
    assert steps[1]["selectors"][0] == "xpath=/html/body/div[3]/div[2]"
    assert 'role=option[name="Riley Moore"]' in steps[1]["selectors"]
    assert steps[1]["expect_text"] == "Riley Moore"
    # codegen collapses the pair into the by-value primitive
    code, _ = transpile("sid", steps)
    assert "await api.select_option('Riley Moore')" in code


def test_send_keys_text_compiles_to_typing_not_a_key_press(tmp_path):
    """`send_keys` carries EITHER a key chord or raw text. Compiling text as a key press
    made replay call keyboard.press("4000") -> Playwright `Unknown key: "4000"`, which
    killed the net-to-gross replay (run 20260814_105247, subtask 4)."""
    history = [
        _item({"send_keys": {"keys": "4000"}}),
        _item({"send_keys": {"keys": "Enter"}}),
        _item({"send_keys": {"keys": "Control+a"}}),
        _item({"send_keys": {"keys": "ArrowDown"}}),
        _item({"send_keys": {"keys": "Riley Moore"}}),
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [
        {"action": "type", "text": "4000"},
        {"action": "press", "keys": "Enter"},
        {"action": "press", "keys": "Control+a"},
        {"action": "press", "keys": "ArrowDown"},
        {"action": "type", "text": "Riley Moore"},
    ]


def test_combobox_pick_recorded_on_the_input_emits_an_opener_click(tmp_path):
    """select_dropdown records the COMBOBOX INPUT (react-select-13-input), because the
    tool opens the menu, types and picks internally — one action, no separate opener
    click. Compiling only the pick left replay hunting for options in a CLOSED menu
    (`role=option[name="Male"] -> no match`, run 20260814_113403). The opener click is
    part of the pick and must be emitted with it."""
    combobox = {"node_name": "INPUT", "ax_name": "Gender",
                "attributes": {"id": "react-select-13-input", "role": "combobox"},
                "x_path": "html/body/form/div[2]/input"}
    history = [_item({"select_dropdown": {"index": 2689, "text": "Male"}},
                     result=[{"extracted_content": "Selected"}], element=combobox)]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert [s["action"] for s in steps] == ["click", "type", "click"]
    # 1) open the menu via the combobox itself
    assert steps[0]["selectors"][0] == "xpath=/html/body/form/div[2]/input"
    # 2) + 3) filter and pick by value — the option anchor must NOT reuse the input's
    # xpath (that is the box, not the row).
    assert steps[1] == {"action": "type", "text": "Male"}
    assert steps[2]["selectors"][0] == 'role=option[name="Male"]'
    assert not any(s.startswith("xpath=") for s in steps[2]["selectors"])

    code, _ = transpile("sid", steps)
    assert "await api.click(" in code and "await api.select_option('Male')" in code
