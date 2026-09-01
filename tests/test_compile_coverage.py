"""Compile coverage fixes surfaced by live NPS runs: scrolls must compile, a
metadata-less find_by_text click must not vanish, save_history's metadata drop must be
repaired, and a zero-step script must never be committed."""
import json
import pathlib
from types import SimpleNamespace

import pytest

from automation.pipeline import subtask_store as ss
from automation.pipeline.runner import restore_result_metadata
from automation.pipeline import script_compile as sc
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


def test_index_miss_actions_are_never_compiled(tmp_path):
    """An indexed action whose element index had already fallen out of the selector map
    returns "not available" INSTEAD of acting — but it still carries the interacted
    element captured from the pre-action DOM, which is why it used to compile. Recording
    step 0 of the Send Email segment (038d896c619d0a0c) misclicked the panel's close-X;
    the #mailbtn click queued behind it never ran, yet became a step, so every replay
    clicked a Send button its own previous step had just removed from the page."""
    miss = ("Element index 5913 not available - page may have changed. "
            "Try refreshing browser state.")
    history = [{
        "state": {"url": "http://app/x", "interacted_element": [
            {"node_name": "BUTTON", "ax_name": None, "attributes": {"type": "button"},
             "x_path": "html/body/div[2]/div/div[1]/div/button"},
            {"node_name": "BUTTON", "ax_name": "Send", "attributes": {"id": "mailbtn"},
             "x_path": "html/body/div[2]/div/div[3]/span/button[1]"},
        ]},
        "model_output": {"action": [
            {"click": {"index": 5813}},
            {"click": {"index": 5913}},
        ]},
        "result": [
            {"extracted_content": 'Clicked button ""'},
            {"extracted_content": miss},
        ],
    }]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert len(steps) == 1
    assert steps[0]["action"] == "click"
    assert "mailbtn" not in json.dumps(steps)
    # The same message on the `error` channel, and on a non-click indexed tool, also drops.
    history[0]["model_output"]["action"][1] = {"select_dropdown": {"index": 5913,
                                                                  "text": "no-reply"}}
    history[0]["result"][1] = {"error": miss}
    assert len(compile_recording(_write(tmp_path, history), emit_start_goto=False)) == 1


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
    # (`scroll` here was `capped_scroll` until that tool was dropped for the built-in.)
    assert discovery_loop_notice(acts("click", "list_actions", "search_page",
                                      "scroll", "find_elements")) is not None
    assert discovery_loop_notice(acts("list_actions", "search_page", "scroll")) is None
    assert discovery_loop_notice(
        acts("list_actions", "search_page", "scroll", "find_elements",
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


async def test_a_select_inside_the_captured_block_reads_as_its_selected_option():
    """Run 20260818_103055 subtask 1: extract_data('Gender') on fakenamegenerator returned
    the option list of every dropdown on the page —

        'Gender Random Male Female Name set American Arabic ... Country Australia ...'

    — and the agent burned 4 more steps trying to get the value out of it. The probe's
    <select> rule only fired when the select WAS the match: query 'Gender' matches the
    <label>, whose text equals the query, so the zero-information-gain expansion climbs to
    the form block and takes it with plain innerText, which for a closed <select> is every
    option. The only query that returned the value was the one that already contained it
    ('Male'), which is backwards for extraction — and extract_data commits a REPLAYABLE
    step, so the blob became the value future replays would re-read and bind."""
    import json as _json

    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import RAW_TEXT_FIND_JS
    from tests.test_heal_promotion import _launch

    # The live page's shape: three labelled selects sharing one block.
    form = """
      <h3>Your Randomly Generated Identity</h3>
      <div class="form">
        <label for="g">Gender</label>
        <select id="g"><option>Random</option><option selected>Male</option>
          <option>Female</option></select>
        <label for="n">Name set</label>
        <select id="n"><option>American</option><option selected>Scottish</option>
          <option>Klingon</option></select>
        <label for="c">Country</label>
        <select id="c"><option>Australia</option>
          <option selected>United Kingdom</option></select>
      </div>
    """
    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(form)

        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["gender"]))
        got = raw["name"]
        assert raw["expanded"] is True          # the label alone teaches nothing
        assert "Male" in got                    # ...so the block answers, with the VALUES
        for unchosen in ("Random", "Female", "American", "Klingon", "Australia"):
            assert unchosen not in got, f"{unchosen!r} is not chosen: {got!r}"
        assert "Scottish" in got and "United Kingdom" in got   # the other two, correctly

        # The select-as-match branch is untouched.
        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["male"]))
        assert raw["name"] == "Male"

        # A block with NO select captures exactly what it captured before: the identity
        # card the aux-tab flow actually depends on.
        await page.set_content("""
          <div class="card"><h3>Harrison Sutherland</h3>
            <div>36 Hull Road</div><div>PAGLESHAM EASTEND</div><div>SS4 6HJ</div></div>
        """)
        raw = await page.evaluate(RAW_TEXT_FIND_JS % _json.dumps(["harrison", "sutherland"]))
        assert raw["expanded"] is True
        assert raw["name"] == "Harrison Sutherland 36 Hull Road PAGLESHAM EASTEND SS4 6HJ"


def test_fluent_counter_ids_are_volatile_but_app_authored_ids_are_not():
    """Run 20260818_11xx, Net-to-Gross seg: the popup fill's only two anchors were
    css=[id="TextField99"] and an xpath GATED on that same id, and the id is a Fluent
    getId() mount counter — yesterday's recording of the identical input said
    TextField69. Both anchors died ("no match" | "positional drift") and the step could
    never replay. _is_dynamic_id already feeds the selector builder AND the fingerprint
    gate; it just scored TextField99 as stable because _DYNAMIC_ID wants 3+ digits."""
    from automation.pipeline.script_compile import _is_dynamic_id

    for volatile in ("TextField99", "TextField69", "Toggle21", "Toggle2105",
                     "Dropdown4", "ComboBox7", "id__42", "SearchBox129"):
        assert _is_dynamic_id(volatile), volatile
    # btnReverseCalc10 is this very skill's WORKING anchor (the Feb-27 pencil) — an
    # app-authored id that happens to end in digits. A generic letters+digits rule would
    # swallow it and break the two click steps that replay fine today.
    for stable in ("btnReverseCalc10", "btnInvoice", "gender", "productItems", "cb2"):
        assert not _is_dynamic_id(stable), stable


def test_a_query_derived_expect_is_stamped_scattered(tmp_path):
    """find_by_text matches on a HAYSTACK (text + every descendant icon's aria-label,
    each token anywhere). When the clicked element has no accessible name of its own,
    compile records that QUERY as the step's identity — and replay's _names_value wants
    the tokens CONSECUTIVE, which a grid row can never satisfy. Mark the provenance so
    replay can verify the way the query matched."""
    row = {"node_name": "tr", "attributes": {},
           "x_path": "html/body/table/tbody/tr[11]"}
    history = [_item({"find_by_text": {"text": "Feb-27 Net to gross", "click_first": True}},
                     result=[{"extracted_content": "clicked",
                              "metadata": {"interacted_element": row}}])]
    step = compile_recording(_write(tmp_path, history), emit_start_goto=False)[0]
    assert step["expect_text"] == "Feb-27 Net to gross"
    assert step["expect_scattered"] is True

    # A real accessible name keeps the strict rule: it names ONE element, not a haystack.
    named = {"node_name": "button", "ax_name": "Save", "attributes": {"id": "btnSave"},
             "x_path": "html/body/button"}
    history = [_item({"find_by_text": {"text": "Save", "click_first": True}},
                     result=[{"extracted_content": "clicked",
                              "metadata": {"interacted_element": named}}])]
    step = compile_recording(_write(tmp_path, history), emit_start_goto=False)[0]
    assert step["expect_text"] == "Save"
    assert "expect_scattered" not in step


def test_undeclared_same_target_repeat_dissolves_to_one_click():
    """The Download-menu bug (segment e665b42d2c22fcee, 2026-08-24): the agent clicked
    Download, waited, and clicked it AGAIN (re-clicking what it thought had not
    registered). _push_step read the wait as a deliberate rhythm and fused the pair into
    repeat_click('download', 2) — which opens the menu and then shuts it, so every replay
    hunted Excel on a closed menu. Nothing caught it: the skill still downloaded
    something, so the entry sat at uses=2, fail_count=0.

    A cadence is only real when the slice ASKS for one, so an unpinned cluster dissolves
    to a single click and the recorded inter-click wait survives as settle time."""
    steps = [
        {"action": "click", "selectors": ['text="Download"'], "count": 2,
         "repeat_wait_s": 1.0},
        {"action": "click", "selectors": ['text="Excel"']},
    ]
    out = sc._apply_repeat_hint(steps, None)
    assert [s["action"] for s in out] == ["click", "wait", "click"]
    assert "count" not in out[0] and "repeat_wait_s" not in out[0]
    assert out[1]["seconds"] == 1.0

    # A declared cadence is untouched, and still pinned to the wording's number.
    counter = [{"action": "click", "selectors": ['text="Save & Next"'], "count": 4}]
    assert sc._apply_repeat_hint(counter, 5)[0]["count"] == 5
    assert sc.repeat_hint_from_wording("click Save & Next exactly 5 times") == 5
    assert sc.repeat_hint_from_wording(
        "Then click Download and select PDF, then click Download again and "
        "select Excel.") is None

    # click_indexed clusters are a different mechanism (N distinct rows) — never dissolved.
    indexed = [{"action": "click_indexed", "selector_template": 'css=[id$="-{n}"]',
                "start": 0, "count": 3}]
    assert sc._apply_repeat_hint(indexed, None) == indexed


# ---------------- shadow containment vs. shadow HOSTING (2026-08-25) ----------------
# An xpath into a shadow root can never resolve (see tests/test_shadow_dom_find.py for the
# measurement), so compile must know when an element is inside one. browser-use's listing
# is the only record — but the `|SHADOW(open)|` prefix marks a node that IS a shadow HOST
# (`is_shadow_host = any(child is a DOCUMENT_FRAGMENT)`, serializer.py:513), and every
# native <input>/<select> hosts its own USER-AGENT shadow root. Reading the prefix as
# containment marked every input on the app as unreachable-by-xpath and broke two library
# entries outright. Containment is an `Open Shadow` line ABOVE the element, at a smaller
# indent, with the shadow tree nested below it (serializer.py:1069).

_SHADOW_SM = """Interactive elements:
[Start of page]
\t[81]<button />
\t|SHADOW(open)|[300]<my-widget />
\t\tOpen Shadow
\t\t\tAmount
\t\t\t*[363]<input type=text inputmode=decimal value=\u00a30.00 />
\t\tShadow End
\tAmount
\t|SHADOW(open)|*[370]<input type=text inputmode=decimal value=\u00a30.00 />
\t*[371]<button />
\t\tSave
[End of page]"""


def test_containment_is_read_from_the_ancestors_not_the_elements_own_marker():
    from automation.pipeline.script_compile import _sm_in_shadow

    assert _sm_in_shadow(_SHADOW_SM, 363) is True    # nested under `Open Shadow`
    assert _sm_in_shadow(_SHADOW_SM, 300) is False   # the HOST is not inside its own root
    assert _sm_in_shadow(_SHADOW_SM, 370) is False   # a native input hosting a UA root
    assert _sm_in_shadow(_SHADOW_SM, 371) is False   # ...and its sibling after Shadow End
    assert _sm_in_shadow(_SHADOW_SM, 99999) is False
    assert _sm_in_shadow("", 363) is False


def test_no_element_in_the_real_library_is_treated_as_shadow_contained():
    """Fixture from live recordings rather than a hand-written belief about the markup —
    but DERIVED, never hardcoded: these files are rewritten every time a segment
    re-authors, so pinning an element index makes the test rot on the user's next run.

    Two facts, and the gap between them is the whole bug: the `|SHADOW(open)|` prefix
    fires on inputs all over this app, and NOT ONE of them is actually inside a shadow
    root (no recording contains an `Open Shadow` line at all)."""
    from automation.pipeline.script_compile import _sm_in_shadow

    marked = contained = seen = 0
    for rec_path in sorted(pathlib.Path("library").glob("*.recording.json")):
        for item in json.loads(rec_path.read_text()).get("history", []):
            sm = item.get("state_message") or ""
            for action in (item.get("model_output") or {}).get("action") or []:
                params = next(iter((action or {}).values()), None)
                idx = params.get("index") if isinstance(params, dict) else None
                if idx is None or f"[{idx}]<" not in sm:
                    continue
                seen += 1
                line = next(ln for ln in sm.splitlines() if f"[{idx}]<" in ln)
                marked += "|SHADOW(" in line
                contained += _sm_in_shadow(sm, idx)
    if not seen:
        pytest.skip("no recordings in library/ to read")
    assert marked, "expected the host marker to appear — it is what misled compile"
    assert contained == 0, f"{contained} of {seen} recorded elements read as shadow-contained"


def test_a_shadow_contained_fill_drops_the_dead_xpath(tmp_path):
    field = {"node_name": "INPUT", "x_path": "html/body/div/div/input",
             "attributes": {"type": "text", "inputmode": "decimal", "placeholder": ""}}
    item = _item({"input": {"index": 363, "text": "4000", "clear": True}},
                 element=field, state_message=_SHADOW_SM)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    fills = [s for s in steps if s["action"] == "fill"]
    assert fills, steps
    sels = fills[0]["selectors"]
    assert sels[0] == 'css=[inputmode="decimal"]'        # a real attribute leads
    assert all(not s.startswith("xpath=") for s in sels)  # the dead anchor is gone


def test_a_native_input_hosting_a_ua_shadow_root_keeps_its_xpath(tmp_path):
    """The regression: index 370 carries the same `|SHADOW(open)|` prefix as the widget
    above it, but it is a plain input in the light DOM and its recorded xpath is its
    strongest anchor. Stripping it left the app's attribute-less fields — the NI number
    box, the Net-to-Gross popup's Net amount box — with no anchor that resolves."""
    field = {"node_name": "INPUT", "x_path": "html/body/div/div/input",
             "attributes": {"type": "text", "inputmode": "decimal"}}
    item = _item({"input": {"index": 370, "text": "4000", "clear": True}},
                 element=field, state_message=_SHADOW_SM)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    assert steps[0]["selectors"][0] == "xpath=/html/body/div/div/input"


_SHADOW_SM_NO_LABEL = """Interactive elements:
[Start of page]
\t|SHADOW(open)|[300]<my-widget />
\t\tOpen Shadow
\t\t\t*[363]<select />
\t\tShadow End
[End of page]"""


def test_a_shadow_element_with_neither_attribute_nor_label_is_unanchorable(tmp_path):
    # Nothing can locate it — no attribute to match, no label beside it to scope to, and
    # xpath cannot cross the boundary. Say so: _author_segment refuses the commit and the
    # segment authors live rather than caching a step that fails every replay.
    nameless = {"node_name": "SELECT", "x_path": "html/body/div/div/select",
                "attributes": {"class": "form-select form-select-sm"}}
    item = _item({"select_dropdown": {"index": 363, "text": "Payment on behalf"}},
                 element=nameless, state_message=_SHADOW_SM_NO_LABEL)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    assert [s["action"] for s in steps] == ["unanchorable"]
    assert "shadow root" in steps[0]["why"]


def test_inputmode_is_a_distinguishing_attribute():
    from automation.pipeline.script_compile import _selectors_from_parts

    assert _selectors_from_parts("input", {"inputmode": "decimal"}, "") == \
        ['css=[inputmode="decimal"]']
    # An element with nothing at all still yields nothing — the widening is narrow.
    assert _selectors_from_parts("button", {"type": "button"}, "") == []


# ---------------- the label rung is gated on MISSING ATTRIBUTES ----------------
# It used to be gated on `in_shadow`, which was both wrong (it fired on every native
# input) and beside the point: what makes a label the only way to find a control is that
# no attribute names it. tests/test_shadow_dom_find.py measures the selector itself
# against the live markup; these pin WHEN compile offers it.


def test_an_attributeless_control_gets_the_label_rung_after_its_xpath(tmp_path):
    listing = """Interactive elements:
[Start of page]
\tPeriod to
\t*[737]<select />
[End of page]"""
    nameless = {"node_name": "SELECT", "x_path": "html/body/div/div/select",
                "attributes": {"class": "form-select form-select-sm"}}
    item = _item({"select_dropdown": {"index": 737, "text": "Jun-26"}},
                 element=nameless, state_message=listing)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    assert [s["action"] for s in steps] == ["select"]
    assert steps[0]["selectors"] == [
        "xpath=/html/body/div/div/select",
        sc._label_scoped_css("select", "Period to")]


def test_the_label_rung_refuses_a_label_that_is_the_pages_data(tmp_path):
    """Run 20260825_163029: widening the rung from shadow-contained to attribute-less
    first offered it to Pay Forecast table cells, whose neighbouring line is the row's
    AMOUNT — `*:text-is("\u00a32446.44")` goes stale the moment the pay changes, and can
    match a different row showing the same figure."""
    listing = """Interactive elements:
[Start of page]
\t\u00a32446.44
\t*[812]<td />
[End of page]"""
    cell = {"node_name": "TD", "x_path": "html/body/table/tbody/tr[11]/td[15]",
            "attributes": {}}
    item = _item({"click": {"index": 812}}, element=cell, state_message=listing)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    assert steps[0]["selectors"] == ["xpath=/html/body/table/tbody/tr[11]/td[15]"]


def test_the_label_rung_refuses_a_neighbours_label(tmp_path):
    """An element that already HAS a name is not named by a different line above it: the
    Calculate button sat under "Calculate for remaining periods", which names another
    control. An element whose own name is itself data (the popup's "\u00a3" box) reads as
    nameless and keeps its rung — that is the case the rung exists for."""
    from automation.pipeline.script_compile import _label_names

    assert _label_names("Student loan", "Student loan") is True     # agrees
    assert _label_names("Calculate for remaining periods", "Calculate") is False
    assert _label_names("Net amount", "\u00a3") is True             # a nameless box
    assert _label_names("Net amount", "") is True
    assert _label_names("\u00a32446.44", "") is False               # data, not a name
    assert _label_names("", "") is False
    assert _label_names(None, "") is False


def test_a_control_with_an_attribute_gets_no_label_rung(tmp_path):
    listing = """Interactive elements:
[Start of page]
\tPeriod to
\t*[737]<select name=periodTo />
[End of page]"""
    named = {"node_name": "SELECT", "x_path": "html/body/div/div/select",
             "attributes": {"name": "periodTo"}}
    item = _item({"select_dropdown": {"index": 737, "text": "Jun-26"}},
                 element=named, state_message=listing)
    steps = compile_recording(_write(tmp_path, [item]), emit_start_goto=False)

    assert steps[0]["selectors"] == ["xpath=/html/body/div/div/select",
                                     'css=[name="periodTo"]']


def test_the_label_reader_ignores_elements_edges_and_icon_glyphs():
    from automation.pipeline.script_compile import _sm_preceding_label

    # A Fluent icon is a literal Private-Use-Area text node and names nothing.
    glyph = "Interactive elements:\n\t\ue70f\n\t|SHADOW(open)|*[9]<select />"
    assert _sm_preceding_label(glyph, 9) is None
    # An element line above is not a label either.
    elem = "Interactive elements:\n\t[8]<button />\n\t|SHADOW(open)|*[9]<select />"
    assert _sm_preceding_label(elem, 9) is None
    # ...and a paragraph is not a label.
    long = "Interactive elements:\n\t" + "x" * 60 + "\n\t|SHADOW(open)|*[9]<select />"
    assert _sm_preceding_label(long, 9) is None
    assert _sm_preceding_label("Interactive elements:\n\tName\n\t*[9]<select />", 9) == "Name"






def _named_btn(text):
    """Two controls sharing one DOM position, told apart only by their recorded name."""
    return {"node_name": "button", "ax_name": text,
            "attributes": {"type": "button"},
            "x_path": "html/body/div/div/form/div[2]/button[2]"}


def test_a_relabelled_control_at_the_same_position_is_not_a_repeat(tmp_path):
    history = [_item({"click": {"index": 1}}, element=_named_btn("Next")),
               _item({"wait": {"seconds": 1}}),
               _item({"click": {"index": 1}}, element=_named_btn("Next")),
               _item({"click": {"index": 1}}, element=_named_btn("Submit"))]

    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    clicks = [s for s in steps if s["action"] == "click"]
    names = [(s.get("fingerprint") or {}).get("text") for s in clicks]
    assert names == ["Next", "Submit"], f"the Submit must survive as its own step: {names}"
    assert clicks[0]["count"] == 2          # the two real Next clicks still fuse
    assert "count" not in clicks[1]         # the Submit is one click, not an iteration


def test_nameless_repeat_clicks_at_one_position_still_fuse(tmp_path):
    """The split is name-driven and conservative: with no recorded name to disagree on
    (icon-only buttons), the existing repeat behaviour must be untouched."""
    icon = {"node_name": "button", "ax_name": None,
            "attributes": {"class": "ms-Button--icon"},
            "x_path": "html/body/div/div/form/div[2]/button[2]"}
    history = []
    for _ in range(3):
        history.append(_item({"click": {"index": 1}}, element=icon))
        history.append(_item({"wait": {"seconds": 1}}))

    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    clicks = [s for s in steps if s["action"] == "click"]
    assert len(clicks) == 1 and clicks[0]["count"] == 3






_OPTION_EL = {"node_name": "DIV", "ax_name": "no-reply",
              "attributes": {"id": "react-select-18-option-1", "class": "rs-option"}}
_COMBO_EL = {"node_name": "INPUT", "ax_name": "From :",
             "attributes": {"role": "combobox", "id": "react-select-18-input"},
             "x_path": "html/body/div[2]/div/div/div[3]/input"}


def _pick_item(opener=_COMBO_EL, **kw):
    meta = {"interacted_element": _OPTION_EL}
    if opener is not None:
        meta["opener_element"] = opener
    return _item({"select_dropdown": {"text": "no-reply", "near_text": "From", "index": 0}},
                 result=[{"extracted_content": "Selected 'no-reply'", "metadata": meta}],
                 **kw)


def test_tool_opened_combobox_pick_compiles_its_opener(tmp_path):
    steps = compile_recording(_write(tmp_path, [_pick_item()]), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "type", "click"]
    # The opener is anchored on the CONTROL, never on the option row.
    assert steps[0]["selectors"][0] == "xpath=/html/body/div[2]/div/div/div[3]/input"
    assert steps[0].get("fingerprint", {}).get("tag") == "input"
    # The label the agent addressed it by survives as the last selector rung: a
    # react-select input has no attribute that can name it.
    assert any('text-is("From :")' in s for s in steps[0]["selectors"])
    assert steps[1]["text"] == "no-reply" and steps[2]["expect_text"] == "no-reply"


def test_a_pick_after_a_real_click_on_the_box_gets_no_second_opener(tmp_path):
    """The common shape (entry 4154bfa3a788527f): the agent clicked the combobox itself
    before calling select_dropdown. A second click there would CLOSE the menu."""
    history = [_item({"click": {"index": 7}}, element=_COMBO_EL), _pick_item()]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert [s["action"] for s in steps] == ["click", "type", "click"]


def test_a_pick_with_no_opener_stamp_compiles_as_before(tmp_path):
    """Old recordings carry no stamp — they must keep compiling, not raise."""
    steps = compile_recording(_write(tmp_path, [_pick_item(opener=None)]),
                              emit_start_goto=False)
    assert [s["action"] for s in steps] == ["type", "click"]


# --- the wrong ROW (entry 07044b6a0dbf7988, run 20260827_104331) -----------------------
# The Data Request grid's external-link icon has no name, every row carries one with the
# same title, and its href embeds the RECORD id. Compile's only anchor was the positional
# path `.../div[2]/div[9]/…`: the run created CDR072, clicked whatever link sat there, and
# wrote 13 UpdateCal POSTs into CDR054 — the previous day's request.

_LINK_EL = {
    "node_name": "A", "ax_name": None,
    "attributes": {"target": "_blank", "title": "Open payroll review request as client",
                   "href": "/links/10/c/6a61d0ab5636abb464ba0e13/r/6a8f6b691522ef667cd114cf/"
                           "calcdatarequest"},
    "x_path": "html/body/div[1]/div/div/div[2]/div[9]/div/div/div[2]/div/a",
    "row": {"scope": '[role="row"]',
            "cells": ["1", "PR/01797494/27/CDR072", "FOOD LIMITED", "Drafted"]},
}


def test_a_nameless_in_row_click_is_anchored_by_its_row(tmp_path):
    steps = compile_recording(
        _write(tmp_path, [_item({"click": {"index": 9}}, element=_LINK_EL)]),
        emit_start_goto=False)
    sels = steps[0]["selectors"]
    # The row's own data leads — ahead of the positional path.
    assert sels[0] == ('css=[role="row"]:has-text("PR/01797494/27/CDR072") '
                       'a[title="Open payroll review request as client"]')
    assert sels.index(sels[0]) < sels.index("xpath=/" + _LINK_EL["x_path"])
    # A column value that repeats down the grid still gets a candidate; it resolves
    # ambiguously at replay and _resolve skips it. Order is longest-first.
    assert any('has-text("FOOD LIMITED")' in s for s in sels)
    # The recorded href points at the row this recording acted on, forever. Gone.
    assert not any("6a8f6b69" in s for s in sels)


def test_the_row_identity_survives_browser_uses_own_element_capture(tmp_path):
    """The row is read by our click override, but browser-use's state.interacted_element
    wins over the stamp — the row must ride across, or it is dropped for every indexed
    click that has one."""
    state_el = {k: v for k, v in _LINK_EL.items() if k != "row"}
    state_el["backend_node_id"] = 40465
    stamped = dict(_LINK_EL, backend_node_id=40465)
    history = [_item({"click": {"index": 9}}, element=state_el,
                     result=[{"extracted_content": "Clicked a",
                              "metadata": {"interacted_element": stamped}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps[0]["selectors"][0].startswith('css=[role="row"]:has-text(')

    # Two captures that describe DIFFERENT nodes never lend each other a row.
    other = dict(stamped, backend_node_id=999)
    history[0]["result"][0]["metadata"]["interacted_element"] = other
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps[0]["selectors"][0].startswith("xpath=")


def test_a_named_row_click_keeps_todays_ladder(tmp_path):
    """A named control is already guarded by expect_text at replay — the record side
    stamps no row for it, and its selectors must not move."""
    named = {"node_name": "BUTTON", "ax_name": "PR/01797494/27/CDR072",
             "attributes": {"type": "button"},
             "x_path": "html/body/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div/button"}
    steps = compile_recording(
        _write(tmp_path, [_item({"click": {"index": 9}}, element=named)]),
        emit_start_goto=False)
    assert steps[0]["selectors"][0].startswith("xpath=")


def test_our_own_synthetic_combobox_id_is_volatile():
    """library/5a90660d1df6a541 (Additions/Deductions), the two <select> steps: both were
    anchored on `ao-cb-7` / `ao-cb-8` — the id _CB_RESOLVE_JS STAMPS ONTO a control that
    had none, so the resolver has something to hold. It is our own scratch attribute and
    it does not exist on any later run.

    _NATIVE_SELECT_BY_ID_JS already strips the prefix from the attrs IT reports, but these
    elements reached compile through browser-use's own state.interacted_element capture,
    taken AFTER the stamp — so the guard was bypassed and the id arrived looking
    app-authored. That cost the step twice over: a `css=[id="ao-cb-7"]` candidate that can
    never match, and (worse) a fingerprint carrying attrs={"id": "ao-cb-7"}, which made
    _xpath_matches_fingerprint REFUSE the correct positional-xpath hit on every replay and
    drop the step into healing — where two anonymous sibling <select>s score alike and the
    heal cannot break the tie.

    The 3+ digit rule already caught `ao-cb-123` by accident, so the failure was
    counter-value dependent: the same control replayed fine once the page had stamped a
    hundred controls, and not at all before that."""
    from automation.pipeline.script_compile import _is_dynamic_id, _selectors_from_parts

    for volatile in ("ao-cb-7", "ao-cb-8", "ao-cb-1", "ao-cb-123"):
        assert _is_dynamic_id(volatile), volatile
    # `cb2` stays stable — the guard is the exact `ao-cb-` prefix we stamp, not a
    # letters-and-digits rule (see the Fluent-counter test above).
    assert not _is_dynamic_id("cb2")

    # The attrs exactly as recorded for the "Period to" select (history item 4).
    attrs = {"class": "mt-l form-select form-select-sm",
             "style": "width: 300px; font-size: 12px;", "id": "ao-cb-7"}
    assert not [s for s in _selectors_from_parts("select", attrs, "") if "ao-cb" in s]


def test_an_errored_select_dropdown_compiles_to_nothing(tmp_path):
    """entry 4154bfa3a788527f, run 20260827_131953: the agent aimed select_dropdown
    ('May-26') at the PAY FREQUENCY combobox, which answered "no such option. The dropdown
    ACTUALLY lists: 'Monthly', 'Weekly', ..." — it set nothing. It then clicked the correct
    date box and picked May-26 there, and that pair worked.

    Compile stored BOTH, wrong one first, because the select_dropdown branch never asked
    whether the action succeeded: the recorded interacted_element is the PRE-action DOM
    snapshot, so it names the wrong box even though the tool did nothing to it. The skill
    typed 'May-26' into the pay-frequency dropdown on every replay. react-select discards
    unmatched filter text, so it passed some runs and killed the segment on others
    ("no unique candidate matched ... role=option[name='May-26'] -> no match").

    Unlike the no_fill/no_click twins above, this tool's refusal stamps NO metadata — but
    it is atomic (it opens, picks, and reads the value back), so a result carrying an error
    is proof the value was never set."""
    wrong = {"node_name": "INPUT", "attributes": {"id": "react-select-14-input"},
             "x_path": "html/body/form/div[2]/input"}
    right = {"node_name": "INPUT", "attributes": {"id": "react-select-15-input"},
             "x_path": "html/body/form/div[3]/input"}
    rec = _write(tmp_path, [
        _item({"select_dropdown": {"text": "May-26", "index": 1}}, element=wrong,
              result=[{"error": "select_dropdown 'May-26' at index 1: no such option. "
                                "The dropdown ACTUALLY lists: 'Monthly', 'Weekly'."}]),
        _item({"click": {"index": 2}}, element=right,
              result=[{"extracted_content": "Clicked input id=react-select-15-input"}]),
    ])

    steps = compile_recording(rec)
    sels = [s for step in steps for s in (step.get("selectors") or [])]
    assert not [s for s in sels if "div[2]" in s], f"wrong combobox compiled: {sels}"
    assert [s for s in sels if "div[3]" in s], f"right combobox lost: {sels}"


def test_dropdown_opener_comes_from_the_tools_stamp_not_the_snapshot(tmp_path):
    """entry aa3a76b7c82dcf8b, run 20260827_131953: the compiled opener for the Send Email
    From dropdown was byte-for-byte the panel's CLOSE (X) button, so the skill closed the
    panel and then hunted the From menu inside it — 21 minutes and 750k tokens of recovery
    before the segment failed.

    select_dropdown opens the widget itself, so this branch has to synthesize the opener
    click. It read `element` — browser-use's state.interacted_element — and assumed "not an
    option row, so it must be the combobox". For a select_dropdown that snapshot can name
    something the tool never touched; here it named the close button. agent_tools stamps the
    combobox it ACTUALLY opened in metadata.opener_element, which the sibling branch already
    trusts. Prefer the stamp; fall back to the snapshot so stamp-less older recordings keep
    working."""
    close_btn = {"node_name": "BUTTON",
                 "attributes": {"class": "ms-Panel-closeButton"},
                 "x_path": "html/body/panel/div[1]/div/button"}
    combobox = {"node_name": "INPUT",
                "attributes": {"id": "react-select-16-input", "role": "combobox"},
                "x_path": "html/body/panel/form/div[1]/input", "ax_name": "Me From"}
    rec = _write(tmp_path, [
        _item({"select_dropdown": {"text": "no-reply", "index": 1}}, element=close_btn,
              result=[{"extracted_content": "Selected 'no-reply'",
                       "metadata": {"opener_element": combobox}}]),
    ])

    sels = [s for step in compile_recording(rec) for s in (step.get("selectors") or [])]
    assert not [s for s in sels if "div[1]/div/button" in s], f"close button opener: {sels}"
    assert [s for s in sels if "form/div[1]/input" in s], f"real combobox lost: {sels}"


def test_a_control_with_only_its_own_name_gets_an_anchor():
    """entry 8dd0163e663bfbf0 (Add Payments), runs 20260827_2241/2246/2226: the dialog's
    Save is `<button type="button" class="btn btn-primary btn-sm">Save</button>` — no id,
    no name, no aria-label — so the attribute ladder came back EMPTY, and the label rung
    refused because the listing line above it is 'Cancel', which names a DIFFERENT control.
    That left ONE positional xpath and nothing behind it, so the step fell into the
    fingerprint heal on every single run: `score=6.5 margin=3.2`, three runs running,
    identical — the heal was scoring the very name the ladder had declined to use
    (text +3, role +2, tag +1, type +0.5). It was not replaying, it was guessing right."""
    from automation.pipeline.script_compile import _selectors

    save = {"node_name": "BUTTON", "ax_name": "Save",
            "attributes": {"type": "button", "class": "btn btn-primary btn-sm"},
            "x_path": "html/body/form/div[5]/div/div/div[3]/button[2]"}
    sels = _selectors(save, label="Cancel")
    assert 'css=button:text-is("Save")' in sels
    # The neighbour label must STILL be refused — 'Cancel' names another button.
    assert not [s for s in sels if "Cancel" in s]

    # A control an attribute CAN name never reaches the rung: the 2026-08-13
    # "xpaths for everything, no random searches" rule is untouched for those.
    named = dict(save, attributes={"type": "button", "id": "mailbtn"})
    assert not [s for s in _selectors(named, label="Cancel") if ":text-is(" in s]

    # Restricted to <button>/<a> — the tags whose TEXT IS THEIR NAME. Anything else and
    # the text is content or data, which is the search this ladder refuses to do. Both of
    # these were live regressions when the rung was first written tag-agnostic:
    #   - a bare <div> with text must stay unanchorable by design
    #   - the 'Sent' status chip is a <span> whose text is the ROW'S DATA, so
    #     css=span:text-is("Sent") would match whichever row says Sent today
    for tag, name in (("div", "Some text"), ("span", "Sent"), ("input", "Amount")):
        other = {"node_name": tag, "ax_name": name, "attributes": {},
                 "x_path": "html/body/form/x"}
        assert not [s for s in _selectors(other, label="Cancel") if ":text-is(" in s], tag


def test_an_ambiguous_self_name_falls_through_instead_of_clicking_the_first():
    """Two buttons saying 'Save' is the one ambiguity this rung cannot settle, so it is
    denied _resolve's last-candidate first-visible concession — it falls through to the
    fingerprint heal, which is what the step did before the rung existed. The ` >> ` test
    keeps the denial off _label_scoped_css, which also uses :text-is and keeps its
    concession."""
    from automation.pipeline.script_compile import _is_self_named, _label_scoped_css

    assert _is_self_named('css=button:text-is("Save")')
    assert not _is_self_named(_label_scoped_css("select", "Period to"))
    assert not _is_self_named('css=[id="mailbtn"]')


def test_a_control_only_its_class_can_name_gets_an_anchor():
    """entry 07044b6a0dbf7988 (the OTP segment), run 20260828_004155: Fluent's panel close
    button. Its whole DOM-listing line is `*[8657]<button />` — ax_name None, no id, no
    aria-label, no title, no name — so the attribute ladder is empty, the self-name rung
    has no name, and the label rung has no label. Its ONLY anchor was a positional xpath
    through `body/div[2]`, which is Fluent's LAYER HOST, and that index moves run to run
    (div[3] vs div[2]). When it moved the step failed outright and the segment — ~240k
    tokens to author — dropped to the LLM.

    The churn that keeps `class` out of the attribute ladder lives in the MOUNT-COUNTER
    suffix (`closeButton-1220`), not in the component name in front of it."""
    from automation.pipeline.script_compile import _selectors

    close = {"node_name": "BUTTON", "ax_name": None,
             "attributes": {"type": "button", "data-is-visible": "true",
                            "class": "ms-Button ms-Panel-closeButton ms-PanelAction-close "
                                     "ms-Button--icon closeButton-1220"},
             "x_path": "html/body/div[2]/div/div[1]/div/button[2]"}
    sels = _selectors(close)
    # Longest token first, so the component name outranks the generic one.
    assert sels[1] == "css=button.ms-Panel-closeButton"
    # The mount counter is never an anchor, and short/generic tokens are dropped.
    assert not [s for s in sels if "1220" in s or s.endswith(".ms-Button")]
    assert len([s for s in sels if s.startswith("css=button.")]) <= 3

    # An element an attribute CAN name never reaches the rung.
    named = dict(close, attributes={"id": "mailbtn", "class": "ms-Panel-closeButton"})
    assert not [s for s in _selectors(named) if s.startswith("css=button.")]

    # Nor one the label rung already named: the class is the LAST resort, not a peer.
    labelled = {"node_name": "select", "ax_name": "", "attributes": {"class": "form-select-sm"},
                "x_path": "html/body/form/select"}
    assert not [s for s in _selectors(labelled, label="Period to")
                if s.startswith("css=select.")]


def test_a_purely_utility_class_is_not_an_anchor():
    """A STYLING class must never anchor a step, only a framework COMPONENT class.

    The separator is the camelCase hump: Fluent and CSS-modules write
    `ms-Panel-closeButton`, Bootstrap writes `form-select form-select-sm` / `btn
    btn-primary` — classes shared by every control of that kind. Gating on length alone
    let `form-select-sm` through and broke the rule that a shadow <select> with no
    attribute and no label is unanchorable by design; a step must never LOOK anchorable
    while its only candidate matches every select on the page."""
    from automation.pipeline.script_compile import (_selectors, _is_class_scoped,
                                                    _semantic_class_selectors)

    chip = {"node_name": "SPAN", "ax_name": None,
            "attributes": {"class": "label label-148", "style": "display: block;"},
            "x_path": "html/body/div[1]/span/span"}
    assert _selectors(chip) == ["xpath=/html/body/div[1]/span/span"]
    for tag, cls in (("select", "form-select form-select-sm"),
                     ("button", "btn btn-primary btn-sm"),
                     ("span", "label label-148")):
        assert _semantic_class_selectors(tag, {"class": cls}) == [], cls
    # Denied _resolve's first-visible concession: a class names a KIND, not one control.
    assert _is_class_scoped("css=button.ms-Panel-closeButton")
    assert not _is_class_scoped('css=[id="mailbtn"]')


# ------------- the counter tool replaces the loop flag (2026-08-28) -------------
# A repeat used to be INFERRED from adjacency, which needed `kind: loop` to tell an
# iteration from a slow-app retry — inferred in turn from the prompt's wording. The agent
# now states the repeat through the repeat_click tool, so the count is a fact on the step.


def _repeat_item(count, until_done=False, text="Save & Next"):
    return _item({"repeat_click": {"index": 1, "times": 0 if until_done else count}},
                 element=_btn(text),
                 result=[{"metadata": {"repeat": {"count": count,
                                                  "until_done": until_done,
                                                  "wait_s": 0.4}}}])


def test_a_stated_repeat_compiles_to_one_counted_click(tmp_path):
    steps = sc.compile_recording(str(_write(tmp_path, [_repeat_item(11)])),
                                 emit_start_goto=False)
    clicks = [x for x in steps if x["action"] == "click"]
    assert len(clicks) == 1
    assert clicks[0]["count"] == 11
    assert clicks[0]["stated_count"] is True


def test_a_stated_repeat_survives_the_unnumbered_dissolve(tmp_path):
    """_apply_repeat_hint dissolves an adjacency-INFERRED cluster when no wording pins it
    (the Download-toggle bug). A stated count is not a guess and must be left alone — this
    is what removes the "exactly N clicks" wording workaround."""
    out = tmp_path / "s.json"
    steps = sc.save_steps(str(_write(tmp_path, [_repeat_item(11)])), out,
                          emit_start_goto=False, repeat_hint=None)
    assert [x for x in steps if x["action"] == "click"][0]["count"] == 11


def test_until_done_records_the_intent_not_the_number(tmp_path):
    """times=0 means "until it stops advancing". Freezing the authoring run's count would
    under-run a longer list, so the step carries the intent and the replay re-discovers
    the end."""
    steps = sc.compile_recording(str(_write(tmp_path, [_repeat_item(11, until_done=True)])),
                                 emit_start_goto=False)
    click = [x for x in steps if x["action"] == "click"][0]
    assert click["until_done"] is True and click["count"] == 11


def test_a_refused_repeat_never_becomes_a_step(tmp_path):
    """No `repeat` metadata (the tool stamps no_click on a shortfall or refusal) means no
    step — a wrong count must never be cached."""
    item = _item({"repeat_click": {"index": 1, "times": 6}}, element=_btn(),
                 result=[{"metadata": {"no_click": True}}])
    steps = sc.compile_recording(str(_write(tmp_path, [item])), emit_start_goto=False)
    assert [x for x in steps if x["action"] == "click"] == []


def test_adjacent_clicks_are_still_read_as_retries(tmp_path):
    """The retry rule is unchanged and no longer has a loop-flag escape hatch: two
    back-to-back clicks on one target are one click, because the agent would have used
    repeat_click had it meant two."""
    history = [_item({"click": {"index": 1}}, element=_btn()) for _ in range(2)]
    steps = sc.compile_recording(str(_write(tmp_path, history)), emit_start_goto=False)
    clicks = [x for x in steps if x["action"] == "click"]
    assert len(clicks) == 1 and "count" not in clicks[0]



def test_instantiate_skips_expect_text_when_the_token_only_scopes_a_row():
    """A bound value inside a `:has-text(...)` ROW SCOPE names the ROW, not the target.

    Run 20260828_153xxx, subtask 6 (a0d5426d4f0855c0): the row-scoped candidate
    `[role="row"]:has-text("Preston Alexander") div[data-automationid="DetailsRowCheck"]`
    resolved to exactly ONE visible element — the right checkbox — and the single-token
    stamp then rejected it ("none named 'Preston Alexander'"), because a Fluent
    DetailsRowCheck div has no text, no aria-label and no associated label. The scope IS
    the wrong-row guard here; stamping a name gate on top of it vetoes its own answer.
    """
    from automation.pipeline.adapt import instantiate

    template = {
        "params": {"employee": "WI Person"},
        "steps": [
            # row scope + descendant: the value names the ROW, the click hits the checkbox
            {"action": "click", "selectors": [
                'css=[role="row"]:has-text("{{employee}}") div[data-automationid="DetailsRowCheck"]',
                'css=[data-automationid="DetailsRowCheck"]']},
            # no descendant part: the click IS the row, so the value does name the target
            {"action": "click", "selectors": ['css=[role="row"]:has-text("{{employee}}")']},
            # scoped in one candidate, but another names the target outright
            {"action": "click", "selectors": [
                'css=[role="row"]:has-text("{{employee}}") button',
                'role=link[name="{{employee}}"]']},
        ],
    }
    steps = instantiate(template, {"employee": "Preston Alexander"})
    assert steps[0].get("expect_text") is None            # scope-only -> no name gate
    assert steps[1]["expect_text"] == "Preston Alexander"  # the row itself is the target
    assert steps[2]["expect_text"] == "Preston Alexander"  # a candidate names the target


def test_substituted_anchors_skips_expect_text_when_the_token_only_scopes_a_row():
    """The tier-1 twin of the stamp carries the same rule (skills.base)."""
    from automation.skills.base import _substituted_anchors

    anchors = {
        "employee-check": {"selectors": [
            'css=[role="row"]:has-text("{{employee}}") div[data-automationid="DetailsRowCheck"]',
            'css=[data-automationid="DetailsRowCheck"]']},
        "employee-row": {"selectors": ['css=[role="row"]:has-text("{{employee}}")']},
    }
    out = _substituted_anchors(anchors, {"employee": "Preston Alexander"})
    assert out is not None
    assert out["employee-check"].get("expect_text") is None
    assert out["employee-row"]["expect_text"] == "Preston Alexander"
