"""Self-noted values: a value a recording EXTRACTS and then TYPES inside the SAME segment.

The runtime binder (test_bindings.py) only binds to values EARLIER segments produced —
a segment's own extracts happen after its load, so a load-time binding could never
resolve. That left the merged OTP slice ("click Get OTP, copy the number ... paste it
into the code box") with no binding at all: codegen baked the authoring run's literal and
every replay pasted a dead code (run 20260826_095932 captured 776106 and pasted 587923).

The fix is a token that resolves at STEP-EXECUTION time instead of load time: the value
becomes {{noted:<label>}}, and whichever tier runs it reads the label out of the live
extract ledger the copy step just filled.
"""
import json

import pytest

from automation.pipeline.hybrid import _bind_self_noted
from tests.test_bindings import stores  # noqa: F401 - fixture
from automation.skills import codegen
from automation.skills.api import SkillApi


def _copy(label):
    return {"action": "copy", "label": label, "selectors": ["css=.otp"],
            "query": "the 6 digit code"}


def _paste(value):
    return {"action": "paste", "value": value, "selectors": ["css=#code-1"]}


# ------------------------------- commit-time rewrite -------------------------------


def test_a_value_this_recording_extracted_earlier_becomes_a_noted_token():
    steps, labels = _bind_self_noted(
        [_copy("otp"), _paste("587923")], {"otp": "587923"})

    assert steps[1]["value"] == "{{noted:otp}}"
    assert labels == ["otp"]


def test_the_extract_step_itself_is_left_alone():
    steps, _ = _bind_self_noted([_copy("otp"), _paste("587923")], {"otp": "587923"})

    assert steps[0] == _copy("otp")


def test_a_value_extracted_only_AFTER_it_was_typed_is_not_rewritten():
    # Nothing has filled the ledger when the paste runs, so the token could not resolve.
    steps, labels = _bind_self_noted(
        [_paste("587923"), _copy("otp")], {"otp": "587923"})

    assert steps[0]["value"] == "587923"
    assert labels == []


def test_a_value_no_extract_produced_is_left_alone():
    steps, labels = _bind_self_noted(
        [_copy("otp"), _paste("4002")], {"otp": "587923"})

    assert steps[1]["value"] == "4002"
    assert labels == []


def test_the_rewrite_is_boundary_safe():
    # The house-number lesson (run 20260817_115232): a short extract must never rewrite
    # the middle of a longer unrelated value.
    steps, labels = _bind_self_noted(
        [_copy("house"), {"action": "fill", "value": "3500", "selectors": ["css=#pay"]}],
        {"house": "35"})

    assert steps[1]["value"] == "3500"
    assert labels == []


def test_selectors_are_never_rewritten():
    # An element's IDENTITY must not become a runtime value: only what gets typed does.
    steps, _ = _bind_self_noted(
        [_copy("otp"), {"action": "paste", "value": "587923",
                        "selectors": ["css=[data-x='587923']"]}],
        {"otp": "587923"})

    assert steps[1]["selectors"] == ["css=[data-x='587923']"]


def test_a_typed_step_matching_case_insensitively_still_binds():
    steps, labels = _bind_self_noted(
        [_copy("town"), {"action": "fill", "value": "butt green",
                         "selectors": ["css=#town"]}],
        {"town": "BUTT GREEN"})

    assert steps[1]["value"] == "{{noted:town}}"
    assert labels == ["town"]


# ------------------------------- tier 1: generated code -------------------------------


def test_transpile_emits_api_noted_for_a_noted_token():
    code, _ = codegen.transpile("sid", [_copy("otp"), _paste("{{noted:otp}}")])

    assert "await api.paste(" in code
    assert "api.noted('otp')" in code
    assert "587923" not in code


def test_the_emitted_noted_call_passes_the_skill_lint():
    code, _ = codegen.transpile("sid", [_copy("otp"), _paste("{{noted:otp}}")])

    assert codegen.lint_code(code) == []


# ------------------------------- tier 1: the api verb -------------------------------


def test_api_noted_reads_the_live_extract_ledger():
    api = SkillApi(page=None, anchors={})
    api.extracted["otp"] = "776106"

    assert api.noted("otp") == "776106"


def test_api_noted_raises_when_the_producing_step_captured_nothing():
    api = SkillApi(page=None, anchors={})

    with pytest.raises(KeyError):
        api.noted("otp")


# ------------------------------- tier 0: compiled steps -------------------------------


def test_resolve_noted_substitutes_from_the_live_ledger():
    from automation.pipeline.script_compile import resolve_noted

    assert resolve_noted("{{noted:otp}}", {"otp": "776106"}) == "776106"


def test_resolve_noted_raises_rather_than_typing_the_token():
    from automation.pipeline.script_compile import resolve_noted

    with pytest.raises(RuntimeError):
        resolve_noted("{{noted:otp}}", {})


# ------------------------------- wiring: tier 0 end to end -------------------------------


async def test_run_steps_pastes_the_value_this_run_extracted():
    """The whole bug in one test: extract the code the page is showing NOW, then paste it.
    A run that bakes the authoring literal types a dead OTP and still reports success."""
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(
            "<span id='code'>776106</span><input id='box'>")

        out = await run_steps(page, [
            {"action": "extract", "label": "otp", "selectors": ["css=#code"]},
            {"action": "paste", "value": "{{noted:otp}}", "selectors": ["css=#box"]},
        ], timeout_ms=3000)

        assert out["failed_at"] is None, out["error"]
        assert await page.locator("#box").input_value() == "776106"


async def test_run_steps_fails_honestly_when_the_noted_value_is_missing():
    from playwright.async_api import async_playwright

    from automation.pipeline.script_compile import run_steps
    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content("<input id='box'>")

        out = await run_steps(page, [
            {"action": "paste", "value": "{{noted:otp}}", "selectors": ["css=#box"]},
        ], timeout_ms=3000)

        assert out["failed_at"] == 0
        assert "otp" in out["error"]
        assert await page.locator("#box").input_value() == ""


# ------------------------------- wiring: the commit path -------------------------------


def test_commit_rewrites_a_self_noted_paste_before_the_provenance_guards_see_it():
    """_bind_self_noted must run BEFORE the runtime-value legs: once the literal is a
    token it is no longer an unattributable typed value, so the consumer gate stops
    refusing the commit over the very value that is now bound."""
    from automation.pipeline.hybrid import _unattributed_typed_values

    steps, _ = _bind_self_noted([_copy("otp"), _paste("587923")], {"otp": "587923"})

    assert _unattributed_typed_values(steps, "paste the OTP from the previous step", []) == []


# ------------------------------- the live failure, end to end -------------------------------


OTP_RECORDING = {"history": [
    {
        # The OTP span carries NO accessible name in the live recording (that is why its
        # handle compiles to 'copy'): its text IS the volatile value, so nothing about
        # this element's identity may encode the code.
        "state": {"url": "http://app/dr", "interacted_element": [
            {"node_name": "span", "attributes": {},
             "x_path": "html/body/div[1]/span/span"}]},
        "model_output": {"action": [{"copy_text": {"index": 5, "label": "otp"}}]},
        "result": [{"metadata": {"extract": {
            "label": "otp", "value": "587923", "query": "the 6 digit code",
            "interacted_element": {"node_name": "span", "attributes": {},
                                   "x_path": "html/body/div[1]/span/span"}}}}],
    },
    {
        "state": {"url": "http://app/dr", "interacted_element": [
            {"node_name": "input", "ax_name": "Please enter OTP character 1",
             "attributes": {"id": "code-1"}, "x_path": "html/body/div[2]/input"}]},
        "model_output": {"action": [{"paste_text": {"index": 7, "text": "587923"}}]},
        "result": [{"metadata": {"paste": {"value": "587923"}}}],
    },
]}


async def test_the_otp_slice_caches_and_the_next_run_pastes_ITS_OWN_code(
        stores, monkeypatch, capsys):
    """The whole live failure (run 20260826_095932) as one test: one subtask copies the
    OTP and pastes it. Run 1 authors with 587923; run 2 must replay — at zero tokens —
    and the authoring run's dead code must appear nowhere in what it executes."""
    from automation.pipeline import hybrid
    from automation.pipeline import subtask_store as ss
    from automation.pipeline.hybrid import run_hybrid_task
    from automation.tasks import SubtaskDecl, TaskSpec
    from tests.test_bindings import _CapturingSession
    from tests.test_hybrid import FakeSession, _runner, _seg

    otp_prompt = ("click Get OTP, copy the 6 digit number (OTP), then paste the OTP "
                  "from the previous step into the first code box")
    prompt = "go to the section. " + otp_prompt
    spec = TaskSpec(key="otp", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=otp_prompt),
    ))

    # ---- run 1: authoring.
    author = _seg(True, mode="authored")
    author.extracted = {"otp": "587923"}
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"), author])
    fake.recording = OTP_RECORDING
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    assert (await run_hybrid_task(fake.runner or _runner(), prompt,
                                  spec=spec)).is_successful is True

    ctx = ss.normalize_context("http://app/section")
    sid = ss.subtask_id(otp_prompt, ctx)
    assert sid in ss.load_manifest(), "the OTP slice must cache, not re-author every run"
    body = (ss.code_path(sid).read_text() if ss.code_path(sid).exists()
            else ss.steps_path(sid).read_text())
    assert "587923" not in body, "the authoring run's dead OTP was baked into the entry"
    assert "otp" in body
    assert "bound to the live extract" in capsys.readouterr().out

    # ---- run 2: replay. Nothing it executes may carry the stale code.
    fake2 = _CapturingSession(_runner(), replays=[_seg(True), _seg(True)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    assert (await run_hybrid_task(fake2.runner or _runner(), prompt,
                                  spec=spec)).is_successful is True

    assert fake2.agent_calls == 0 and fake2.replay_calls == 2
    skill = fake2.skills_seen[sid]
    executed = skill.code if skill.body == "code" else json.dumps(skill.steps)
    assert "587923" not in executed
    assert "noted" in executed


async def test_a_self_noted_slice_caches_even_when_the_run_has_findings(
        stores, monkeypatch, capsys):
    """The consumer gate refuses to cache a noted-data slice with nothing bindable in it.
    A self-noted value IS bindable — just at step time rather than load time — so the
    gate must count it, or the merged OTP slice re-authors on every run that happens to
    have an earlier finding (~240k tokens, 405s)."""
    from automation.pipeline import hybrid
    from automation.pipeline import subtask_store as ss
    from automation.pipeline.decompose import consumes_noted_data
    from automation.pipeline.hybrid import run_hybrid_task
    from automation.tasks import SubtaskDecl, TaskSpec
    from tests.test_hybrid import FakeSession, _runner, _seg

    otp_prompt = ("click Get OTP, copy the 6 digit number (OTP), then paste the OTP "
                  "from the previous step into the first code box")
    assert consumes_noted_data(otp_prompt)      # the consumer net does match
    prompt = "go to the section. " + otp_prompt
    spec = TaskSpec(key="otpf", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=otp_prompt),
    ))

    author = _seg(True, mode="authored")
    author.extracted = {"otp": "587923"}
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="the panel is open"), author])
    fake.recording = OTP_RECORDING
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    assert (await run_hybrid_task(fake.runner or _runner(), prompt,
                                  spec=spec)).is_successful is True

    sid = ss.subtask_id(otp_prompt, ss.normalize_context("http://app/section"))
    assert sid in ss.load_manifest()
    assert "no bindable runtime value" not in capsys.readouterr().out
    assert ss.has_script(sid)


async def test_the_stale_recording_net_stands_down_for_a_self_noted_entry(
        stores, monkeypatch):
    """The consumer-wording net retires a cached entry that would type stale values. A
    self-noted entry types NOTHING stale — it reads the value the run just captured — so
    it must survive, exactly as a {{bound_N}} entry does."""
    from automation.pipeline import hybrid
    from automation.pipeline import subtask_store as ss
    from automation.pipeline.hybrid import run_hybrid_task
    from automation.tasks import SubtaskDecl, TaskSpec
    from tests.test_bindings import _CapturingSession
    from tests.test_hybrid import FakeSession, _runner, _seg

    otp_prompt = ("click Get OTP, copy the 6 digit number (OTP), then paste the OTP "
                  "from the previous step into the first code box")
    prompt = "go to the section. " + otp_prompt
    spec = TaskSpec(key="otpn", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=otp_prompt),
    ))

    author = _seg(True, mode="authored")
    author.extracted = {"otp": "587923"}
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"), author])
    fake.recording = OTP_RECORDING
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)
    sid = ss.subtask_id(otp_prompt, ss.normalize_context("http://app/section"))
    assert sid in ss.load_manifest()

    # Run 2 HAS a finding, so the consumer net arms.
    fake2 = _CapturingSession(_runner(), replays=[
        _seg(True, finding="the panel is open"), _seg(True)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    await run_hybrid_task(fake2.runner or _runner(), prompt, spec=spec)

    assert sid in ss.load_manifest(), "the self-noted entry was retired as if stale"
    assert fake2.agent_calls == 0
