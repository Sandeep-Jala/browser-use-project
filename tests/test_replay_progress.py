"""The takeover brief: what a broken replay tells the agent that takes over.

Before this, a failed replay handed the agent one boolean ("something partially
happened") — so a recording that died on its LAST action and one that died on its second
produced the identical prompt, and the agent had to rediscover the boundary by
inspection. These tests pin the two things the brief must get right: the boundary itself,
and its HONESTY (a dispatched action is not a verified effect, the action that broke is
unknown rather than done, and a ledger that cannot be aligned narrates nothing at all).
"""
import pytest

from automation.pipeline.prompts import scoped_subtask_prompt
from automation.skills.base import Skill, replay_progress

# ---------------------------------- tier 0: compiled steps ----------------------------------

STEPS = [
    {"action": "click", "expect_text": "Data Request"},
    {"action": "click", "fingerprint": {"text": "Request"}},
    {"action": "fill", "value": "3500",
     "fingerprint": {"attrs": {"aria-label": "Gross Pay"}}},
    {"action": "click", "expect_text": "Save"},
    {"action": "click", "expect_text": "Close"},
]


def _steps_skill(steps=None):
    return Skill(sid="s0", body="steps", steps=list(steps or STEPS))


def test_tier0_names_what_ran_what_broke_and_what_was_left():
    log = [{"step": 0, "action": "click", "used": "css=#dr"},
           {"step": 1, "action": "click", "used": "css=#req"}]
    p = replay_progress(_steps_skill(), {"executed": 2, "failed_at": 2, "log": log,
                                         "error": "TimeoutError: x", "extracted": {}})

    assert p["done"] == ['clicked "Data Request"', 'clicked "Request"']
    assert p["attempted"] == 'fill "Gross Pay" with "3500"'
    assert p["remaining"] == ['click "Save"', 'click "Close"']
    assert (p["done_count"], p["total"], p["ran_to_end"]) == (2, 5, False)
    assert p["summary"].startswith("ran 2 of 5 recorded action(s)")


def test_tier0_repeat_entries_do_not_shift_the_boundary():
    """A `count: 3` click logs THREE entries at the same step index (run_steps), and
    `executed` still counts the step once. The boundary comes from failed_at, so the
    extra entries must not push the brief a step along."""
    steps = [{"action": "click", "expect_text": "Add row", "count": 3},
             {"action": "click", "expect_text": "Save"}]
    log = [{"step": 0, "action": "click", "used": "x"} for _ in range(3)]
    p = replay_progress(_steps_skill(steps), {"executed": 1, "failed_at": 1, "log": log,
                                              "error": "boom", "extracted": {}})

    assert p["done"] == ['clicked "Add row" 3×']
    assert p["attempted"] == 'click "Save"'
    assert p["remaining"] == []


def test_a_replayed_extract_carries_the_value_it_actually_read():
    steps = [{"action": "extract", "label": "otp", "expect_text": "OTP"},
             {"action": "click", "expect_text": "Proceed"}]
    p = replay_progress(_steps_skill(steps),
                        {"executed": 1, "failed_at": 1,
                         "log": [{"step": 0, "action": "extract", "used": "x"}],
                         "error": "boom", "extracted": {"otp": "482913"}})

    assert p["done"] == ['captured otp from "OTP" = "482913"']


def test_a_fluent_glyph_name_is_not_reported_as_a_control():
    """Fluent draws its icons as literal Private-Use-Area TEXT NODES, so a raw recorded
    name can be an invisible codepoint. Printing it would name a control called nothing."""
    steps = [{"action": "click", "expect_text": ""},
             {"action": "click", "expect_text": "Save"}]
    p = replay_progress(_steps_skill(steps),
                        {"executed": 1, "failed_at": 1,
                         "log": [{"step": 0, "action": "click", "used": "x"}],
                         "error": "boom", "extracted": {}})

    assert p["done"] == ["clicked an unnamed control"]


def test_a_generated_id_is_not_used_as_a_name():
    """Observed live (run 20260908_091814 seg 19): the control that broke was a
    react-select input whose only identity was id="react-select-11-input". That id is
    regenerated on every render, so printing it sends the agent hunting for a string that
    is not on this run's page — its ROLE is the useful thing to say."""
    steps = [{"action": "click", "expect_text": "FPS"},
             {"action": "click", "fingerprint": {"tag": "input", "role": "combobox",
                                                 "attrs": {"id": "react-select-11-input"}}}]
    p = replay_progress(_steps_skill(steps),
                        {"executed": 1, "failed_at": 1,
                         "log": [{"step": 0, "action": "click", "used": "x"}],
                         "error": "no unique candidate", "extracted": {}})

    assert p["attempted"] == "click an unnamed combobox"
    assert "react-select-11-input" not in str(p)


def test_a_stable_id_is_still_a_usable_name():
    steps = [{"action": "click", "fingerprint": {"tag": "a", "attrs": {"id": "btnFPS"}}},
             {"action": "click", "expect_text": "Save"}]
    p = replay_progress(_steps_skill(steps),
                        {"executed": 1, "failed_at": 1,
                         "log": [{"step": 0, "action": "click", "used": "x"}],
                         "error": "boom", "extracted": {}})

    assert p["done"] == ['clicked "btnFPS"']


def test_a_clean_start_failure_is_recorded_but_never_shown_as_progress():
    """Dying on the FIRST action leaves the page untouched, so the run is not dirty and
    the prompt shows nothing (see the clean-start prompt test) — but the brief is still
    built, because "it broke on its first action" is the forensics progress.json keeps."""
    p = replay_progress(_steps_skill(), {"executed": 0, "failed_at": 0, "log": [],
                                         "error": "boom", "extracted": {}})

    assert p["done"] == [] and p["done_count"] == 0
    assert p["attempted"] == 'click "Data Request"'


def test_a_failed_at_outside_the_body_narrates_nothing():
    assert replay_progress(_steps_skill(), {"executed": 9, "failed_at": 99, "log": [],
                                            "error": "boom", "extracted": {}}) is None


# ---------------------------------- tier 1: generated code ----------------------------------

CODE = '''"""Generated tier-1 skill test."""


async def run(api, *, bound_1='PR/01797494'):
    await api.click('data-request')
    await api.click('bound-1-target')
    await api.copy('otp-field', 'otp')
    await api.repeat_click('next-page', 3, 0.5)
    await api.paste('code-box-1', api.noted('otp'))
    await api.click('proceed-securely')
'''

ANCHORS = {
    "data-request": {"expect_text": "Data Request"},
    "bound-1-target": {"expect_text": "PR/01797494"},
    "otp-field": {"expect_text": "One time passcode"},
    "next-page": {"expect_text": "Next page"},
    "code-box-1": {"expect_text": "Please enter OTP character 1"},
    "proceed-securely": {"expect_text": "Proceed Securely"},
}


def _code_skill():
    return Skill(sid="s1", body="code", code=CODE, anchors=dict(ANCHORS),
                 params={"bound_1": "PR/01797494"})


def _entry(action, handle, **extra):
    return {"step": 0, "action": action, "handle": handle, "used": "x", **extra}


def test_tier1_matches_the_ledger_by_handle_not_by_index():
    log = [_entry("click", "data-request"), _entry("click", "bound-1-target"),
           _entry("copy", "otp-field", value="482913")]
    p = replay_progress(_code_skill(), {"executed": 3, "failed_at": 3, "log": log,
                                        "error": "boom", "extracted": {"otp": "482913"}})

    assert p["done"] == ['clicked "Data Request"', 'clicked "PR/01797494"',
                         'captured otp from "One time passcode" = "482913"']
    assert p["attempted"] == 'click "Next page" 3×'
    assert p["remaining"] == [
        'paste the value noted as otp into "Please enter OTP character 1"',
        'click "Proceed Securely"']
    assert p["total"] == 6


def test_tier1_repeat_clicks_log_many_entries_for_one_call():
    """api.repeat_click records once PER CLICK, so api.executed is not a position in the
    call list. The brief must still land on the next CALL, not three calls along."""
    log = [_entry("click", "data-request"), _entry("click", "bound-1-target"),
           _entry("copy", "otp-field", value="482913")] \
        + [_entry("click", "next-page") for _ in range(3)]
    p = replay_progress(_code_skill(), {"executed": 6, "failed_at": 6, "log": log,
                                        "error": "boom", "extracted": {"otp": "482913"}})

    assert p["done"][-1] == 'clicked "Next page" 3×'
    assert p["attempted"] == \
        'paste the value noted as otp into "Please enter OTP character 1"'
    assert p["remaining"] == ['click "Proceed Securely"']


def test_a_repeat_that_broke_mid_way_is_blamed_on_itself():
    """2 of 3 clicks landed. The action in doubt is the repeat, not the call after it —
    and the count matters to the agent: it has to finish the repeat, not restart it."""
    log = [_entry("click", "data-request"), _entry("click", "bound-1-target"),
           _entry("copy", "otp-field")] + [_entry("click", "next-page") for _ in range(2)]
    p = replay_progress(_code_skill(), {"executed": 5, "failed_at": 5, "log": log,
                                        "error": "boom", "extracted": {}})

    assert p["attempted"] == 'click "Next page" 3×'
    assert p["done"][-1] == 'captured otp from "One time passcode"'
    assert p["remaining"][0] == \
        'paste the value noted as otp into "Please enter OTP character 1"'


def test_a_ledger_that_does_not_line_up_narrates_nothing():
    """Fails CLOSED. Naming actions that may not be the ones that ran is worse than the
    generic wording — a brief the agent cannot trust is what sends it acting on a page
    that is not in front of it."""
    log = [_entry("click", "data-request"), _entry("click", "some-other-control")]
    assert replay_progress(_code_skill(), {"executed": 2, "failed_at": 2, "log": log,
                                           "error": "boom", "extracted": {}}) is None


def test_unparseable_code_degrades_to_no_brief():
    skill = Skill(sid="s2", body="code", code="def (:", anchors={}, params={})
    assert replay_progress(skill, {"executed": 1, "failed_at": 1,
                                   "log": [_entry("click", "x")],
                                   "error": "boom", "extracted": {}}) is None


def test_the_optional_mark_is_not_narrated_as_an_action():
    """api.begin_optional() marks where the slice's declared error branch starts. It
    records nothing and touches no control, so it is not an action to report."""
    code = ('"""d"""\n\n\nasync def run(api):\n'
            "    await api.click('save')\n"
            "    await api.begin_optional()\n"
            "    await api.click('cancel')\n")
    skill = Skill(sid="s3", body="code", code=code, params={},
                  anchors={"save": {"expect_text": "Save"},
                           "cancel": {"expect_text": "Cancel"}})
    p = replay_progress(skill, {"executed": 1, "failed_at": 1,
                                "log": [_entry("click", "save")],
                                "error": "boom", "extracted": {}})

    assert p["done"] == ['clicked "Save"']
    assert p["attempted"] == 'click "Cancel"'
    assert p["total"] == 2                      # the mark is not one of the actions


def test_a_failure_after_the_last_action_says_so():
    """The ledger accounts for every call, yet the replay reported a failure — it broke
    after its last action (a post-click check), not on one of them. Saying "broke while
    attempting: None" would be worse than saying nothing."""
    log = [_entry("click", "data-request"), _entry("click", "bound-1-target"),
           _entry("copy", "otp-field")] \
        + [_entry("click", "next-page") for _ in range(3)] \
        + [_entry("paste", "code-box-1"), _entry("click", "proceed-securely")]
    p = replay_progress(_code_skill(), {"executed": 8, "failed_at": 8, "log": log,
                                        "error": "boom", "extracted": {}})

    assert p["attempted"] is None
    assert p["remaining"] == []
    assert p["summary"] == ("ran all 6 recorded action(s); the failure came after "
                            "the last one")


# ------------------------------- ran to the end, gate failed -------------------------------


def test_a_replay_that_ran_every_action_is_not_reported_as_partial():
    """The takeover branch fires for ANY failed segment, including one whose steps all
    ran and whose GATE then failed. Telling that agent the step "partially completed and
    then stopped" is false — and invites it to redo a save that already went through."""
    p = replay_progress(_steps_skill(), {"executed": 5, "failed_at": None, "log": [],
                                         "error": "segment gate failed", "extracted": {}})

    assert p["ran_to_end"] is True
    assert p["remaining"] == []
    assert p["done_count"] == 5
    assert "end check failed" in p["summary"]


# ---------------------------------- the prompt it produces ----------------------------------


def _prompt(progress, **kw):
    return scoped_subtask_prompt("do the OTP step", [], [], dirty=True,
                                 prior_failure="TimeoutError: x",
                                 replay_progress=progress, **kw)


def test_the_prompt_states_the_boundary_and_the_limits_of_its_own_evidence():
    p = replay_progress(_code_skill(),
                        {"executed": 2, "failed_at": 2,
                         "log": [_entry("click", "data-request"),
                                 _entry("click", "bound-1-target")],
                         "error": "boom", "extracted": {}})
    text = _prompt(p)

    assert "REPLAY PROGRESS" in text
    assert "ran 2 of its 6 actions" in text
    assert 'clicked "Data Request"' in text
    assert "EFFECT is NOT verified" in text          # dispatched != accomplished
    assert "Treat it as UNKNOWN" in text             # the step that broke
    assert "verify each one rather than replaying it blindly" in text
    # The generic paragraph is REPLACED, not stacked on top of the specific one.
    assert "partially completed it and then stopped" not in text


def test_the_ran_to_end_prompt_warns_against_redoing_a_save():
    p = replay_progress(_steps_skill(), {"executed": 5, "failed_at": None, "log": [],
                                         "error": "gate failed", "extracted": {}})
    text = _prompt(p)

    assert "ran ALL 5 of its actions" in text
    assert "duplicates the record" in text
    assert "partially completed it and then stopped" not in text


def test_without_a_brief_the_generic_dirty_paragraph_still_stands():
    text = scoped_subtask_prompt("do the step", [], [], dirty=True,
                                 prior_failure="boom", replay_progress=None)
    assert "partially completed it and then stopped" in text
    assert "REPLAY PROGRESS" not in text


@pytest.mark.parametrize("progress", [None, {}])
def test_a_clean_start_takes_no_brief_at_all(progress):
    text = scoped_subtask_prompt("do the step", [], [], dirty=False,
                                 replay_progress=progress)
    assert "REPLAY PROGRESS" not in text
    assert "partially completed it and then stopped" not in text
