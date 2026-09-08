"""The OTP false-pass: a segment whose only effect is in-page reported success.

Run 20260826_095932 pasted a dead OTP, clicked Proceed Securely, and PASSED — the paste
landed, the button was there, the URL is identical either side of the wall, and the server
answers 200 with the refusal inside the body. The run then marched on into a page that had
never opened and burned ~300k tokens authoring against it.

Three candidate signals were measured DEAD before this one (see the memory file): the
response body is never captured on acceptance (the page navigates and tears down before it
can be read — checked 12 runs, the only body ever captured came from the REJECTED run);
console/401 is identical in both; and the recording's own url + tab count are byte-identical
across the deciding step.

The document title is the one thing that differs: the anonymous wall sets no <title> so it
shows the raw URL, while the accepted view shows "Acting Office - Live Test". So a segment
that CHANGED the title must change it the same way on replay.

The title is read LIVE and never from the recording — see the note below _hist.

Pinned conservatively, because the user is already fighting false-fails:
  - only when the title actually changed during the segment (otherwise it proves nothing);
  - only when the end title carries no digits (a record ref or id would be volatile);
  - never for a segment that closed its own page (its surviving title is the same coin
    flip as its surviving url — library/7320039db9ba7e26 is exactly that segment);
  - demote-only, and fail-open on an unreadable title.
"""
import json

from automation.pipeline import hybrid
from automation.pipeline import subtask_store as ss
from automation.pipeline.hybrid import (Gate, _base_gate, _pin_end_title,
                                        evaluate_gate, run_hybrid_task)
from automation.tasks import SubtaskDecl, TaskSpec
from tests.test_bindings import stores  # noqa: F401 - fixture
from tests.test_close_tab_gate import CLOSER_RECORDING, _EndsAt, _as_subtask
from tests.test_hybrid import FakeSession, _runner, _seg


def _hist(*titles, tabs=1):
    return {"history": [
        {"state": {"url": "http://app/p", "interacted_element": [], "title": t,
                   "tabs": [{"url": "http://app/t%d" % i} for i in range(tabs)]},
         "model_output": {"action": [{"click": {"index": 1}}]},
         "result": []}
        for t in titles]}


def _write(tmp_path, doc):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(doc))
    return p


# ------------------------------- what gets pinned -------------------------------
#
# The title is read LIVE (at commit, or on a passed replay) and NEVER from the recording.
# browser-use captures each item's state before that item's actions, and this SPA updates
# document.title asynchronously, so a recorded title can be a whole page stale: subtask 0's
# final `done` item records url ".../rti/payrun" alongside title "Dashboard - …". Pinning
# that produced a live false-fail on the very first run.


def _rec(tmp_path, tabs=(1, 1)):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"history": [
        {"state": {"url": "http://app/p", "interacted_element": [],
                   "tabs": [{"url": "http://app/t%d" % i} for i in range(n)]},
         "model_output": {"action": [{"click": {"index": 1}}]}, "result": []}
        for n in tabs]}))
    return p


def test_a_title_the_segment_changed_is_pinned(tmp_path):
    assert _pin_end_title("Setting - Acting Office", "Acting Office",
                          _rec(tmp_path)) == "Acting Office"


def test_an_unchanged_title_pins_nothing(tmp_path):
    # It would prove nothing: the segment could have done anything or nothing at all.
    assert _pin_end_title("Payroll", "Payroll", _rec(tmp_path)) is None


def test_a_title_carrying_digits_pins_nothing(tmp_path):
    # A record ref or id in the title is volatile — pinning it invents a false-fail. This
    # also makes the learn-on-replay path safe: the OTP WALL's title is a raw URL full of
    # digits, so a false pass can never teach the gate the wrong title.
    assert _pin_end_title("Dashboard", "Request PR/01797494/27", _rec(tmp_path)) is None


def test_a_segment_that_closed_its_own_page_pins_nothing(tmp_path):
    """Its surviving title is the same coin flip as its surviving url — the trap that
    poisoned library/7320039db9ba7e26's end_context."""
    assert _pin_end_title("Acting Office", "Employee Approval Request",
                          _rec(tmp_path, tabs=(2, 1))) is None


def _nav_rec(tmp_path, *urls):
    """A recording whose items sit on `urls` in order — the last two decide whether the
    segment's final action navigated."""
    p = tmp_path / "nav.json"
    p.write_text(json.dumps({"history": [
        {"state": {"url": u, "interacted_element": [], "title": "",
                   "tabs": [{"url": u}]},
         "model_output": {"action": [{"click": {"index": 1}}]}, "result": []}
        for u in urls]}))
    return p


def test_a_segment_that_ended_on_a_navigation_pins_nothing(tmp_path):
    """Regression for run 20260907_152222 / library/50410b61b2e42519.

    "…select the business name CREAMOS LTD, then go to the Employee section" ends with
    api.click('employees'). This SPA updates document.title a whole navigation late, so the
    commit read "Payroll & RTI - …" — the title of the page it had just LEFT — and pinned
    it. Five consecutive runs then died at subtask 0: all eight steps ran, the url
    postcondition matched /paye/clients/*/, and only the pin disagreed.

    The lag can only mislead a segment whose last action navigated, and there the url gate
    already proves the move. Same trap, same slice, on 09-04 with "Best LIMITED"
    (92c267f7c8f0f0cc) — every company-name edit re-keys the slice and re-rolls the coin.
    """
    rec = _nav_rec(tmp_path, "http://app/paye", "http://app/c/x/rti/payrun",
                   "http://app/c/x/")

    assert _pin_end_title("Dashboard - Acting Office", "Payroll & RTI - Acting Office",
                          rec) is None


def test_a_segment_that_ended_in_place_still_pins(tmp_path):
    """The case the pin exists for — library/07044b6a0dbf7988, the OTP slice.

    Its last three states are all on /links/10/c/*/r/*/calcdatarequest: the anonymous wall
    sets no <title> so it shows the raw URL, and only after Proceed Securely does the title
    become "Acting Office - Live Test". end_context matches BOTH sides, so the title is the
    only thing that can tell an accepted OTP from a dead one.
    """
    rec = _nav_rec(tmp_path, "http://app/links/10/c/x/r/y/calcdatarequest",
                   "http://app/links/10/c/x/r/y/calcdatarequest")

    assert _pin_end_title("Setting - Acting Office", "Acting Office",
                          rec) == "Acting Office"


def test_an_unreadable_recording_does_not_block_the_pin(tmp_path):
    """Fail-open, in parity with _recording_closed_its_page: an unreadable diagnosis must
    never silently drop a legitimate pin."""
    bad = tmp_path / "broken.json"
    bad.write_text("{{ not json")

    assert _pin_end_title("Setting", "Acting Office", bad) == "Acting Office"
    assert _pin_end_title("Setting", "Acting Office", None) == "Acting Office"


def test_a_recording_with_one_readable_url_cannot_tell_and_pins(tmp_path):
    """Nothing to compare the final url against — same fail-open reading."""
    assert _pin_end_title("Setting", "Acting Office",
                          _nav_rec(tmp_path, "http://app/only")) == "Acting Office"


def test_a_missing_title_pins_nothing(tmp_path):
    assert _pin_end_title("Dashboard", "", _rec(tmp_path)) is None
    assert _pin_end_title("", "Dashboard", _rec(tmp_path)) is None


def test_whitespace_is_collapsed_before_comparing(tmp_path):
    assert _pin_end_title("A  B", "A B", _rec(tmp_path)) is None


# ------------------------------- how it gates -------------------------------


def _sub():
    return _as_subtask(SubtaskDecl(prompt="click Get OTP and paste it"))


def test_the_pin_rides_on_whatever_base_kind_resolves():
    gate = _base_gate(_sub(), {"end_title": "Acting Office"}, "/links/*/calcdatarequest")

    assert gate.kind == "steps"          # no end_context -> steps, unchanged
    assert gate.end_title == "Acting Office"


class _Page:
    def __init__(self, title):
        self._title = title

    async def title(self):
        if isinstance(self._title, Exception):
            raise self._title
        return self._title


async def test_a_replay_that_reached_the_recorded_title_passes():
    ok, detail = await evaluate_gate(
        Gate(kind="steps", end_title="Acting Office - Live Test"),
        steps_ok=True, page=_Page("Acting Office - Live Test"), requests_window=[])

    assert ok is True
    # Recorded on the PASS too, same convention as detail["checks"]: a gate carrying a pin
    # is not a bare gate. Without this there is no artifact evidence the check ever ran,
    # and a silently-inert gate looks exactly like a passing one.
    assert detail["end_title"] == {"expected": "Acting Office - Live Test",
                                   "reached": "Acting Office - Live Test", "ok": True}


async def test_a_replay_still_sitting_on_the_OTP_WALL_fails():
    """The exact false pass: the wall has no <title>, so the title is the raw URL."""
    ok, detail = await evaluate_gate(
        Gate(kind="steps", end_title="Acting Office - Live Test"),
        steps_ok=True,
        page=_Page("test.actingoffice.com/links/10/c/6a61d0/r/6a8ec7/calcdatarequest"),
        requests_window=[])

    assert ok is False
    assert detail["end_title"]["ok"] is False
    assert detail["end_title"]["expected"] == "Acting Office - Live Test"
    assert "calcdatarequest" in detail["end_title"]["reached"]


async def test_the_pin_is_demote_only():
    # It must never resurrect a segment the base gate already failed.
    ok, _ = await evaluate_gate(
        Gate(kind="steps", end_title="Acting Office"),
        steps_ok=False, page=_Page("Acting Office"), requests_window=[])

    assert ok is False


async def test_an_unreadable_title_does_not_demote():
    # Fail OPEN: an additive check must not become a new false-fail source.
    ok, detail = await evaluate_gate(
        Gate(kind="steps", end_title="Acting Office"),
        steps_ok=True, page=_Page(RuntimeError("page closed")), requests_window=[])

    assert ok is True
    assert "end_title" not in detail          # nothing was observed, so nothing is claimed


async def test_a_title_mismatch_reports_a_readable_reason():
    from automation.pipeline.hybrid import _check_failure_reason

    _, detail = await evaluate_gate(
        Gate(kind="steps", end_title="Acting Office - Live Test"),
        steps_ok=True, page=_Page("test.actingoffice.com/links/10/c/6a/calcdatarequest"),
        requests_window=[])

    why = _check_failure_reason(detail)
    assert why is not None
    assert "Acting Office - Live Test" in why


async def test_a_gate_with_no_pin_is_untouched():
    ok, detail = await evaluate_gate(
        Gate(kind="steps"), steps_ok=True, page=_Page("anything"), requests_window=[])

    assert ok is True
    assert "end_title" not in detail


# ------------------------------- the commit -------------------------------


def _navigating(*titles):
    """Same shape as _hist but with an action the compiler can actually anchor, so the
    segment commits (a zero-step compile is never committed)."""
    doc = _hist(*titles)
    for item in doc["history"]:
        item["model_output"] = {"action": [{"navigate": {"url": "http://app/portal"}}]}
    return doc


OTP_SHAPED = _navigating("Setting - Acting Office", "Acting Office")


async def test_the_commit_records_the_end_title(stores, monkeypatch):
    prompt = "click Get OTP and paste the code"
    spec = TaskSpec(key="t", prompt=prompt, subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")],
                   urls=["http://app/section", "http://app/portal"],
                   titles=["Setting - Acting Office", "Acting Office"])
    fake.recording = OTP_SHAPED
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    sid = ss.subtask_id(prompt, ss.normalize_context("http://app/section"))
    assert ss.load_manifest()[sid]["end_title"] == "Acting Office"


async def test_a_close_tab_segment_commits_no_end_title(stores, monkeypatch):
    prompt = "do the thing and then close this tab"
    spec = TaskSpec(key="c", prompt=prompt, subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")],
                   urls=["http://app/section", "http://app/datarequests"],
                   titles=["Acting Office", "Employee Approval Request"])
    fake.recording = CLOSER_RECORDING
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    sid = ss.subtask_id(prompt, ss.normalize_context("http://app/section"))
    entry = ss.load_manifest()[sid]
    assert not entry.get("end_title")
    assert not entry.get("end_context")


# ------------------------- learning the pin from a passed replay -------------------------
#
# Existing entries always REPLAY and never re-author, so without this the pin would never
# reach the very entry it was built for (library/07044b6a0dbf7988, the OTP slice). A replay
# that PASSED its base gate demonstrably reached the intended end state, so the title read
# at that moment is trustworthy — and better evidence than the recording's lagging capture.
# The digit rule is what makes it safe: a false pass leaves the page on the OTP wall, whose
# title is a raw URL full of digits, so a wrong title can never be learned.


async def _replay_once(monkeypatch, *, titles, seg_ok=True):
    prompt = "click Get OTP and paste the code"
    spec = TaskSpec(key="t", prompt=prompt, subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")],
                   urls=["http://app/section", "http://app/portal"],
                   titles=["Setting - Acting Office", "Acting Office"])
    fake.recording = OTP_SHAPED
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)
    sid = ss.subtask_id(prompt, ss.normalize_context("http://app/section"))

    # Wipe the pin the authoring run left, to stand in for every entry committed before
    # this feature existed.
    ss.update_manifest(sid, prompt, create=True, end_title=None)

    # A failed replay hands over to the agent, so keep one queued for that path.
    # executed=1 on the failing replay makes the takeover DIRTY, so the agent path cannot
    # commit either — leaving the replay as the only thing that could have taught a pin.
    fake2 = _EndsAt(_runner(), replays=[_seg(seg_ok, executed=0 if seg_ok else 1)],
                    agents=[_seg(True, mode="authored")],
                    urls=["http://app/section", "http://app/portal"], titles=list(titles))
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    await run_hybrid_task(fake2.runner or _runner(), prompt, spec=spec)
    return ss.load_manifest()[sid]


async def test_a_passed_replay_teaches_the_entry_its_end_title(stores, monkeypatch):
    entry = await _replay_once(monkeypatch,
                               titles=["Setting - Acting Office", "Acting Office"])

    assert entry["end_title"] == "Acting Office"


async def test_a_replay_left_on_the_OTP_WALL_teaches_nothing(stores, monkeypatch):
    """The safety property. A URL-as-title is full of digits, so even if a false pass
    slipped through, the gate can never learn the wall as the expected end state."""
    entry = await _replay_once(monkeypatch, titles=[
        "Setting - Acting Office",
        "test.actingoffice.com/links/10/c/6a61d0ab/r/6a8ec70b/calcdatarequest"])

    assert not entry.get("end_title")


async def test_a_FAILED_replay_teaches_nothing(stores, monkeypatch):
    entry = await _replay_once(monkeypatch, seg_ok=False,
                               titles=["Setting - Acting Office", "Acting Office"])

    assert not entry.get("end_title")


# ------------------------------- settling the read -------------------------------
#
# Reading the title LIVE is not enough, which is what run 20260903_091650_671199 proved.
# This SPA updates document.title a whole navigation behind, so the commit's single
# immediate read picked up the PREVIOUS page's title and pinned it; evaluate_gate then read
# the settled title through _settled() and could never agree with it. Commit reads early,
# gate reads late — so the commit now gets the same settle budget the gate will use.


def _reader(*titles):
    """A current_title() whose successive calls walk `titles` (the last one repeats)."""
    seen = list(titles)

    async def read():
        return seen[0] if len(seen) == 1 else seen.pop(0)
    return read


async def test_the_read_waits_for_the_title_to_stop_changing(monkeypatch):
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    # The exact live shape: the Employees page still reporting the previous page's title.
    read = _reader("Payroll & RTI - Acting Office", "Employees - Acting Office",
                   "Employees - Acting Office")

    assert await hybrid._settled_title(read) == "Employees - Acting Office"


async def test_an_already_settled_title_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)

    assert await hybrid._settled_title(_reader("Acting Office")) == "Acting Office"


async def test_a_title_that_never_settles_returns_its_last_read(monkeypatch):
    """Best effort, not a hang. _pin_end_title's own guards still apply to whatever
    comes back, and a pin is demote-only in the first place."""
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    churn = _reader(*[f"T{i}" for i in range(50)])

    assert (await hybrid._settled_title(churn)).startswith("T")


async def test_an_unreadable_title_settles_immediately(monkeypatch):
    """No page, no wait: an empty title pins nothing anyway, so polling it is pure delay
    on every commit that has nothing to pin."""
    calls = {"n": 0}

    async def read():
        calls["n"] += 1
        return ""

    assert await hybrid._settled_title(read) == ""
    assert calls["n"] == 1


async def test_the_commit_pins_the_settled_title_not_the_lagging_one(stores, monkeypatch):
    """Regression for run 20260903_091650_671199 / library/77a0af5f643402f2.

    The segment ended on the Employees page, but document.title still read
    "Payroll & RTI - …" at the instant the commit looked. That value became the expected
    end state, and the entry then false-failed four consecutive runs — its steps all ran,
    its url gate matched, and only the pin disagreed.
    """
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    prompt = "go to the Payroll module and then to the Employee section"
    spec = TaskSpec(key="t", prompt=prompt, subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")],
                   urls=["http://app/section", "http://app/portal"],
                   titles=["Dashboard - Acting Office", "Payroll & RTI - Acting Office",
                           "Employees - Acting Office", "Employees - Acting Office"])
    fake.recording = _navigating("Dashboard - Acting Office", "Employees - Acting Office")
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    sid = ss.subtask_id(prompt, ss.normalize_context("http://app/section"))
    assert ss.load_manifest()[sid]["end_title"] == "Employees - Acting Office"


def _navigating_away(*titles):
    """`_navigating`, but the LAST item sits on a different url — the segment's final
    action moved the page. `_hist` otherwise keeps every item on http://app/p."""
    doc = _navigating(*titles)
    doc["history"][-1]["state"]["url"] = "http://app/moved"
    return doc


async def test_the_commit_pins_nothing_when_the_last_action_navigated(stores,
                                                                     monkeypatch):
    """The live shape of run 20260907_152222 / library/50410b61b2e42519, end to end.

    The authoring run passed and committed "Payroll & RTI - …" for a segment that had
    already reached the Employees page — the title lags a whole navigation. Five runs then
    died on it, and the takeover agent was re-gated against the same pin, so the entry
    could never retire itself either.
    """
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    prompt = "go to the Payroll module and then to the Employee section"
    spec = TaskSpec(key="n", prompt=prompt, subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")],
                   urls=["http://app/section", "http://app/moved"],
                   titles=["Dashboard - Acting Office",
                           "Payroll & RTI - Acting Office",
                           "Payroll & RTI - Acting Office",
                           "Payroll & RTI - Acting Office"])
    fake.recording = _navigating_away("Dashboard - Acting Office",
                                      "Payroll & RTI - Acting Office")
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    sid = ss.subtask_id(prompt, ss.normalize_context("http://app/section"))
    assert ss.load_manifest()[sid].get("end_title") is None
