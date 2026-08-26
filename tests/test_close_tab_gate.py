"""A segment that closes its own page must not pin an end context.

Run 20260826_130102 stopped at subtask 8 ("… click submit, and then close this tab") with

    gate failed: {'end_context': 'blank',
                  'reached': 'test.actingoffice.com/paye/clients/*/datarequests'}

The agent did the work. When that subtask closes its page, whatever current_url() returns
is an accident of which browser target survives — and _author_segment committed the
accident, after which _base_gate turned it into every later run's postcondition. Five runs
show both survivors occurring for the SAME segment: 124837 ended on about:blank (and
passed by coincidence), 130102 ended on the app's datarequests tab (and failed).

That is why the guard keys on what the segment DID — the recording's tab count shrank —
and not on whether the surviving URL looks degenerate. A value-only guard would happily
commit 130102's perfectly ordinary '/paye/clients/*/datarequests' and fail the next
blank-survivor run instead: the same bug with the values swapped.
"""
import json

from automation.pipeline import hybrid
from automation.pipeline import subtask_store as ss
from automation.pipeline.hybrid import (_base_gate, _is_degenerate_url,
                                        _recording_closed_its_page, run_hybrid_task)
from automation.tasks import SubtaskDecl, TaskSpec
from tests.test_bindings import stores  # noqa: F401 - fixture
from tests.test_hybrid import FakeSession, _runner, _seg


def _item(n_tabs, action="click"):
    return {"state": {"url": "http://app/portal", "interacted_element": [],
                      "tabs": [{"url": "http://app/t%d" % i} for i in range(n_tabs)]},
            "model_output": {"action": [{action: {"index": 1}}]},
            "result": []}


def _write(tmp_path, history):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"history": history}))
    return p


# ------------------------------- the predicate -------------------------------


def test_a_recording_whose_tab_count_shrinks_closed_its_page(tmp_path):
    assert _recording_closed_its_page(_write(tmp_path, [_item(1), _item(2), _item(1)]))


def test_a_recording_with_a_steady_tab_count_did_not(tmp_path):
    assert not _recording_closed_its_page(_write(tmp_path, [_item(2), _item(2)]))


def test_a_recording_that_OPENED_a_tab_did_not_close_its_page(tmp_path):
    # The mirror of _stamp_opens_tab; growth must never read as a close.
    assert not _recording_closed_its_page(_write(tmp_path, [_item(1), _item(2)]))


def test_a_single_item_recording_did_not(tmp_path):
    assert not _recording_closed_its_page(_write(tmp_path, [_item(1)]))


def test_an_unreadable_recording_does_not_suppress_the_end_context(tmp_path):
    # Fail OPEN: guessing "closed" would silently drop a legitimate postcondition gate.
    assert not _recording_closed_its_page(tmp_path / "missing.json")


def test_degenerate_urls_are_the_ones_naming_no_location():
    assert _is_degenerate_url("about:blank")
    assert _is_degenerate_url("")
    assert _is_degenerate_url(None)
    assert not _is_degenerate_url("http://app/section")


# ------------------------------- the heal -------------------------------


def test_a_stored_blank_end_context_no_longer_gates():
    """library/7320039db9ba7e26 carries end_context 'blank'. Step 1 only stops NEW bad
    commits, so the existing entry must stop gating on read or the next run still fails."""
    sub = SubtaskDecl(prompt="click submit, and then close this tab")
    gate = _base_gate(_as_subtask(sub), {"end_context": "blank"}, "/links/*/calcdatarequest")

    assert gate.kind == "steps"


def test_a_real_end_context_still_gates():
    sub = SubtaskDecl(prompt="go to the data requests page")
    gate = _base_gate(_as_subtask(sub), {"end_context": "/paye/clients/*/datarequests"},
                      "/paye/clients/*/rti/payrun")

    assert gate.kind == "postcondition"
    assert gate.end_context == "/paye/clients/*/datarequests"


def _as_subtask(decl):
    """A Subtask-shaped stand-in: _base_gate reads only these attributes."""
    from types import SimpleNamespace
    return SimpleNamespace(marker=None, postcondition=None, fallback=False,
                           template_prompt=decl.prompt, instantiated_prompt=decl.prompt)


# ------------------------------- the commit -------------------------------


class _EndsAt(FakeSession):
    """FakeSession whose current_url()/current_title() walk scripted lists (last repeats)."""

    def __init__(self, *args, urls=None, titles=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._urls = list(urls or ["http://app/section"])
        self._titles = list(titles or [""])

    async def current_url(self):
        return self._urls[0] if len(self._urls) == 1 else self._urls.pop(0)

    async def current_title(self):
        return self._titles[0] if len(self._titles) == 1 else self._titles.pop(0)


CLOSER_RECORDING = {"history": [
    {"state": {"url": "http://app/portal", "interacted_element": [],
               "tabs": [{"url": "http://app/section"}, {"url": "http://app/portal"}]},
     "model_output": {"action": [{"navigate": {"url": "http://app/portal"}}]},
     "result": []},
    {"state": {"url": "http://app/section", "interacted_element": [],
               "tabs": [{"url": "http://app/section"}]},
     "model_output": {"action": [{"done": {}}]},
     "result": []},
]}


async def _commit_once(monkeypatch, *, recording, urls):
    """Author one subtask and return its manifest entry."""
    prompt = "do the thing and then close this tab"
    spec = TaskSpec(key="close", prompt=prompt,
                    subtasks=(SubtaskDecl(prompt=prompt),))
    fake = _EndsAt(_runner(), agents=[_seg(True, mode="authored")], urls=list(urls))
    fake.recording = recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)
    sid = ss.subtask_id(prompt, ss.normalize_context(urls[0]))
    return ss.load_manifest().get(sid)


async def test_a_segment_that_closed_its_page_commits_no_end_context_even_on_a_normal_url(
        stores, monkeypatch):
    """THE regression. Run 130102's survivor was the app's own datarequests tab — an
    ordinary URL a value-only guard would commit, poisoning every blank-survivor run
    after it. The recording says the page was closed; that is what must decide."""
    entry = await _commit_once(monkeypatch, recording=CLOSER_RECORDING,
                               urls=["http://app/section", "http://app/datarequests"])

    assert entry is not None
    assert not entry.get("end_context")


async def test_a_segment_ending_on_about_blank_commits_no_end_context(stores, monkeypatch):
    entry = await _commit_once(monkeypatch, recording=CLOSER_RECORDING,
                               urls=["http://app/section", "about:blank"])

    assert entry is not None
    assert not entry.get("end_context")


async def test_a_segment_ending_on_a_closed_page_never_commits_the_app_root(
        stores, monkeypatch):
    """current_url() returns "" for a closed page and normalize_context("") is "/" —
    indistinguishable from a legitimate app root, so this one is worse than 'blank'."""
    entry = await _commit_once(monkeypatch, recording=CLOSER_RECORDING,
                               urls=["http://app/section", ""])

    assert entry is not None
    assert entry.get("end_context") != "/"
    assert not entry.get("end_context")


async def test_an_ordinary_segment_still_commits_its_real_end_context(stores, monkeypatch):
    from tests.test_hybrid import FAKE_RECORDING

    entry = await _commit_once(monkeypatch, recording=FAKE_RECORDING,
                               urls=["http://app/section", "http://app/datarequests"])

    assert entry is not None
    assert entry["end_context"] == "/datarequests"
