"""Skill layer: tier preference (code > steps), instantiation, the execute harness for
generated code, and anchor-keyed heal promotion."""
import json

import pytest

from automation.pipeline import subtask_store as ss
from automation.pipeline.decompose import Subtask
from automation.skills import Skill, execute, load_skill, promote_healed_anchors
from automation.skills.base import _execute_code


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    (tmp_path / "library").mkdir()
    return tmp_path


class StubApi:
    """Records calls in order; can be told to blow up at a given ledger position."""

    def __init__(self, fail_at=None):
        self.calls = []
        self.log = []
        self.executed = 0
        self.fail_at = fail_at

    async def _do(self, verb, *args):
        if self.fail_at is not None and self.executed == self.fail_at:
            raise RuntimeError("boom")
        self.calls.append((verb, *args))
        self.log.append({"step": self.executed, "action": verb})
        self.executed += 1

    async def click(self, handle):
        await self._do("click", handle)

    async def fill(self, handle, value, clear=True):
        await self._do("fill", handle, str(value))

    async def select_option(self, label):
        await self._do("select_option", str(label))

    async def press(self, keys):
        await self._do("press", keys)

    async def wait(self, seconds):
        await self._do("wait", seconds)

    async def goto(self, url):
        await self._do("goto", url)


GOOD_CODE = (
    '"""t"""\n\n\n'
    "async def run(api, *, business_name='Acme Ltd'):\n"
    "    await api.click('clients')\n"
    "    await api.fill('search', business_name)\n"
    "    await api.press('Enter')\n"
    "    await api.select_option(business_name)\n"
)


# ------------------------------- tier 0 (steps) -------------------------------


def test_load_skill_concrete_entry(stores):
    sid = "abc123"
    steps = [{"action": "click", "selectors": ["text=Go"]}]
    ss.steps_path(sid).write_text(json.dumps(steps))
    ss.update_manifest(sid, "go to the section")

    skill = load_skill(sid, Subtask(index=0, template_prompt="go to the section"))
    assert skill is not None
    assert skill.body == "steps" and skill.steps == steps and len(skill) == 1


def test_load_skill_missing_or_unreadable_returns_none(stores):
    assert load_skill("nope", Subtask(index=0, template_prompt="x")) is None
    sid = "bad"
    ss.steps_path(sid).write_text("{not json")
    assert load_skill(sid, Subtask(index=0, template_prompt="x")) is None


async def test_execute_unknown_body_fails_closed(stores):
    out = await execute(Skill(sid="s", body="wat"), page=None)
    assert out["failed_at"] == 0
    assert "unknown skill body" in out["error"]


# ------------------------------- tier 1 (code) -------------------------------


def _seed_code_entry(sid, *, code=GOOD_CODE, anchors=None, params=None):
    ss.steps_path(sid).write_text(json.dumps([{"action": "click", "selectors": ["t"]}]))
    ss.code_path(sid).write_text(code)
    ss.anchors_path(sid).write_text(json.dumps(anchors or {
        "clients": {"selectors": ['role=link[name="Clients"]']},
        "search": {"selectors": ['css=[placeholder="Search"]']},
    }))
    ss.update_manifest(sid, "select a business", **({"params": params} if params else {}))


def test_load_skill_prefers_code_tier(stores):
    sid = "code1"
    _seed_code_entry(sid)
    skill = load_skill(sid, Subtask(index=0, template_prompt="select a business"))
    assert skill is not None and skill.body == "code"
    assert "async def run" in skill.code
    assert len(skill) == 4          # awaited api calls


def test_broken_code_falls_back_to_steps(stores):
    sid = "code2"
    _seed_code_entry(sid, code="async def run(api):\n    import os\n")
    skill = load_skill(sid, Subtask(index=0, template_prompt="select a business"))
    assert skill is not None and skill.body == "steps"


def test_code_tier_aligns_params_and_substitutes_anchor_tokens(stores):
    sid = "code3"
    _seed_code_entry(
        sid,
        anchors={"business_name-target":
                 {"selectors": ['role=link[name="{{business_name}}"]']}},
        params={"business_name": "Acme Ltd"},
    )
    ss.template_path(sid).write_text(json.dumps({
        "source_prompt": "select business Acme Ltd",
        "params": {"business_name": "Acme Ltd"},
        "steps": [],
    }))
    sub = Subtask(index=0, template_prompt="select business {{business}}",
                  values={"business": "Zeta Inc"})
    skill = load_skill(sid, sub)
    assert skill is not None and skill.body == "code"
    assert skill.params == {"business_name": "Zeta Inc"}
    assert skill.anchors["business_name-target"]["selectors"] == \
        ['role=link[name="Zeta Inc"]']


async def test_execute_code_runs_calls_in_order(stores):
    stub = StubApi()
    skill = Skill(sid="s", body="code", code=GOOD_CODE,
                  params={"business_name": "Zeta Inc"})
    out = await _execute_code(skill, page=None, timeout_ms=1000, api=stub)
    assert out["failed_at"] is None and out["executed"] == 4
    assert stub.calls == [("click", "clients"), ("fill", "search", "Zeta Inc"),
                          ("press", "Enter"), ("select_option", "Zeta Inc")]


async def test_execute_code_reports_failing_call(stores):
    stub = StubApi(fail_at=1)
    skill = Skill(sid="s", body="code", code=GOOD_CODE, params={})
    out = await _execute_code(skill, page=None, timeout_ms=1000, api=stub)
    assert out["failed_at"] == 1 and out["executed"] == 1
    assert "boom" in out["error"]
    assert [e["action"] for e in out["log"]] == ["click"]


async def test_execute_code_lints_before_running(stores):
    evil = "async def run(api):\n    __import__('os')\n"
    out = await execute(Skill(sid="s", body="code", code=evil), page=None)
    assert out["failed_at"] == 0
    assert "lint" in out["error"]


# ------------------------------- heal promotion (anchors) -------------------------------


def test_promote_healed_anchors_prepends_winner_selectors(stores):
    sid = "heal1"
    ss.anchors_path(sid).write_text(json.dumps({
        "save": {"selectors": ['css=[id="old-save"]'],
                 "fingerprint": {"tag": "button", "text": "Save"}},
    }))
    log = [{"step": 0, "action": "click", "handle": "save", "used": "healed:button#s2",
            "healed": {"tag": "button", "role": "button", "text": "Save",
                       "attrs": {"id": "save-v2"}}}]
    promoted = promote_healed_anchors(ss.anchors_path(sid), log)
    assert promoted == ["save"]
    anchor = json.loads(ss.anchors_path(sid).read_text())["save"]
    assert anchor["selectors"][0] == 'role=button[name="Save"]'
    assert 'css=[id="save-v2"]' in anchor["selectors"]
    assert 'css=[id="old-save"]' in anchor["selectors"]      # old anchors stay as fallbacks
    assert anchor["fingerprint"]["attrs"]["id"] == "save-v2"


def test_promote_healed_anchors_ignores_unknown_handles(stores):
    sid = "heal2"
    ss.anchors_path(sid).write_text(json.dumps({"a": {"selectors": ["x"]}}))
    log = [{"step": 0, "handle": "ghost", "healed": {"tag": "div", "attrs": {}}}]
    assert promote_healed_anchors(ss.anchors_path(sid), log) == []
