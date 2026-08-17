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
    assert anchor["selectors"][0] == 'css=[id="save-v2"]'   # identity attrs lead
    assert 'css=[id="save-v2"]' in anchor["selectors"]
    assert 'css=[id="old-save"]' in anchor["selectors"]      # old anchors stay as fallbacks
    assert anchor["fingerprint"]["attrs"]["id"] == "save-v2"


def test_promote_healed_anchors_ignores_unknown_handles(stores):
    sid = "heal2"
    ss.anchors_path(sid).write_text(json.dumps({"a": {"selectors": ["x"]}}))
    log = [{"step": 0, "handle": "ghost", "healed": {"tag": "div", "attrs": {}}}]
    assert promote_healed_anchors(ss.anchors_path(sid), log) == []


# ---------------- repeat_click / click_indexed replay semantics (2026-08-12) ----------------


class _FakeReadyPage:
    """Scripted page for the repeat/indexed verbs: wait_for_timeout is recorded, and
    the readiness locator reports ready after `ready_after` polls."""

    def __init__(self, ready_after=0):
        self.waits = []
        self.ready_after = ready_after
        self.polls = 0

    async def wait_for_timeout(self, ms):
        self.waits.append(int(ms))

    def locator(self, sel):
        page = self

        class _Loc:
            @property
            def first(self):
                return self

            async def count(self):
                page.polls += 1
                return 1 if page.polls > page.ready_after else 0

            async def is_visible(self):
                return True

            async def is_enabled(self):
                return True

        return _Loc()


async def test_repeat_click_clicks_count_times_with_floor_and_poll(monkeypatch):
    from automation.skills import api as api_mod

    clicks = []

    async def fake_click(page, step, timeout_ms):
        clicks.append(step["selectors"][0])
        return step["selectors"][0], None

    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)
    page = _FakeReadyPage()
    api = api_mod.SkillApi(
        page, {"save-next": {"selectors": ['role=button[name="Save & Next"]']}})
    await api.repeat_click("save-next", 5, 1.0)
    assert len(clicks) == 5
    assert len([e for e in api.log if e["action"] == "click"]) == 5
    assert page.waits.count(1000) == 4       # the recorded floor between clicks only


async def test_repeat_click_raises_when_target_never_ready(monkeypatch):
    from automation.skills import api as api_mod

    async def fake_click(page, step, timeout_ms):
        return "sel", None

    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)
    monkeypatch.setattr(api_mod, "_REPEAT_READY_CAP_S", 0.05)
    page = _FakeReadyPage(ready_after=float("inf"))   # never becomes ready
    api = api_mod.SkillApi(page, {"save-next": {"selectors": ["css=[id=btnSave]"]}})
    with pytest.raises(RuntimeError, match="ready"):
        await api.repeat_click("save-next", 2, 0.0)


async def test_click_indexed_substitutes_each_index(monkeypatch):
    from automation.skills import api as api_mod

    seen = []

    class _Loc:
        async def click(self, timeout=None, force=False):
            pass

    async def fake_rws(page, step, timeout_ms, rounds=4):
        seen.append(step["selectors"][0])
        return _Loc(), step["selectors"][0]

    monkeypatch.setattr(api_mod, "_resolve_with_scroll", fake_rws)
    page = _FakeReadyPage()
    api = api_mod.SkillApi(
        page, {"boxes": {"selector_template": 'css=[id$="-{n}-checkbox"]'}})
    await api.click_indexed("boxes", 0, 20)
    assert seen == [f'css=[id$="-{n}-checkbox"]' for n in range(20)]
    assert len([e for e in api.log if e["action"] == "click"]) == 20


async def test_resolve_with_scroll_scrolls_until_found(monkeypatch):
    from automation.pipeline import script_compile as sc

    calls = {"resolve": 0, "wheel": 0, "tops": 0}

    async def fake_resolve(page, step, timeout_ms):
        calls["resolve"] += 1
        if calls["resolve"] < 3:
            raise RuntimeError("no unique candidate matched")
        return "LOC", "sel-win", None

    async def fake_wheel(page, pages, down=True):
        calls["wheel"] += 1

    async def fake_tops(page):
        calls["tops"] += 1
        return 1

    monkeypatch.setattr(sc, "_resolve", fake_resolve)
    monkeypatch.setattr(sc, "_wheel_scroll", fake_wheel)
    monkeypatch.setattr(sc, "_scroll_tops", fake_tops)
    loc, sel = await sc._resolve_with_scroll(None, {"selectors": ["css=x"]}, 2500)
    assert (loc, sel) == ("LOC", "sel-win")
    # First miss sweeps from the TOP (run 20260817_133135: a down-only hunt from
    # mid-page can never reach a target above it), later misses wheel down; a SUCCESS
    # leaves the scroll where the match was found — no trailing reset.
    assert calls["tops"] == 1
    assert calls["wheel"] == 1


async def test_run_steps_honours_count_and_click_indexed(monkeypatch):
    from automation.pipeline import script_compile as sc

    clicks, indexed = [], []

    async def fake_flyout(page, steps, idx, timeout_ms):
        clicks.append(idx)
        return "sel", None

    class _Loc:
        async def click(self, timeout=None, force=False):
            indexed.append("click")

    async def fake_rws(page, step, timeout_ms, rounds=4):
        return _Loc(), step["selectors"][0]

    monkeypatch.setattr(sc, "_click_with_flyout_recovery", fake_flyout)
    monkeypatch.setattr(sc, "_resolve_with_scroll", fake_rws)
    page = _FakeReadyPage()
    out = await sc.run_steps(page, [
        {"action": "click", "selectors": ["css=[id=btnSave]"], "count": 3,
         "repeat_wait_s": 1.0},
        {"action": "click_indexed", "selector_template": 'css=[id$="-{n}-box"]',
         "start": 2, "count": 4},
    ])
    assert out["error"] is None
    assert len(clicks) == 3
    assert len(indexed) == 4


# ---------------- guarded xpath-first resolution (2026-08-12) ----------------


class _FpNode:
    def __init__(self, tag, text, visible=True, attrs=None):
        self._tag, self._text, self._visible = tag, text, visible
        self._attrs = attrs or {}

    async def wait_for(self, state=None, timeout=None):
        return None

    async def is_visible(self):
        return self._visible

    async def inner_text(self, timeout=None):
        return self._text

    async def get_attribute(self, name):
        return None

    async def evaluate(self, expr):
        return {"tag": self._tag, "attrs": dict(self._attrs)}

    async def is_editable(self):
        return False


class _FpLocator:
    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def first(self):
        return self._nodes[0] if self._nodes else _FpNode("div", "", visible=False)

    async def count(self):
        return len(self._nodes)

    def nth(self, n):
        return self._nodes[n]


class _FpPage:
    def __init__(self, mapping):
        self._m = mapping

    def locator(self, sel):
        return _FpLocator(self._m.get(sel, []))

    async def wait_for_timeout(self, ms):
        return None


def test_selectors_are_xpath_then_hard_identity_only():
    """2026-08-13: the ladder locates, it never searches — xpath first, then attribute
    identity. `role=[name=…]` and `text="…"` (both name/text lookups) are gone."""
    from automation.pipeline.script_compile import _selectors
    sels = _selectors({"node_name": "button", "ax_name": "Save & Next",
                       "attributes": {"id": "btnSave", "aria-label": "Save and next"},
                       "x_path": "html/body/div[1]/button"})
    assert sels == ["xpath=/html/body/div[1]/button", 'css=[id="btnSave"]',
                    'css=[aria-label="Save and next"]']
    # An element with neither location nor identity is unanchorable by design.
    assert _selectors({"node_name": "div", "ax_name": "Some text", "attributes": {}}) == []


async def test_resolve_xpath_hit_gated_by_identity_not_text():
    """The gate compares IDENTITY (tag + hard attrs), never text: an extract anchored on
    a fresh-data block shows different text every run, and comparing it vetoed exactly
    the xpath the anchor exists for (run 20260813_132507, seg 1)."""
    from automation.pipeline import script_compile as sc
    step = {"selectors": ["xpath=/html/body/div[2]", 'css=[id="ok"]'],
            "fingerprint": {"tag": "div", "attrs": {"id": "identity-card"}}}
    # Same element, brand-new data in it -> the xpath still wins.
    fresh = _FpPage({"xpath=/html/body/div[2]": [
        _FpNode("div", "Struan Boyd 5 Long Acre LEEDS", attrs={"id": "identity-card"})]})
    _loc, sel, _ = await sc._resolve(fresh, step, 500)
    assert sel == "xpath=/html/body/div[2]"
    # Positional drift: the slot now holds a different element -> refuse, fall through.
    drifted = _FpPage({"xpath=/html/body/div[2]": [
                          _FpNode("div", "anything", attrs={"id": "advert"})],
                       'css=[id="ok"]': [_FpNode("div", "x", attrs={"id": "ok"})]})
    _loc2, sel2, _ = await sc._resolve(drifted, step, 500)
    assert sel2 == 'css=[id="ok"]'
