"""scroll_panels: the live-agent twin of replay's _scroll_containers.

Motivating failure (run 20260817_110501 seg 6, and 20260814_105247 / 20260817_093555
before it): the Add Data Request panel is a FIXED side panel owning its own scroll box;
the agent's page-level scrolls moved the Data Request table BEHIND it (video shows the
background at rows 53-70 while the panel list sat partway). find_by_text's fallback
already container-scrolls, but its 6 rounds x 0.8 panel-heights ran out mid-list once
the employee list grew past ~60 rows (every run appends an employee), so the fallback
honestly reported "no match" from the middle of the list. The tool gives the agent an
explicit, repeatable container-scroll; the raised rounds cap lets the existing fallback
traverse a grown list (its moved==0 break keeps a high cap free on short lists)."""
from test_compile_coverage import _item, _write

from automation.pipeline import agent_tools
from automation.pipeline.script_compile import compile_recording


def _tool():
    return agent_tools.build_tools().registry.registry.actions["scroll_panels"]


def test_scroll_panels_is_registered_and_advertises_panels():
    desc = _tool().description
    assert "panel" in desc.lower()
    assert "container" in desc.lower()


def test_scroll_panels_warns_that_prescroll_indexes_go_stale():
    # Run 20260817_124339 seg 6: the agent batched [scroll, scroll, click(index),
    # click(Save)] in ONE step — the indexes came from the PRE-scroll page, and the
    # click ticked Aaran Duncan's checkbox instead of the noted employee's. Both the
    # description and the moved receipt must say pre-scroll indexes are STALE.
    assert "STALE" in _tool().description


async def test_scroll_panels_reports_moved_count(monkeypatch):
    seen = {}

    async def fake_eval(session, expr, **kw):
        seen["expr"] = expr
        return 3

    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)
    res = await _tool().function(pages=0.8, browser_session=object())

    assert res.error is None
    assert "3 container" in res.extracted_content
    assert "STALE" in res.extracted_content   # pre-scroll indexes must be flagged dead
    assert "0.8" in seen["expr"]          # the fraction reached the shared JS


async def test_scroll_panels_zero_moved_is_honest_not_an_error(monkeypatch):
    async def fake_eval(session, expr, **kw):
        return 0

    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)
    res = await _tool().function(browser_session=object())

    # End-of-list is information (the list is fully revealed), never a failure.
    assert res.error is None
    assert "no container moved" in res.extracted_content
    assert "end" in res.extracted_content


def test_panel_scroll_rounds_cover_a_long_employee_list():
    # 6 rounds ran out mid-list on ~80 employees (run 20260817_110501): the fallback
    # exited from the middle of the panel and the agent concluded the employee was not
    # in the list. The loop breaks on moved==0, so a high cap costs nothing when lists
    # are short — it must comfortably out-scroll a list that grows every run.
    assert agent_tools._PANEL_SCROLL_ROUNDS >= 20


def test_scroll_panels_compiles_to_its_own_replay_step(tmp_path):
    # Compiling it into a viewport "scroll" step would replay as a wheel over the page
    # BEHIND the panel — the exact failure the tool exists to avoid. It must keep its
    # own action so replay routes it through _scroll_containers.
    history = [_item({"scroll_panels": {"pages": 0.8}},
                     result=[{"extracted_content": "scroll_panels: scrolled 2"}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [{"action": "scroll_panels", "pages": 0.8}]
