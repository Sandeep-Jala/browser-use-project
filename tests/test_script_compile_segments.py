"""compile_recording(emit_start_goto=...) — mid-flow segments must not reload the SPA."""
import json

from automation.pipeline.script_compile import compile_recording, save_steps

RECORDING = {"history": [
    {"state": {"url": "http://app/start", "interacted_element": []},
     "model_output": {"action": [{"navigate": {"url": "http://app/section"}}]},
     "result": []},
    {"state": {"url": "http://app/section",
               "interacted_element": [{"node_name": "button", "ax_name": "Save",
                                       "attributes": {"id": "btnSave"}, "x_path": "//button[1]"}]},
     "model_output": {"action": [{"click": {"index": 3}}]},
     "result": []},
]}


def _write(tmp_path):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(RECORDING))
    return p


def test_default_emits_leading_goto(tmp_path):
    steps = compile_recording(_write(tmp_path))
    assert steps[0] == {"action": "goto", "url": "http://app/start"}


def test_segment_compile_drops_only_the_leading_goto(tmp_path):
    rec = _write(tmp_path)
    default = compile_recording(rec)
    segment = compile_recording(rec, emit_start_goto=False)
    # Identical except the synthetic leading goto; explicit navigate actions survive.
    assert segment == default[1:]
    assert segment[0] == {"action": "goto", "url": "http://app/section"}
    assert any(s.get("action") == "click" for s in segment)


def test_save_steps_passes_through(tmp_path):
    rec = _write(tmp_path)
    out = tmp_path / "steps.json"
    steps = save_steps(rec, out, emit_start_goto=False)
    assert json.loads(out.read_text()) == steps
    assert steps[0]["url"] == "http://app/section"
