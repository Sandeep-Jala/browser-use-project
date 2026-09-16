"""Per-run video recording (automation/browser/recording.py).

The point of these tests is the FAIL-SOFT contract. Recording rides on a browser-use
PrivateAttr (`BrowserSession._recording_watchdog`) that has no public accessor, and its
encoder is an optional dependency — so every way this can go wrong (watchdog renamed away,
`start_recording` raising because imageio is missing, a stop with nothing running) must
return None and let the run continue. A missing video is never worth failing a task over.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from automation.browser.recording import start_run_recording, stop_run_recording
from automation.config import _env_video_size
from automation.pipeline.report import _render_video


class _FakeWatchdog:
    """browser-use's RecordingWatchdog surface: start_recording / stop_recording."""

    def __init__(self, *, start_raises=None, stop_raises=None, stopped_path=None):
        self.calls: list[tuple] = []
        self._start_raises = start_raises
        self._stop_raises = stop_raises
        self._stopped_path = stopped_path

    async def start_recording(self, output_path, size=None, framerate=None):
        self.calls.append(("start", Path(output_path), size))
        if self._start_raises:
            raise self._start_raises
        return Path(output_path)

    async def stop_recording(self):
        self.calls.append(("stop",))
        if self._stop_raises:
            raise self._stop_raises
        return self._stopped_path


def _session(watchdog):
    return SimpleNamespace(_recording_watchdog=watchdog)


# --------------------------------- start ---------------------------------

async def test_start_records_to_the_runs_own_file(tmp_path):
    wd = _FakeWatchdog()
    out = tmp_path / "20260728_120000" / "run.mp4"

    path = await start_run_recording(_session(wd), out)

    assert path == out
    assert wd.calls == [("start", out, None)]
    assert out.parent.is_dir()   # created for the encoder, which will not mkdir for us


async def test_size_is_passed_through_as_a_viewport_dict(tmp_path):
    """The size becomes the screencast's maxWidth/maxHeight — the cost lever."""
    wd = _FakeWatchdog()

    await start_run_recording(_session(wd), tmp_path / "run.mp4", size=(1280, 800))

    assert wd.calls[0][2] == {"width": 1280, "height": 800}


async def test_missing_watchdog_degrades_instead_of_raising(tmp_path):
    """If browser-use ever renames the private attr, runs must keep working."""
    session = SimpleNamespace()   # no _recording_watchdog at all

    assert await start_run_recording(session, tmp_path / "run.mp4") is None
    assert await stop_run_recording(session) is None


async def test_start_failure_is_swallowed(tmp_path):
    """What a missing imageio actually looks like: start_recording raises RuntimeError."""
    wd = _FakeWatchdog(start_raises=RuntimeError(
        'Failed to initialize video recorder — ensure optional deps are installed'))

    assert await start_run_recording(_session(wd), tmp_path / "run.mp4") is None


# --------------------------------- stop ---------------------------------

async def test_stop_returns_the_saved_path(tmp_path):
    saved = tmp_path / "run.mp4"
    wd = _FakeWatchdog(stopped_path=saved)

    assert await stop_run_recording(_session(wd)) == saved


async def test_stop_with_nothing_recording_is_a_quiet_none():
    wd = _FakeWatchdog(stopped_path=None)   # browser-use returns None when idle

    assert await stop_run_recording(_session(wd)) is None


async def test_stop_failure_is_swallowed():
    wd = _FakeWatchdog(stop_raises=RuntimeError("CDP gone"))

    assert await stop_run_recording(_session(wd)) is None


# ------------------------------ size parsing ------------------------------

def test_video_size_unset_means_detect_the_viewport(monkeypatch):
    monkeypatch.delenv("RECORD_VIDEO_SIZE", raising=False)
    assert _env_video_size("RECORD_VIDEO_SIZE") is None


@pytest.mark.parametrize("raw, parsed", [
    ("1280x800", (1280, 800)),
    ("1280 X 800", (1280, 800)),
    ("640x480", (640, 480)),
])
def test_video_size_parses(monkeypatch, raw, parsed):
    monkeypatch.setenv("RECORD_VIDEO_SIZE", raw)
    assert _env_video_size("RECORD_VIDEO_SIZE") == parsed


def test_bad_video_size_dies_at_startup_with_the_fix(monkeypatch):
    """Fail loudly HERE, not 40 steps into a run (the VISION_DETAIL_LEVEL lesson)."""
    monkeypatch.setenv("RECORD_VIDEO_SIZE", "1280*800")
    with pytest.raises(SystemExit, match="WIDTHxHEIGHT"):
        _env_video_size("RECORD_VIDEO_SIZE")


# --------------------------- run lifecycle wiring ---------------------------

def _hybrid_session(video_path, watchdog):
    """A HybridSession far enough along to call finalize() — no browser needed."""
    from automation.pipeline.hybrid import HybridSession

    hs = HybridSession(SimpleNamespace(config=SimpleNamespace()))
    hs.collectors = []
    hs.session = _session(watchdog)
    hs.pw_browser = SimpleNamespace(close=_noop)
    hs.video_path = video_path
    return hs


async def _noop():
    return None


async def test_finalize_stops_the_recording_and_publishes_it_as_an_artifact(tmp_path):
    saved = tmp_path / "run.mp4"
    wd = _FakeWatchdog(stopped_path=saved)

    result = await _hybrid_session(tmp_path / "run.mp4", wd).finalize("t", None)

    assert ("stop",) in wd.calls
    assert result.artifacts["video"] == saved   # reaches report.py through RunResult


async def test_finalize_leaves_the_recorder_alone_on_an_unrecorded_run():
    """--record off: nothing to stop, and no 'video' key inviting a broken player."""
    wd = _FakeWatchdog()

    result = await _hybrid_session(None, wd).finalize("t", None)

    assert wd.calls == []
    assert "video" not in result.artifacts


# ------------------------------ report player ------------------------------

def test_report_embeds_a_relative_video_source(tmp_path):
    """Relative src: report.html sits beside run.mp4, so the folder stays portable."""
    video = tmp_path / "run.mp4"
    video.write_bytes(b"\x00")
    result = SimpleNamespace(artifacts={"video": video})

    html = _render_video(result)

    assert "src='run.mp4'" in html
    assert str(tmp_path) not in html          # no absolute path leaked into the page
    assert "time-lapse" in html               # the caveat travels with the artifact


def test_report_omits_the_player_when_the_run_was_not_recorded():
    assert _render_video(SimpleNamespace(artifacts={})) == ""


def test_report_omits_the_player_when_the_file_vanished(tmp_path):
    result = SimpleNamespace(artifacts={"video": tmp_path / "gone.mp4"})
    assert _render_video(result) == ""
