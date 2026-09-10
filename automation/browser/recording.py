"""Per-run screen recording, on top of browser-use's own CDP screencast recorder.

browser-use already ships the whole recorder: `RecordingWatchdog` captures frames via CDP
`Page.startScreencast` and `VideoRecorderService` encodes them to mp4 with imageio/ffmpeg.
The watchdog is attached to EVERY session it builds. Two things were missing here — the
`imageio` extra (declared in pyproject now; without it the recorder logs one line and
silently produces no file) and someone to turn it on.

WHY NOT `BrowserProfile(record_video_dir=...)`, the obvious knob: setting it makes the
watchdog auto-start on BrowserConnectedEvent, which happens in main() before a run
directory exists — the file lands under a uuid7 name in a fixed folder and spans the whole
process, login included. Leaving the field unset keeps that auto-start disabled so we can
start the recording ourselves once the run has a home, and name it after the run.

The coupling to guard: `_recording_watchdog` is a PrivateAttr with no public accessor, so
this module reaches for it defensively and returns None on anything unexpected. A missing
video must never be the reason a run dies — every caller treats the result as optional.

WHAT THE VIDEO LOOKS LIKE: a time-lapse of the page viewport. CDP emits a frame only when
the page visually CHANGES, and the encoder appends frames at a fixed rate with no
timestamp padding, so the seconds spent waiting on an LLM call collapse to nothing — a
long run becomes a short, jumpy clip. Browser chrome, the tab strip, and native dialogs
are outside the capture entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("framework.recording")


def _watchdog(session: Any) -> Any | None:
    """browser-use's RecordingWatchdog for `session`, or None if it isn't reachable."""
    watchdog = getattr(session, "_recording_watchdog", None)
    if watchdog is None:
        logger.warning("no recording watchdog on this browser session — skipping video "
                       "(browser-use may have renamed _recording_watchdog)")
    return watchdog


async def start_run_recording(session: Any, output_path: Path,
                              size: tuple[int, int] | None = None) -> Path | None:
    """Begin recording this run to `output_path`; return it, or None if recording is off.

    `size` (width, height) becomes the screencast's maxWidth/maxHeight, so Chrome
    downscales each frame BEFORE sending it — the cheap way to cut recording overhead.
    None detects the live viewport instead.
    """
    watchdog = _watchdog(session)
    if watchdog is None:
        return None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # browser-use wants a ViewportSize (a TypedDict) — a plain dict satisfies it.
        viewport = {"width": size[0], "height": size[1]} if size else None
        path = await watchdog.start_recording(output_path, size=viewport)
    except Exception as exc:  # noqa: BLE001 - a video is never worth failing a run over
        logger.warning("could not start video recording: %s", exc)
        return None
    logger.info("📹 recording this run to %s", path)
    return Path(path) if path else output_path


async def stop_run_recording(session: Any) -> Path | None:
    """Finish the recording and flush the file. Returns its path, or None if none ran.

    Safe to call unconditionally: with no recording in progress browser-use returns None.
    Call it before the browser goes away — the encoder needs the CDP connection to stop
    the screencast cleanly.
    """
    watchdog = _watchdog(session)
    if watchdog is None:
        return None
    try:
        path = await watchdog.stop_recording()
    except Exception as exc:  # noqa: BLE001 - a half-written video must not fail the run
        logger.warning("could not stop video recording cleanly: %s", exc)
        return None
    if path:
        logger.info("📹 saved run video: %s", path)
    return Path(path) if path else None
