"""`os.replace` over a file someone else is reading.

POSIX `rename(2)` is atomic and always succeeds: the reader keeps its handle on the old inode and
sees that file whole. Two places here are built on exactly that and say so in their docstrings —
`control.write_control` (the operator's pause/stop channel, re-read every 0.4 s by the running
task) and `progress.json` (written by `hybrid`, polled by the UI's supervisor).

Windows has no such guarantee: `MoveFileEx` fails with a sharing violation whenever a handle to
the destination is open, surfacing as `PermissionError`. So those two writes fail INTERMITTENTLY
there, decided by whether the poll happened to land inside the write — and a Stop button that
works four times in five is worse than one that never works, because nobody believes the report.

The retry cannot be exercised by a real contended replace on POSIX (there is nothing to lose the
race to), so the failure is injected. That is the honest shape for a platform-specific fix: the
behaviour under a `PermissionError` is the thing being specified.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from automation.pipeline.atomic import replace_with_retry


def test_the_ordinary_case_is_one_call_and_no_sleeping(tmp_path):
    src = tmp_path / "new.json"
    dst = tmp_path / "live.json"
    src.write_text("fresh", encoding="utf-8")
    dst.write_text("stale", encoding="utf-8")

    slept: list[float] = []
    replace_with_retry(src, dst, sleep=slept.append)

    assert dst.read_text(encoding="utf-8") == "fresh"
    assert not src.exists()
    assert slept == []


def test_a_reader_holding_the_file_is_retried_not_reported(monkeypatch, tmp_path):
    """The whole point: the reader's handle is open for microseconds, so the second attempt
    almost always lands."""
    src = tmp_path / "new.json"
    dst = tmp_path / "live.json"
    src.write_text("fresh", encoding="utf-8")

    real_replace = os.replace
    attempts = {"n": 0}

    def flaky(a, b):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    slept: list[float] = []
    replace_with_retry(src, dst, sleep=slept.append)

    assert attempts["n"] == 2
    assert dst.read_text(encoding="utf-8") == "fresh"
    assert len(slept) == 1 and slept[0] > 0


def test_it_gives_up_loudly_rather_than_losing_a_stop_command(monkeypatch, tmp_path):
    """Silence here would mean an operator's `stop` vanishing, which is the one outcome worse
    than an exception."""
    src = tmp_path / "new.json"
    src.write_text("fresh", encoding="utf-8")

    def always_busy(a, b):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", always_busy)
    slept: list[float] = []

    with pytest.raises(PermissionError):
        replace_with_retry(src, tmp_path / "live.json", sleep=slept.append)

    assert len(slept) == 2          # slept between attempts, not after the last one


def test_other_os_errors_are_not_retried(monkeypatch, tmp_path):
    """A missing source or a cross-device move is a real fault that retrying cannot mend, and
    swallowing it for 150 ms would only delay the report."""
    src = tmp_path / "new.json"
    src.write_text("fresh", encoding="utf-8")
    calls = {"n": 0}

    def cross_device(a, b):
        calls["n"] += 1
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "replace", cross_device)

    with pytest.raises(OSError, match="cross-device"):
        replace_with_retry(src, tmp_path / "live.json", sleep=lambda s: None)
    assert calls["n"] == 1


def test_the_control_channel_uses_it(tmp_path):
    """Wiring check. `write_control` is the contended writer whose docstring promises the reader
    sees a whole file, so it has to be the retrying replace and not a bare one."""
    from automation.pipeline import control

    calls: list[tuple[Path, Path]] = []
    real = control.replace_with_retry

    def spy(src: Path, dst: Path, **kwargs):
        calls.append((src, dst))
        return real(src, dst, **kwargs)

    control.replace_with_retry = spy          # type: ignore[assignment]
    try:
        control.write_control(tmp_path / "control.json", "stop")
    finally:
        control.replace_with_retry = real     # type: ignore[assignment]

    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "control.json"
    assert control.read_control(tmp_path / "control.json")["command"] == "stop"
