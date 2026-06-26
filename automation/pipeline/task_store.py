"""Task identity + recording registry.

Maps a task PROMPT to a stable id (a hash of the normalized prompt) and to the paths of its
recorded trace + compiled script. This is how the runner knows whether a task already has a
reusable recording: same prompt -> same id -> same filename, so it either exists or it doesn't.
A `manifest.json` keeps the id -> prompt mapping human-readable for managing many tasks.

Identity is based on the ORIGINAL user prompt (not the expander's non-deterministic rewrite).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

RECORDINGS_DIR = Path("recordings")
MANIFEST_PATH = RECORDINGS_DIR / "manifest.json"


def task_id(prompt: str) -> str:
    """Stable short id for a prompt (whitespace/case-insensitive)."""
    norm = " ".join(prompt.split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def recording_path(tid: str) -> Path:
    return RECORDINGS_DIR / f"{tid}.json"


def steps_path(tid: str) -> Path:
    return RECORDINGS_DIR / f"{tid}.steps.json"


def has_script(tid: str) -> bool:
    return steps_path(tid).exists()


def update_manifest(tid: str, prompt: str, **fields: Any) -> None:
    """Record/refresh this task's entry in recordings/manifest.json."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    entry = data.get(tid, {})
    entry.setdefault("prompt", prompt.strip())
    entry.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    entry.update(fields)
    entry["updated"] = datetime.now().isoformat(timespec="seconds")
    data[tid] = entry
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))
