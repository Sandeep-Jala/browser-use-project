"""The files a prompt names, under `automation/uploads/`.

A prompt references an upload by BARE BASENAME, and `__main__.py` resolves every reference BEFORE
login and refuses the whole run if one is missing. So this module's job is not merely to store a
file — it is to guarantee the prompt side will be able to see it.

The validation that earns its place is `referenced_files(name) == [name]`. The prompt-side scanner
carries a lookbehind that hides a bare name in URL-ish contexts, and a name containing a space only
resolves when QUOTED. A file can therefore sit in the directory, perfectly named, and be
unreachable from any prompt — which is far better to learn at upload time, with the exact snippet
to paste, than from a refused run.

Everything else here is inherited rather than invented: the allowed extensions and the
"0 bytes counts as missing" rule both come from `files.py`, the second because this repo lives on
an iCloud-synced Desktop where an undownloaded placeholder is a real, empty file.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO

from automation.pipeline import files as pfiles
from automation.pipeline.files import ALLOWED_EXTENSIONS, referenced_files

log = logging.getLogger("framework.ui")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_CHUNK = 1024 * 1024


class FileInUse(RuntimeError):
    """A prompt still names this file. Deleting it would fail that prompt's next run before login."""


@dataclass(frozen=True)
class UploadRow:
    name: str
    path: Path
    size: int
    modified: datetime
    empty: bool
    visible_to_parser: bool
    snippet: str
    referenced_by: tuple[str, ...] = ()
    warning: str | None = None


def _dir() -> Path:
    """Resolved through `files.UPLOADS_DIR` every call, not captured at import — one
    configuration point shared with `find_file` and `resolve_prompt_files`, which read the same
    global. Two copies would be two things to keep in step."""
    return pfiles.UPLOADS_DIR


def snippet_for(name: str) -> str:
    """How a prompt must spell this file. Quoted when the name contains a space — the bare-name
    scanner cannot match one, so an unquoted reference would simply never resolve."""
    return f"'{name}'" if " " in name else name


def _parser_sees(name: str) -> bool:
    """Would `referenced_files` find this name in a prompt that spells it correctly?"""
    probe = f"open {snippet_for(name)} and save"
    return [n.lower() for n in referenced_files(probe)] == [name.lower()]


def save_upload(name: str, stream: IO[bytes], *, overwrite: bool = False) -> UploadRow:
    """Store one uploaded file. Raises ValueError for anything a run could not use."""
    if Path(name).name != name or not name.strip():
        raise ValueError(f"{name}: use a bare file name — task files live in {_dir()}/")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{name}: unsupported extension {('.' + ext) if ext else '(none)'} — the prompt "
            f"scanner only recognises: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    target = d / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"{name} already exists — upload again with overwrite to replace it")

    # Stream to a dot-prefixed part file, then replace. A partially written file with a non-zero
    # size is precisely what `find_file` would accept, and a run would hand it to the browser.
    part = d / f".{name}.part"
    written = 0
    try:
        with part.open("wb") as out:
            while chunk := stream.read(_CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"{name}: too large — the limit is "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                out.write(chunk)
        if written == 0:
            # find_file tests st_size > 0, so an empty file would be reported as MISSING at run
            # time — a confusing way to discover an upload went wrong.
            raise ValueError(f"{name}: the file is empty, and a run treats an empty file as "
                             f"missing")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)

    visible = _parser_sees(name)
    return UploadRow(
        name=name, path=target, size=written,
        modified=datetime.fromtimestamp(target.stat().st_mtime),
        empty=False, visible_to_parser=visible, snippet=snippet_for(name),
        referenced_by=_referenced_by().get(name.lower(), ()),
        warning=None if visible else (
            f"a prompt cannot refer to this name: the file scanner will not see "
            f"{name!r} in prompt text, so a run would never resolve it. Rename it to something "
            f"starting with a letter or digit."),
    )


def list_uploads() -> list[UploadRow]:
    """Every usable upload, newest name order, with the prompts that reference it."""
    d = _dir()
    if not d.is_dir():
        return []
    refs = _referenced_by()
    rows: list[UploadRow] = []
    for f in sorted(d.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        size = f.stat().st_size
        visible = _parser_sees(f.name)
        warning = None
        if size == 0:
            warning = ("0 bytes — a placeholder that was never downloaded from iCloud. A run "
                       "treats this as a missing file and refuses before it logs in.")
        elif not visible:
            warning = ("a prompt cannot refer to this name — the file scanner will not see it in "
                       "prompt text.")
        rows.append(UploadRow(
            name=f.name, path=f, size=size,
            modified=datetime.fromtimestamp(f.stat().st_mtime),
            empty=size == 0, visible_to_parser=visible, snippet=snippet_for(f.name),
            referenced_by=refs.get(f.name.lower(), ()), warning=warning))
    return rows


def delete_upload(name: str, *, force: bool = False) -> Path:
    """Move a file to `automation/uploads/.trash/`.

    Refuses while a prompt still names it: that reference is checked before a run logs in, so
    deleting it breaks that prompt in a way nothing else would warn about.
    """
    if Path(name).name != name or not name.strip():
        raise ValueError(f"{name}: use a bare file name")
    src = _dir() / name
    if not src.is_file():
        raise FileNotFoundError(f"no such upload: {name}")
    users = _referenced_by().get(name.lower(), ())
    if users and not force:
        raise FileInUse(
            f"{name} is referenced by {', '.join(users)} — those prompts would fail before "
            f"logging in. Delete again with force if you mean it.")
    trash = _dir() / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(src), str(dest))
    return dest


def _referenced_by() -> dict[str, tuple[str, ...]]:
    """`{lowercased filename: (task keys that name it, …)}` across BOTH registries.

    Uses the prompt scanner itself rather than a substring search, so the answer matches what a
    run will actually resolve — including the quoting rule for names with spaces.
    """
    from automation.tasks import load_tasks
    out: dict[str, list[str]] = {}
    try:
        registry = load_tasks()
    except Exception:  # noqa: BLE001 - a broken registry must not break the files page
        log.exception("could not read the task registry for the file cross-reference")
        return {}
    for key, spec in registry.items():
        for name in referenced_files(spec.prompt):
            out.setdefault(name.lower(), []).append(key)
    return {k: tuple(sorted(set(v))) for k, v in out.items()}
