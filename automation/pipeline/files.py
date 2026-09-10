"""Local files referenced by a task prompt.

The upload UX is prompt-first with ONE storage convention: task files live in
automation/uploads/ and the task text names them by BASENAME ("upload the file
New_Employees_List_-_WI_LTD.csv ..."). Before login, every filename the prompt mentions
must resolve to a non-empty file in that folder — the resolved ABSOLUTE paths become the
agent's `available_file_paths` (browser-use's upload_file allowlist) and the paths a
replayed upload step re-checks.

A native OS file dialog can never be driven by DOM automation, so the file is attached
programmatically. The absolute, existing path is not a nicety but a hard requirement:
observed live (2026-07-22), a relative nonexistent path "uploaded" fine via CDP, then
clicking Save made the page READ the ungranted file and Chrome killed the renderer
(RESULT_CODE_KILLED_BAD_MESSAGE — the "Aw, Snap!" tab).

Stdlib-only on purpose: script_compile's replay executor imports UPLOADS_DIR/find_file,
and this module must never pull the pipeline back in.
"""
from __future__ import annotations

import re
from pathlib import Path

# The one place task files live (user-established convention, 2026-07-22).
UPLOADS_DIR = Path("automation/uploads")

# Extensions treated as "a local file the task wants to use". Deliberately explicit: a
# bare host like fakenamegenerator.com must never look like a file reference.
_EXTS = ("csv", "tsv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "txt", "json",
         "xml", "docx", "doc", "zip")
_EXT_ALT = "|".join(_EXTS)

# 'Name With Spaces.csv' / "Name With Spaces.csv" — quoted names may contain spaces.
_QUOTED = re.compile(r'["\']([^"\']+\.(?:%s))["\']' % _EXT_ALT, re.IGNORECASE)
# Bare single-token names. The lookbehind rejects URL/path contexts (https://host/a.csv)
# and mid-name starts — including after '-' and '.', so gen-random.csv.php in a URL can
# never leak a phantom "random.csv" reference.
_BARE = re.compile(r'(?<![\w/\\.:\-])([\w][\w.\-]*\.(?:%s))\b' % _EXT_ALT,
                   re.IGNORECASE)


def referenced_files(prompt: str) -> list[str]:
    """File names the prompt mentions, in first-seen order, case-deduped."""
    seen: set[str] = set()
    out: list[str] = []

    def _add(name: str) -> None:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)

    text = prompt or ""
    for m in _QUOTED.finditer(text):
        _add(m.group(1))
    # Quoted spans are consumed before the bare scan, so 'Staff List 2026.xlsx' can
    # never also leak a phantom bare "2026.xlsx".
    for m in _BARE.finditer(_QUOTED.sub(" ", text)):
        _add(m.group(1))
    return out


def find_file(name: str) -> Path | None:
    """UPLOADS_DIR/<basename> when it exists NON-EMPTY, else None. A 0-byte hit counts
    as missing — the repo lives under the iCloud-synced Desktop, and a cloud placeholder
    must never be treated as an uploadable file."""
    cand = UPLOADS_DIR / Path(name).name
    try:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    except OSError:
        pass
    return None


def resolve_prompt_files(prompt: str) -> tuple[list[str], list[str]]:
    """(resolved absolute paths, problem lines) for every file the prompt references.

    The caller fails the run BEFORE login on any problem — a missing or empty file must
    die at startup, not minutes into a run (or worse, crash the tab at save time).
    """
    resolved: list[str] = []
    problems: list[str] = []
    for name in referenced_files(prompt):
        if Path(name).name != name:
            # A quoted path slipped in: the convention is ONE folder, bare names only —
            # that is what makes committed skills portable across machines and runs.
            problems.append(f"{name}: use a bare file name — task files live in "
                            f"{UPLOADS_DIR}/")
            continue
        hit = find_file(name)
        if hit is None:
            problems.append(f"{name}: not found (or empty) in {UPLOADS_DIR}/ — place "
                            f"the file there")
        else:
            path = str(hit.resolve())
            if path not in resolved:
                resolved.append(path)
    return resolved, problems
