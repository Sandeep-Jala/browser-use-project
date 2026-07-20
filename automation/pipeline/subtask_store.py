"""Task/subtask identity + the shared subtask library.

The library holds one entry per (parameterized subtask prompt, starting page context) pair,
shared GLOBALLY across parent tasks — the "select {{business}} business" prefix every task
starts with is ONE entry here, authored once and replayed everywhere. It is the only
recording store: there is no whole-task script tier above it.

Identity: `subtask_id(template_prompt, context)` hashes the TOKENIZED prompt (values lifted
into {{param}} tokens), so "add invoice for customer Suresh Gopi" and "... for customer Mr
Jones" resolve to the same entry; the `context` half is the normalized URL the page is on
when the subtask starts, disambiguating same-worded subtasks that begin on different pages.

`task_id(prompt)` is the coarser twin: a stable hash of a PARENT task prompt, used to key its
cached decomposition and to label runs.

Layout (LIBRARY_DIR):
  {sid}.steps.json      compiled segment steps (concrete values from the authoring run)
  {sid}.template.json   adapt.parameterize output ({{param}} tokens + defaults)
  {sid}.recording.json  raw agent history of the authoring segment
  {sid}.meta.json       mutable per-run stats (uses/fail_count) — kept OUT of the manifest
                        so ~100k replays/day don't serialize on one atomically-rewritten file
  manifest.json         identity registry (template_prompt, params, context, end_context...),
                        written only when an entry is created or archived
  archive/              retired entries, timestamped

DECOMPOSITIONS_DIR holds one cached decomposition per PARENT prompt hash (task_id).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

LIBRARY_DIR = Path("library")
LIBRARY_MANIFEST = LIBRARY_DIR / "manifest.json"
DECOMPOSITIONS_DIR = Path("decompositions")

# Volatile URL path segments that must not split library identity: pure digit runs, GUIDs,
# and long hex ids are all instance data (record ids, session ids), not page structure.
_VOLATILE_SEGMENT = re.compile(
    r"^(\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,})$",
    re.IGNORECASE,
)

# {{param}} tokens in a template prompt — the ONE definition of the decomposer's token
# grammar (decompose.py reuses it; group 1 is the token name). Token NAMES are erased
# from library identity: the LLM decomposer names tokens freely ({{business}} one run,
# {{business_name}} the next), and a name difference must not split two identically-worded
# subtasks into separate library entries.
TOKEN_RE = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


def normalize_context(url: str) -> str:
    """Stable page-state key from a live URL.

    Lowercased path plus '#fragment' (SPA routes live in the fragment here); origin and query
    stripped; volatile segments (record ids, GUIDs, hex runs) replaced by '*' so two visits
    to the same page through different records share a context.
      https://app/x/bookkeeping/12345/inputs/sales?tab=2#invoices
        -> /x/bookkeeping/*/inputs/sales#invoices
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url or "")
    path = parts.path or "/"
    segments = [
        "*" if _VOLATILE_SEGMENT.match(seg) else seg.lower()
        for seg in path.split("/")
    ]
    norm = "/".join(segments) or "/"
    frag = (parts.fragment or "").strip()
    if frag:
        # The fragment is itself a path on SPA routers — normalize its segments too.
        frag_norm = "/".join(
            "*" if _VOLATILE_SEGMENT.match(seg) else seg.lower()
            for seg in frag.split("/")
        )
        norm += f"#{frag_norm}"
    return norm


def task_id(prompt: str) -> str:
    """Stable short id for a PARENT task prompt (whitespace/case-insensitive).

    Keys the cached decomposition (decompositions/<tid>.json) and labels runs. Identity is
    the ORIGINAL user prompt, so editing a task's wording by even one word gives it a new id
    and orphans its cached decomposition — see tests/test_tasks.py::test_task_ids_stable.
    """
    norm = " ".join(prompt.split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def subtask_id(template_prompt: str, context: str) -> str:
    """Stable short id for a (tokenized subtask prompt, starting context) pair.

    Normalized hard so decompositions of DIFFERENT parent tasks converge on one entry:
    whitespace collapsed, lowercased, token names erased ({{business}} == {{business_name}}
    — wording carries the identity, not what the LLM called the slot), and trailing
    punctuation dropped (where the decomposer cuts a span decides whether it ends in '.').
    """
    norm = " ".join(template_prompt.split()).lower()
    norm = TOKEN_RE.sub("{{*}}", norm).rstrip(" .,;")
    return hashlib.sha256(f"{norm}\n{context}".encode("utf-8")).hexdigest()[:16]


def steps_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.steps.json"


def template_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.template.json"


def recording_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.recording.json"


def meta_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.meta.json"


def code_path(sid: str) -> Path:
    """Tier-1 body: the generated skill function (skills/codegen.py)."""
    return LIBRARY_DIR / f"{sid}.skill.py"


def anchors_path(sid: str) -> Path:
    """Tier-1 element identities: handle -> ranked selectors + fingerprint."""
    return LIBRARY_DIR / f"{sid}.anchors.json"


def aliases_path() -> Path:
    """Semantic-router alias table: {alias_sid: {"sid": canonical, "slots": {...}}}."""
    return LIBRARY_DIR / "aliases.json"


def embeddings_path() -> Path:
    """Semantic-router vector cache: {"model": name, "vectors": {sid: [floats]}}."""
    return LIBRARY_DIR / "embeddings.json"


def load_aliases() -> dict[str, Any]:
    p = aliases_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - a corrupt alias table just means no aliases
        return {}


def save_alias(alias_sid: str, data: dict[str, Any]) -> None:
    aliases = load_aliases()
    aliases[alias_sid] = data
    _atomic_write_json(aliases_path(), aliases)


def has_script(sid: str) -> bool:
    """True when the entry has an executable body. Tier-1 code is the normal case; a steps
    file exists only for entries the transpiler couldn't express (see codegen)."""
    return code_path(sid).exists() or steps_path(sid).exists()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def archive_entry(sid: str) -> list[Path]:
    """Move a library entry's files into library/archive/ (timestamped). The manifest entry
    is dropped; meta is removed (a re-authored entry starts with fresh stats). Returns the
    archived paths (empty if nothing existed)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = LIBRARY_DIR / "archive"
    moved: list[Path] = []
    for src, kind in ((steps_path(sid), "steps"), (template_path(sid), "template"),
                      (recording_path(sid), "recording"),
                      (recording_path(sid).with_suffix(".failed.json"), "recording-failed"),
                      (code_path(sid), "skill"), (anchors_path(sid), "anchors")):
        if not src.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        dst = archive_dir / f"{sid}.{kind}.{stamp}{src.suffix}"
        os.replace(src, dst)
        moved.append(dst)
    meta_path(sid).unlink(missing_ok=True)
    manifest = load_manifest()
    if sid in manifest:
        del manifest[sid]
        _atomic_write_json(LIBRARY_MANIFEST, manifest)
    return moved


def load_manifest() -> dict[str, Any]:
    """Read library/manifest.json; {} if missing or corrupt."""
    if not LIBRARY_MANIFEST.exists():
        return {}
    try:
        return json.loads(LIBRARY_MANIFEST.read_text())
    except Exception:  # noqa: BLE001 - a corrupt manifest just means an empty library
        return {}


def update_manifest(sid: str, template_prompt: str, **fields: Any) -> None:
    """Record/refresh a library entry in manifest.json (atomic write). Called only on
    author/archive — per-run counters go in the entry's meta file, not here."""
    data = load_manifest()
    entry = data.get(sid, {})
    entry.setdefault("template_prompt", " ".join(template_prompt.split()))
    entry.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    entry.update(fields)
    entry["updated"] = datetime.now().isoformat(timespec="seconds")
    data[sid] = entry
    _atomic_write_json(LIBRARY_MANIFEST, data)


def load_meta(sid: str) -> dict[str, Any]:
    p = meta_path(sid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - corrupt meta just means fresh stats
        return {}


def bump_meta(sid: str, **deltas: int) -> dict[str, Any]:
    """Increment per-run counters (uses, fail_count, ...) in the entry's meta file.
    A successful use resets fail_count: only CONSECUTIVE failures should retire an entry."""
    meta = load_meta(sid)
    for key, delta in deltas.items():
        meta[key] = int(meta.get(key, 0)) + int(delta)
    if deltas.get("uses"):
        meta["fail_count"] = 0
        meta["last_used"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(meta_path(sid), meta)
    return meta


def archive_if_failing(sid: str, threshold: int = 2) -> bool:
    """Retire the entry once its consecutive fail_count reaches `threshold`, so the next run
    authors a clean replacement instead of re-fighting a stale recording. True if archived."""
    if int(load_meta(sid).get("fail_count", 0)) >= threshold:
        archive_entry(sid)
        return True
    return False


# ------------------------------- decomposition cache -------------------------------


def decomposition_path(tid: str) -> Path:
    return DECOMPOSITIONS_DIR / f"{tid}.json"


def load_decomposition(tid: str) -> dict[str, Any] | None:
    p = decomposition_path(tid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - corrupt cache entry = no cache entry
        return None
    return data if isinstance(data, dict) and data.get("subtasks") else None


def save_decomposition(tid: str, data: dict[str, Any]) -> None:
    _atomic_write_json(decomposition_path(tid), data)


def all_decompositions() -> dict[str, dict[str, Any]]:
    """Every cached decomposition keyed by tid (for derived matching)."""
    if not DECOMPOSITIONS_DIR.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for p in DECOMPOSITIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and data.get("subtasks"):
            out[p.stem] = data
    return out
