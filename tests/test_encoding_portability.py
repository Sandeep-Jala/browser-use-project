"""Text I/O must name its encoding. On Windows, the default is not UTF-8.

`Path.read_text()` and `Path.write_text()` with no `encoding=` use the locale encoding — cp1252
on a default Windows box, UTF-8 on macOS and modern Linux. That difference is invisible here and
fatal there, and this repo carries the data to prove it: the committed recordings under `library/`
contain bytes cp1252 cannot decode, so the reads that load them raise `UnicodeDecodeError` on
Windows and nowhere else.

The two failure shapes are not equally visible, and the quiet one is worse:

* `pipeline/script_compile.py`'s `compile_recording` has no try/except, so it RAISES and the
  replay engine is simply dead.
* `pipeline/runner.py`'s metadata-restore pass and `pipeline/hybrid.py`'s three diagnosis helpers
  are wrapped in `except Exception` and documented "fail OPEN". They would fail open on every
  single run, and per `runner.py`'s own docstring that means each `find_by_text` click "compiles
  to NOTHING and the committed script silently loses its clicks". A wrong script that looks fine.

So the guard is a source-level sweep rather than a fixture: a fixture pins the sites that exist
today, and the bug is reintroduced by the next `read_text()` someone adds. The 15 call sites that
already passed `encoding="utf-8"` before this sweep are the evidence that the convention is easy
to forget — they sat next to 26 that did not.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "automation"
LIBRARY_DIR = REPO_ROOT / "library"

_TEXT_IO = ("read_text", "write_text")


def _label(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise — the helper is also pointed at a
    tmp_path by the self-check below, which is not under the repo."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _unencoded_text_io(package_root: Path) -> list[str]:
    """Every `.read_text(...)`/`.write_text(...)` call in the package with no `encoding=`.

    AST, not grep: these calls routinely span lines, and a line-based search reports a false
    positive for every one whose `encoding=` sits on a continuation line (it did, for
    `ui/supervisor.py`'s adoption-record write, which was correct all along).
    """
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _TEXT_IO:
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                offenders.append(f"{_label(path)}:{node.lineno} ({func.attr})")
    return offenders


def test_no_text_io_relies_on_the_locale_encoding():
    offenders = _unencoded_text_io(PACKAGE_ROOT)
    assert offenders == [], (
        "these calls would use cp1252 on Windows — pass encoding=\"utf-8\":\n  "
        + "\n  ".join(offenders))


def test_the_ast_sweep_would_actually_catch_a_regression(tmp_path):
    """Guard the guard. A sweep that silently matches nothing is worse than no sweep, and this
    one depends on an AST shape (`Attribute` whose attr is read_text/write_text) that a
    refactor could quietly stop matching."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "good.py").write_text(
        "from pathlib import Path\n"
        "def f(p: Path):\n"
        "    return p.read_text(encoding='utf-8')\n",
        encoding="utf-8")
    (pkg / "bad.py").write_text(
        "from pathlib import Path\n"
        "def f(p: Path):\n"
        "    p.write_text('x')\n"
        "    return p.read_text()\n",
        encoding="utf-8")

    found = _unencoded_text_io(pkg)
    assert len(found) == 2, found
    assert all("bad.py" in hit for hit in found)
    assert any("write_text" in hit for hit in found)
    assert any("read_text" in hit for hit in found)


def test_committed_recordings_are_utf8_and_not_cp1252_decodable():
    """The measurement behind all of the above, run against the real data rather than asserted.

    If this ever reports zero undecodable files, the cp1252 hazard has stopped being hypothetical
    only for the files currently in the tree — the sweep above is still what keeps it fixed.
    """
    recordings = sorted(LIBRARY_DIR.glob("*.recording*.json"))
    if not recordings:                     # a fresh clone has no library/ — it is gitignored
        return

    undecodable: list[str] = []
    for path in recordings:
        raw = path.read_bytes()
        path.read_text(encoding="utf-8")   # must always work; raises here if it does not
        try:
            raw.decode("cp1252")
        except UnicodeDecodeError:
            undecodable.append(path.name)

    assert undecodable, (
        "no committed recording currently contains a byte cp1252 rejects. That does not make "
        "the locale-encoding bug safe — it makes this particular measurement stale.")
