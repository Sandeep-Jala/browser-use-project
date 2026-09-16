"""One-click start for Auto Agent. `start.command` (macOS) and `start.bat` (Windows) run this.

**Stdlib only, and on purpose.** This file runs BEFORE the project's dependencies exist, so it
cannot import `automation` or anything out of the venv — and it must not re-exec itself into the
venv either, because that would reintroduce the trap `pyproject.toml` already documents: iCloud
sets the macOS UF_HIDDEN flag on the venv's editable-install `.pth`, Python >= 3.11 silently skips
hidden `.pth` files, and `import automation` then fails. It shells out to `uv` and nothing else.

**It targets Python 3.8+, not 3.11.** `.python-version` pins 3.11 for the PROJECT, and `uv sync`
downloads and manages that interpreter itself. Requiring 3.11 of the bootstrap would reject a
stock Mac — Apple ships `/usr/bin/python3` as 3.9.6 — for a setup uv would have completed without
complaint.

Three decisions worth keeping:

* The server is started as `python -m automation.ui`, never the `auto-agent` console script.
  `-m` puts the working directory on `sys.path[0]`, so `import automation` resolves from the
  source tree whether or not that `.pth` is readable. A console script depends on it entirely.
* `--no-browser` is passed deliberately. `automation/ui/__main__.py` opens the browser on a fixed
  1.5 s `threading.Timer`, which is a bet rather than a readiness check — a cold gradio import
  loses it and hands the user a connection-refused tab. This polls the port instead.
* The child's stdio is INHERITED, not piped. The server writes straight to the window the user is
  looking at, and Ctrl+C reaches it from the OS with no signal plumbing here. That is the opposite
  of what `automation/ui/supervisor.py` does to a RUN, and for the opposite reason: there the
  point is to insulate the run from a Ctrl+C in this terminal, here it is to let one through.

Every printed string is pure ASCII: on Windows this process' own stdout may be cp1252 before it
has a chance to set anything, and a launcher that crashes while explaining a problem is worse than
one that explains it plainly.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_PORT = 8765            # must match automation/ui/__main__.py's DEFAULT_PORT
PORT_SPAN = 10
DEFAULT_CDP_PORT = 9222        # must match automation/config.py's cdp_port default
READY_TIMEOUT_S = 180.0        # a cold first import of gradio + onnxruntime is genuinely slow
READY_POLL_S = 0.25
MIN_PYTHON = (3, 8)

# `gr.Blocks(title="Auto Agent")` in automation/ui/gradio_app.py puts this in the served HTML.
# It is how we tell OUR server on a busy port from somebody else's.
OURS_MARKER = "Auto Agent"

BASE_REQUIRED_KEYS = ("LOGIN_URL", "LOGIN_EMAIL", "LOGIN_PASSWORD")


class LaunchError(Exception):
    """A failure the user can act on. `main` prints it plainly and exits 1."""


def _say(message: str = "") -> None:
    """print() that FLUSHES.

    The server child inherits this process' stdout and writes to it directly, while Python
    block-buffers our own writes whenever stdout is not a terminal. Without the flush the two
    streams interleave by buffer-drain order rather than by when things happened: a captured log
    shows the server announcing its port BEFORE the launcher says it is installing dependencies,
    and a message printed just before a signal arrives can be lost entirely. Both were observed.
    """
    print(message, flush=True)


# ── environment file ──────────────────────────────────────────────────────────────────────


def parse_env_file(text: str) -> "dict[str, str]":
    """A deliberately lenient `.env` reader, used ONLY to check which keys still need filling in.

    Not a python-dotenv reimplementation, and it does not have to be: the run itself loads `.env`
    through python-dotenv, so a disagreement here can only produce a spurious warning, never a
    silent pass. It handles what a human editor actually produces — `export K=V`, quotes, CRLF
    from Notepad, a UTF-8 BOM, `=` inside values, `#` comments.
    """
    values = {}
    for raw_line in text.lstrip("\ufeff").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def required_keys(env_values: "dict[str, str]") -> "tuple[str, ...]":
    """The keys this install genuinely cannot run without, following the configured provider.

    Mirrors `automation/llm.py`, which raises `SystemExit` for a missing AZURE_OPENAI_KEY under
    the default provider and for GROQ_API_KEY under `groq`. Asking for the wrong one would send
    somebody hunting for a key they do not need.
    """
    provider = (env_values.get("LLM_PROVIDER") or "azure").strip().lower()
    key = "GROQ_API_KEY" if provider == "groq" else "AZURE_OPENAI_KEY"
    return BASE_REQUIRED_KEYS + (key,)


def missing_or_placeholder(
    env_values: "dict[str, str]",
    template_values: "dict[str, str]",
    environ: "dict[str, str] | None" = None,
) -> "list[str]":
    """Which required keys are not really set yet.

    Exactly three cases, with no fuzzy matching: absent, empty after stripping, or still equal to
    the value `.env.example` ships. The third is what catches `AZURE_OPENAI_KEY=your-azure-openai-
    key-here`, and comparing against the template beats guessing at placeholder-looking strings —
    it cannot produce a false alarm about a real key.

    `os.environ` is a fallback rather than the primary source, mirroring
    `load_dotenv(override=False)`: a real environment variable does win at runtime, but reading
    the FILE first is what makes this honest on someone else's machine, where your exported
    variables do not exist.
    """
    env = os.environ if environ is None else environ
    missing = []
    for key in required_keys(env_values):
        value = env_values.get(key)
        if value is None or not value.strip():
            value = env.get(key, "")
        if not value.strip():
            missing.append(key)
            continue
        placeholder = template_values.get(key)
        if placeholder is not None and value.strip() == placeholder.strip():
            missing.append(key)
    return missing


def _open_in_editor(path: Path) -> None:
    """Best-effort: put the file in front of them.

    Finder and Explorer both HIDE dotfiles, so "now edit .env" is a step a non-technical
    recipient cannot actually complete. Failure here is not worth reporting — the message that
    follows names the absolute path either way.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-e", str(path)], check=False)
        elif os.name == "nt":
            subprocess.run(["notepad", str(path)], check=False)
    except OSError:
        pass


def check_env_file(repo_root: Path, *, open_editor: bool = True) -> None:
    """Create `.env` from the template if absent, then refuse to continue until it is filled in.

    Refusing is the whole point. Nothing validates these keys at server startup: the server boots
    happily and `automation/llm.py` raises `SystemExit` inside the RUN SUBPROCESS, so a missing
    key reaches the user as a mysteriously crashed run rather than as a setup step they skipped.
    """
    env_path = repo_root / ".env"
    template_path = repo_root / ".env.example"

    if not template_path.is_file():
        raise LaunchError(
            "This folder is missing .env.example, so it is not a complete copy of the project.\n"
            "Ask for the whole folder again (or run: git clone <repo>).")

    template_values = parse_env_file(template_path.read_text(encoding="utf-8"))

    opened_editor = False
    if not env_path.is_file():
        shutil.copyfile(str(template_path), str(env_path))
        _say("[*] created a settings file for you: %s" % env_path)
        if open_editor:
            _open_in_editor(env_path)
            opened_editor = True

    missing = missing_or_placeholder(
        parse_env_file(env_path.read_text(encoding="utf-8")), template_values)
    if not missing:
        return

    # Only claim the editor opened if we actually tried to open it. A message that describes a
    # window the user cannot see sends them looking for it instead of at the path we just printed.
    opened = " It should be open in a text editor now." if opened_editor else ""
    raise LaunchError(
        "Auto Agent needs a few settings before it can run.\n\n"
        "  Edit this file: %s%s\n"
        "  Fill in:        %s\n\n"
        "Save it, then start Auto Agent again." % (env_path, opened, ", ".join(missing)))


# ── uv ────────────────────────────────────────────────────────────────────────────────────


def uv_install_hint(system: str) -> str:
    if system == "Windows":
        return ('  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 '
                '| iex"\n  (or, with winget:  winget install --id=astral-sh.uv -e)')
    if system == "Darwin":
        return ("  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "  (or, with Homebrew:  brew install uv)")
    return "  curl -LsSf https://astral.sh/uv/install.sh | sh"


def uv_candidates(system: str, home: Path, environ: "dict[str, str]") -> "list[Path]":
    """Where uv lives when it is installed but not on PATH.

    This is not belt-and-braces. A double-clicked `.command` runs a NON-INTERACTIVE LOGIN shell,
    which reads `~/.zprofile` but never `~/.zshrc` — and uv's installer writes its PATH line to
    both, so `shutil.which("uv")` misses it on a machine where uv works fine in Terminal. The
    same gap appears right after installing uv, when the already-open window still has the old
    PATH.
    """
    if system == "Windows":
        local_app = environ.get("LOCALAPPDATA", "")
        paths = [home / ".local" / "bin" / "uv.exe"]
        if local_app:
            paths += [Path(local_app) / "Microsoft" / "WinGet" / "Links" / "uv.exe",
                      Path(local_app) / "Programs" / "uv" / "uv.exe"]
        return paths
    return [home / ".local" / "bin" / "uv",
            Path("/opt/homebrew/bin/uv"),
            Path("/usr/local/bin/uv")]


def find_uv(system: "str | None" = None, home: "Path | None" = None,
            environ: "dict[str, str] | None" = None,
            which=shutil.which) -> str:
    """Locate uv, or explain how to get it.

    Deliberately does NOT download and run Astral's installer. That installer edits shell
    profiles, which is a persistent change to someone's machine that nobody consented to by
    double-clicking a file called "start" — and piping a remote script into a shell is not a
    habit a launcher should be teaching a non-technical user. It is a one-time copy-paste.
    """
    import platform
    system = platform.system() if system is None else system
    home = Path.home() if home is None else home
    environ = dict(os.environ) if environ is None else environ

    found = which("uv")
    if found:
        return found
    for candidate in uv_candidates(system, home, environ):
        if candidate.is_file():
            return str(candidate)

    raise LaunchError(
        "Auto Agent needs 'uv' (it installs everything else). It is not installed yet.\n\n"
        "Copy this line, paste it into a terminal, press Enter:\n\n%s\n\n"
        "Then start Auto Agent again." % uv_install_hint(system))


# ── ports ─────────────────────────────────────────────────────────────────────────────────


def port_is_free(port: int) -> bool:
    """A real bind, matching `_refuse_if_port_taken` in `automation/ui/__main__.py` exactly.

    No SO_REUSEADDR, and no connect-test instead: uvicorn will bind without it, so this probe has
    to fail under precisely the conditions uvicorn would. A looser check would call a port free
    that the server then cannot take.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def http_probe(url: str, timeout: float = 1.5) -> "str | None":
    """The response body as text, or None if nothing answered.

    An HTTP ERROR still counts as an answer: `_LocalOnly` returns 403 for an unexpected Host
    header and a bad path returns 404, and both prove a server is serving.
    """
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.read(4096).decode("utf-8", "replace")
        except Exception:
            return ""
    except Exception:
        return None


def choose_port(start: int = DEFAULT_PORT, span: int = PORT_SPAN, *,
                is_free=port_is_free, probe=http_probe) -> "tuple[int, str]":
    """`(port, reason)` where reason is "free" | "already-running" | "moved".

    The "already-running" branch is not a nicety. Without it a second double-click starts a
    SECOND server, and both of them `supervisor.adopt()` the same
    `artifacts/.auto_agent/run.json` — two processes each believing they own the one run, on a
    codebase whose central invariant is one run at a time.
    """
    if is_free(start):
        return start, "free"

    body = probe("http://127.0.0.1:%d/" % start)
    if body is not None and OURS_MARKER in body:
        return start, "already-running"

    for port in range(start + 1, start + span):
        if is_free(port):
            return port, "moved"

    raise LaunchError(
        "Ports %d to %d are all in use, so Auto Agent has nowhere to listen.\n"
        "Restart the computer, or close whatever is using them, and try again."
        % (start, start + span - 1))


def warn_if_cdp_busy(env_values: "dict[str, str]", *, connect=None) -> None:
    """A debugging browser on the CDP port blocks RUNS, not the server. So warn and carry on.

    `supervisor._foreign_cdp_owner` makes the real decision by scanning process command lines
    with psutil, which this file does not have. A TCP connect is the stdlib approximation, and
    making it fatal here would be strictly worse than letting the supervisor refuse the run with
    a better-informed message.
    """
    raw = (env_values.get("CDP_PORT") or "").strip()
    try:
        port = int(raw) if raw else DEFAULT_CDP_PORT
    except ValueError:
        port = DEFAULT_CDP_PORT

    if connect is None:
        def connect(p: int) -> bool:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                return sock.connect_ex(("127.0.0.1", p)) == 0
            finally:
                sock.close()

    if connect(port):
        _say("[!] Something is already using the browser debugging port (%d)." % port)
        _say("    Auto Agent will start, but it will refuse to launch a run until that")
        _say("    browser is closed.")


# ── the server ────────────────────────────────────────────────────────────────────────────


def check_python(version_info=None) -> None:
    """A courtesy floor, not the project's requirement.

    `uv sync` reads `.python-version` and provides 3.11 to the project itself, so the only thing
    that has to be true here is that this file can run. Gating at 3.11 would turn a stock macOS
    (python3 == 3.9.6) into a hard failure for no reason at all.
    """
    info = sys.version_info if version_info is None else version_info
    if tuple(info[:2]) < MIN_PYTHON:
        raise LaunchError(
            "This launcher needs Python %d.%d or newer; this one is %d.%d.\n"
            "Install a current Python from https://www.python.org/downloads/ and try again."
            % (MIN_PYTHON[0], MIN_PYTHON[1], info[0], info[1]))


def server_argv(uv: str, port: int) -> "list[str]":
    """`python -m automation.ui`, never the `auto-agent` console script — see the module
    docstring. Pinned by a test, because it reads like a tidy-up waiting to happen."""
    return [uv, "run", "python", "-m", "automation.ui", "--no-browser", "--port", str(port)]


def child_env(base: "dict[str, str] | None" = None, *, windows: "bool | None" = None,
              repo_root: Path = REPO_ROOT) -> "dict[str, str]":
    """Environment for the server process.

    PYTHONPATH is belt-and-braces for the same `.pth` problem `-m` already solves. PYTHONUTF8 is
    Windows-only and load-bearing there: the framework logs emoji, `logging` SWALLOWS the
    resulting UnicodeEncodeError, and one of the dropped lines is the 'operator control:'
    announcement the UI watches for to arm Pause/Resume/Stop.
    """
    env = dict(os.environ if base is None else base)
    windows = (os.name == "nt") if windows is None else windows
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(repo_root) + os.pathsep + existing) if existing else str(repo_root)
    # The server's three startup lines are plain `print`s, and Python block-buffers them whenever
    # its stdout is not a terminal. That is exactly the case when someone captures this window's
    # output to send it to you, so without this the log of a start that hung is missing the last
    # thing the server managed to say.
    env["PYTHONUNBUFFERED"] = "1"
    if windows:
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(argv: "list[str]", *, label: str, cwd: Path = REPO_ROOT) -> None:
    """One setup command, with its output left on screen. Raises LaunchError on a non-zero exit."""
    _say("[*] %s" % label)
    try:
        completed = subprocess.run(argv, cwd=str(cwd))
    except OSError as exc:
        raise LaunchError("Could not run '%s': %s" % (argv[0], exc)) from None
    if completed.returncode != 0:
        raise LaunchError(
            "%s failed (exit %d). The details are above.\n"
            "If it mentions the network or a proxy, check the internet connection and try again."
            % (label, completed.returncode))


def wait_until_ready(proc, port: int, *, timeout: float = READY_TIMEOUT_S,
                     probe=http_probe, sleep=time.sleep) -> None:
    """Poll until the server answers, or the child dies, or we run out of patience.

    Checking `proc.poll()` every iteration is what turns a dead child into an immediate, accurate
    message instead of three minutes of silence followed by a timeout that blames the wrong thing.
    """
    url = "http://127.0.0.1:%d/" % port
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise LaunchError(
                "The Auto Agent server stopped while starting up (exit %d).\n"
                "The reason is in the messages above." % code)
        if probe(url) is not None:
            return
        sleep(READY_POLL_S)
    raise LaunchError(
        "The Auto Agent server did not finish starting within %d seconds.\n"
        "Leave this window open and try again; if it keeps happening, the messages above will "
        "say why." % int(timeout))


def _stop(proc) -> int:
    """Ctrl+C already reached the child through the console. Just give it time to land."""
    _say("\n[*] stopping Auto Agent...")
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        return proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


def main(argv: "list[str] | None" = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    open_editor = "--no-editor" not in args
    open_browser = "--no-browser" not in args

    try:
        check_python()
        env_path = REPO_ROOT / ".env"
        uv = find_uv()

        run_step([uv, "sync"], label="Installing what Auto Agent needs "
                                     "(first time only, a few minutes)")
        # Unconditional, and no cheap "is it already there" check. The obvious one — look for a
        # chromium-* directory in playwright's cache — reports a false positive as soon as the
        # pinned revision moves, and that failure would surface inside a RUN. This command is
        # already its own idempotency check: it prints "is already installed" and returns.
        run_step([uv, "run", "playwright", "install", "chromium"],
                 label="Checking the browser Auto Agent drives")

        check_env_file(REPO_ROOT, open_editor=open_editor)
        env_values = parse_env_file(env_path.read_text(encoding="utf-8"))
        warn_if_cdp_busy(env_values)

        port, reason = choose_port()
        url = "http://127.0.0.1:%d/" % port
        if reason == "already-running":
            _say("[*] Auto Agent is already running. Opening it in your browser.")
            if open_browser:
                webbrowser.open(url)
            return 0
        if reason == "moved":
            _say("[*] port %d was busy, using %d instead" % (DEFAULT_PORT, port))

        _say("[*] starting Auto Agent...")
        proc = subprocess.Popen(server_argv(uv, port), cwd=str(REPO_ROOT), env=child_env())
        try:
            wait_until_ready(proc, port)
            _say("[*] Auto Agent is ready at %s" % url)
            _say("[*] leave this window open while you use it; press Ctrl+C to stop")
            if open_browser:
                webbrowser.open(url)
            return proc.wait()
        except KeyboardInterrupt:
            code = _stop(proc)
            return 0 if code in (0, 130, -2) else code
        except LaunchError:
            _stop(proc)
            raise

    except LaunchError as exc:
        _say("\nERROR: %s" % exc)
        return 1
    except KeyboardInterrupt:
        _say("\n[*] cancelled")
        return 0
    except Exception as exc:  # noqa: BLE001 - a traceback is not a message for this audience
        _say("\nERROR: Auto Agent could not start because of an unexpected problem.\n"
              "       %s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
