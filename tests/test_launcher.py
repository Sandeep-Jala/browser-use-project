"""`launch.py` — the double-clickable bootstrap.

Everything here is a pure function over injected dependencies. No network, no real `uv`, no
subprocess, and never the repo's own `.env`: a test that touched it would either read the
developer's real credentials or overwrite them.

`import launch` needs no plumbing — `pythonpath = ["."]` in pyproject.toml, the same setting that
lets these tests import `automation`.

The two cases most worth having are the ones that encode a mistake already made once:

* `test_python_floor_accepts_a_stock_mac` — the first draft of this launcher gated on Python 3.11
  to match `.python-version`. That is wrong: `uv sync` downloads and owns the project's 3.11, and
  Apple ships `/usr/bin/python3` as 3.9.6, so the gate would have rejected an ordinary Mac for a
  setup uv handles fine.
* `test_server_argv_uses_dash_m_not_the_console_script` — `uv run auto-agent` looks tidier and is
  subtly broken. `-m` puts the working directory on `sys.path[0]`; the console script relies on
  the editable-install `.pth`, which iCloud hides on this very project (pyproject.toml:39-41).
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

import launch


# ── the settings file ─────────────────────────────────────────────────────────────────────

TEMPLATE = """\
# Sample environment file.
LOGIN_URL=https://your-app.example.com/login
LOGIN_EMAIL=you@example.com
LOGIN_PASSWORD=your-password-here
LLM_PROVIDER=azure
AZURE_OPENAI_KEY=your-azure-openai-key-here
# GROQ_API_KEY=gsk-your-groq-key-here
"""

FILLED = """\
LOGIN_URL=https://real.example.com/login
LOGIN_EMAIL=someone@example.com
LOGIN_PASSWORD=hunter2
AZURE_OPENAI_KEY=sk-real-looking-key
"""


def test_parse_env_file_handles_what_a_human_editor_produces():
    text = (
        "\ufeff# a comment\n"
        "\n"
        "export LOGIN_URL=https://x.example.com/login\r\n"
        "LOGIN_EMAIL='quoted@example.com'\n"
        'LOGIN_PASSWORD="has spaces"\n'
        "AZURE_OPENAI_ENDPOINT=https://r.example.com/openai/v1?api-version=2024-01-01\n"
        "  SPACED   =   value  \n"
        "NOT_AN_ASSIGNMENT\n"
    )
    values = launch.parse_env_file(text)

    assert values["LOGIN_URL"] == "https://x.example.com/login"      # export + CRLF + BOM line
    assert values["LOGIN_EMAIL"] == "quoted@example.com"             # single quotes stripped
    assert values["LOGIN_PASSWORD"] == "has spaces"                  # double quotes stripped
    # '=' inside the value must survive: Azure endpoints carry api-version query strings.
    assert values["AZURE_OPENAI_ENDPOINT"].endswith("api-version=2024-01-01")
    assert values["SPACED"] == "value"
    assert "NOT_AN_ASSIGNMENT" not in values
    assert "# a comment" not in values


def test_a_value_still_equal_to_the_template_counts_as_unset():
    """The core rule. `AZURE_OPENAI_KEY=your-azure-openai-key-here` is a placeholder, and
    comparing against the template catches every one of them without guessing at which strings
    look placeholder-ish — so it can never cry wolf over a real key."""
    template = launch.parse_env_file(TEMPLATE)
    env = launch.parse_env_file(TEMPLATE)          # untouched copy: nothing filled in

    missing = launch.missing_or_placeholder(env, template, environ={})
    assert missing == ["LOGIN_URL", "LOGIN_EMAIL", "LOGIN_PASSWORD", "AZURE_OPENAI_KEY"]


def test_empty_and_absent_values_are_both_missing():
    template = launch.parse_env_file(TEMPLATE)
    env = launch.parse_env_file(
        "LOGIN_URL=https://real.example.com/login\n"
        "LOGIN_EMAIL=\n"
        "LOGIN_PASSWORD=   \n"
        "AZURE_OPENAI_KEY=sk-real\n"
    )
    assert launch.missing_or_placeholder(env, template, environ={}) == [
        "LOGIN_EMAIL", "LOGIN_PASSWORD"]


def test_a_filled_in_file_passes_cleanly():
    template = launch.parse_env_file(TEMPLATE)
    assert launch.missing_or_placeholder(
        launch.parse_env_file(FILLED), template, environ={}) == []


def test_a_real_environment_variable_satisfies_a_key_the_file_omits():
    """Mirrors `load_dotenv(override=False)` in automation/config.py: a variable already exported
    in the environment does win at runtime, so demanding it be in the file would be a lie."""
    template = launch.parse_env_file(TEMPLATE)
    env = launch.parse_env_file(FILLED.replace("AZURE_OPENAI_KEY=sk-real-looking-key\n", ""))

    assert launch.missing_or_placeholder(env, template, environ={}) == ["AZURE_OPENAI_KEY"]
    assert launch.missing_or_placeholder(
        env, template, environ={"AZURE_OPENAI_KEY": "sk-from-the-shell"}) == []


@pytest.mark.parametrize("provider,wanted,unwanted", [
    ("", "AZURE_OPENAI_KEY", "GROQ_API_KEY"),
    ("azure", "AZURE_OPENAI_KEY", "GROQ_API_KEY"),
    ("  AZURE  ", "AZURE_OPENAI_KEY", "GROQ_API_KEY"),
    ("groq", "GROQ_API_KEY", "AZURE_OPENAI_KEY"),
])
def test_required_keys_follow_the_configured_provider(provider, wanted, unwanted):
    """automation/llm.py raises SystemExit for whichever key the ACTIVE provider needs. Asking
    for the other one sends someone hunting for a key they do not need."""
    keys = launch.required_keys({"LLM_PROVIDER": provider} if provider else {})
    assert wanted in keys and unwanted not in keys
    assert launch.BASE_REQUIRED_KEYS[0] in keys


def test_env_file_is_created_from_the_template_then_refuses_to_continue(tmp_path):
    """Refusing is the point: nothing validates these keys at server startup, so continuing
    would surface a missing key as a crashed RUN instead of a setup step."""
    (tmp_path / ".env.example").write_text(TEMPLATE, encoding="utf-8")

    with pytest.raises(launch.LaunchError) as exc:
        launch.check_env_file(tmp_path, open_editor=False)

    assert (tmp_path / ".env").is_file()
    message = str(exc.value)
    assert str(tmp_path / ".env") in message              # names the file by absolute path
    assert "AZURE_OPENAI_KEY" in message                  # and says which keys
    assert "LOGIN_PASSWORD" in message


def test_env_file_check_passes_once_it_is_filled_in(tmp_path):
    (tmp_path / ".env.example").write_text(TEMPLATE, encoding="utf-8")
    (tmp_path / ".env").write_text(FILLED, encoding="utf-8")
    launch.check_env_file(tmp_path, open_editor=False)     # must not raise


def test_env_file_check_does_not_overwrite_an_existing_file(tmp_path):
    (tmp_path / ".env.example").write_text(TEMPLATE, encoding="utf-8")
    (tmp_path / ".env").write_text(FILLED, encoding="utf-8")
    launch.check_env_file(tmp_path, open_editor=False)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == FILLED


def test_a_folder_without_the_template_says_so(tmp_path):
    with pytest.raises(launch.LaunchError, match="not a complete copy"):
        launch.check_env_file(tmp_path, open_editor=False)


# ── uv discovery ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("system,needle", [
    ("Darwin", "astral.sh/uv/install.sh"),
    ("Linux", "astral.sh/uv/install.sh"),
    ("Windows", "install.ps1"),
])
def test_uv_install_hint_matches_the_platform(system, needle):
    assert needle in launch.uv_install_hint(system)


def test_uv_candidates_cover_the_paths_a_double_click_cannot_see():
    """A double-clicked .command is a non-interactive LOGIN shell: it reads ~/.zprofile and never
    ~/.zshrc. uv's installer writes to both, so `which` alone misses uv on a machine where it
    works perfectly in Terminal."""
    mac = launch.uv_candidates("Darwin", Path("/Users/someone"), {})
    assert Path("/Users/someone/.local/bin/uv") in mac
    assert Path("/opt/homebrew/bin/uv") in mac

    win = launch.uv_candidates("Windows", Path("C:/Users/someone"),
                               {"LOCALAPPDATA": "C:/Users/someone/AppData/Local"})
    assert all(str(p).endswith(".exe") for p in win)
    assert any("WinGet" in str(p) for p in win)


def test_find_uv_prefers_path_then_falls_back_to_a_known_location(tmp_path):
    installed = tmp_path / ".local" / "bin" / "uv"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\n", encoding="utf-8")

    assert launch.find_uv("Darwin", tmp_path, {}, which=lambda _: "/usr/bin/uv") == "/usr/bin/uv"
    assert launch.find_uv("Darwin", tmp_path, {}, which=lambda _: None) == str(installed)


def test_find_uv_gives_a_copy_pasteable_line_when_uv_is_absent(tmp_path):
    """Deliberately not downloading and running Astral's installer: it edits shell profiles,
    which is a persistent machine change nobody agreed to by double-clicking 'start'."""
    with pytest.raises(launch.LaunchError) as exc:
        launch.find_uv("Darwin", tmp_path, {}, which=lambda _: None)
    message = str(exc.value)
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in message
    assert "again" in message.lower()          # tells them what to do after installing


# ── ports ─────────────────────────────────────────────────────────────────────────────────


def test_port_is_free_agrees_with_a_real_listener():
    """Live, hermetic: the test binds the socket itself. `port_is_free` must mirror
    `_refuse_if_port_taken`'s plain bind (no SO_REUSEADDR) or it would call a port free that
    uvicorn cannot actually take."""
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        assert launch.port_is_free(port) is False
    finally:
        held.close()
    assert launch.port_is_free(port) is True


def test_choose_port_takes_the_default_when_it_is_free():
    port, reason = launch.choose_port(
        8765, 10, is_free=lambda p: True, probe=lambda url: pytest.fail("should not probe"))
    assert (port, reason) == (8765, "free")


def test_choose_port_recognises_our_own_server_and_does_not_start_a_second():
    """Without this branch a second double-click starts a SECOND server, and both of them adopt
    the same artifacts/.auto_agent/run.json — two supervisors each believing they own the one
    run, in a codebase whose central invariant is one run at a time."""
    port, reason = launch.choose_port(
        8765, 10, is_free=lambda p: False,
        probe=lambda url: "<html><title>Auto Agent</title></html>")
    assert (port, reason) == (8765, "already-running")


def test_choose_port_walks_past_a_stranger_on_the_default_port():
    port, reason = launch.choose_port(
        8765, 10, is_free=lambda p: p == 8767,
        probe=lambda url: "<html><title>Grafana</title></html>")
    assert (port, reason) == (8767, "moved")


def test_choose_port_walks_past_a_silent_listener_too():
    """Something bound but not answering HTTP is still not us."""
    port, reason = launch.choose_port(
        8765, 10, is_free=lambda p: p == 8766, probe=lambda url: None)
    assert (port, reason) == (8766, "moved")


def test_choose_port_gives_up_with_a_message_naming_the_range():
    with pytest.raises(launch.LaunchError) as exc:
        launch.choose_port(8765, 3, is_free=lambda p: False, probe=lambda url: None)
    assert "8765" in str(exc.value) and "8767" in str(exc.value)


def test_cdp_port_in_use_warns_but_never_fails(capsys):
    """It blocks RUNS, not the server, and `supervisor._foreign_cdp_owner` refuses the run with a
    better-informed message than this file could produce."""
    launch.warn_if_cdp_busy({"CDP_PORT": "9222"}, connect=lambda p: True)
    out = capsys.readouterr().out
    assert "9222" in out and "refuse to launch a run" in out

    launch.warn_if_cdp_busy({}, connect=lambda p: False)
    assert capsys.readouterr().out == ""


def test_cdp_port_falls_back_to_the_default_when_the_setting_is_junk():
    seen: list[int] = []
    launch.warn_if_cdp_busy({"CDP_PORT": "not-a-number"},
                            connect=lambda p: seen.append(p) or False)
    assert seen == [launch.DEFAULT_CDP_PORT]


# ── starting the server ───────────────────────────────────────────────────────────────────


def test_python_floor_accepts_a_stock_mac():
    """3.9.6 is what Apple's command line tools ship. uv provides the project's 3.11 itself, so
    rejecting 3.9 here would be a self-inflicted failure."""
    launch.check_python((3, 9, 6))
    launch.check_python((3, 8, 0))
    launch.check_python(sys.version_info)


def test_python_floor_rejects_what_cannot_run_this_file():
    with pytest.raises(launch.LaunchError, match="3.8 or newer"):
        launch.check_python((3, 7, 9))


def test_server_argv_uses_dash_m_not_the_console_script():
    """`-m` puts the working directory on sys.path[0]. `uv run auto-agent` does not, and would
    depend entirely on the editable-install .pth that iCloud hides on this project."""
    assert launch.server_argv("/usr/bin/uv", 8765) == [
        "/usr/bin/uv", "run", "python", "-m", "automation.ui", "--no-browser", "--port", "8765"]


def test_child_env_puts_the_repo_on_pythonpath_and_keeps_what_was_there():
    env = launch.child_env({"PYTHONPATH": "/somewhere/else"}, windows=False,
                           repo_root=Path("/repo"))
    assert env["PYTHONPATH"].split(":")[0] == "/repo"
    assert "/somewhere/else" in env["PYTHONPATH"]
    assert "PYTHONUTF8" not in env
    # Unbuffered on both platforms: the server's startup lines are plain prints, and a captured
    # log of a hung start would otherwise be missing the last thing it said.
    assert env["PYTHONUNBUFFERED"] == "1"


def test_child_env_forces_utf8_on_windows_only():
    """On Windows the framework's emoji log lines are dropped by `logging` on a cp1252 stream,
    and one of the dropped lines is what arms the UI's Pause/Resume/Stop buttons."""
    env = launch.child_env({}, windows=True, repo_root=Path("/repo"))
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


class _FakeProc:
    def __init__(self, exit_code=None):
        self._code = exit_code

    def poll(self):
        return self._code


def test_wait_until_ready_returns_as_soon_as_anything_answers():
    """Any HTTP response proves the socket serves — a 403 from `_LocalOnly` included."""
    replies = iter([None, None, ""])
    launch.wait_until_ready(_FakeProc(), 8765, probe=lambda url: next(replies),
                            sleep=lambda s: None)


def test_wait_until_ready_reports_a_dead_child_instead_of_waiting_out_the_timeout():
    """Without the poll() check this would spin for three minutes and then blame a timeout for
    what was really a server that failed to import."""
    with pytest.raises(launch.LaunchError) as exc:
        launch.wait_until_ready(_FakeProc(exit_code=1), 8765,
                                probe=lambda url: None, sleep=lambda s: None)
    assert "exit 1" in str(exc.value)


def test_wait_until_ready_eventually_times_out():
    with pytest.raises(launch.LaunchError, match="did not finish starting"):
        launch.wait_until_ready(_FakeProc(), 8765, timeout=0.01,
                                probe=lambda url: None, sleep=lambda s: None)


def test_launch_imports_only_the_standard_library():
    """`launch.py` runs BEFORE `uv sync` has created the venv, so importing `automation` — or any
    third-party package — would make the bootstrap depend on the very thing it exists to install.

    Checked by AST rather than by searching the text: the module docstring legitimately discusses
    `import automation` in prose, and a substring search flags its own explanation.
    """
    import ast

    tree = ast.parse(Path(launch.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
            else:
                roots.add("<relative>")

    assert "automation" not in roots
    assert "<relative>" not in roots
    non_stdlib = roots - set(sys.stdlib_module_names)
    assert non_stdlib == set(), f"launch.py must stay stdlib-only; found {sorted(non_stdlib)}"
