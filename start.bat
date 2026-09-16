@echo off
rem Auto Agent - double-click to start.
rem
rem Thin shim: find a Python, hand over to launch.py. Every real decision lives there, so this
rem file and start.command stay interchangeable.
setlocal

rem UTF-8 for this process and everything it spawns. Not cosmetic: uv, uvicorn and the framework
rem all print non-ASCII, and a cp1252 stdout raises UnicodeEncodeError mid-run. PYTHONUTF8 makes
rem Python write through the wide console API, which works whatever the console code page is --
rem more reliable than `chcp 65001`, and unlike chcp it does not mutate the user's window.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem /d so it also works when the folder is on another drive. %~dp0 already ends with a backslash.
cd /d "%~dp0"

rem `py -3` (newest installed), then `python`. NOT `py -3.11`: that fails on a PC that only has
rem 3.13, for no benefit -- uv supplies the 3.11 the project runs on.
rem
rem `if not errorlevel 1` rather than comparing %ERRORLEVEL%: it reads the live value, so it is
rem correct inside a parenthesised block without enabledelayedexpansion. And on a PC with no
rem Python at all, `python` resolves to the Microsoft Store app-execution-alias stub, which
rem prints its own message and exits 9009 -- caught here, so we fall through to the help text.
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo Python was not found on this PC.
    echo.
    echo Install it from https://www.python.org/downloads/windows/
    echo During setup, tick "Add python.exe to PATH".
    echo.
    echo Then double-click start.bat again.
    echo.
    pause
    exit /b 1
)

%PY% launch.py %*
set "STATUS=%ERRORLEVEL%"

if not "%STATUS%"=="0" (
    echo.
    echo Auto Agent stopped with an error ^(code %STATUS%^). The reason is above.
    pause
)

exit /b %STATUS%
