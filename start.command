#!/bin/bash
# Auto Agent - double-click to start.
#
# This file does three things and nothing else: move to the folder it lives in, find any Python 3,
# and hand over to launch.py. Every real decision (uv, dependencies, the browser, the settings
# file, the port) lives in launch.py, so there is ONE implementation and Windows runs the same
# one via start.bat.
#
# bash, not sh: `read -p` is not POSIX. Everything here is safe on macOS's bash 3.2.

# The `||` is not decoration. Without it, a failed cd would run the launcher against whatever
# directory Terminal happened to open in, which is where the user's home folder gets a .env.
cd "$(dirname "$0")" || { echo "Could not find the Auto Agent folder."; exit 1; }

# Plain `python3` FIRST, deliberately. launch.py needs only 3.8+ and uv downloads the 3.11 the
# project itself runs on, so Apple's command-line-tools python3 (3.9.6) is a perfectly good
# bootstrap. Preferring a newer one would just be a longer search for no gain.
PY=""
for candidate in python3 python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
    echo "Python 3 was not found on this Mac."
    echo
    echo "Open Terminal (Applications > Utilities > Terminal), paste this line, press Return:"
    echo "    xcode-select --install"
    echo
    echo "Click Install, wait for it to finish, then double-click start.command again."
    echo
    read -r -p "Press Return to close this window. " _
    exit 1
fi

"$PY" launch.py "$@"
status=$?

# Only on failure. A clean stop leaves Terminal showing "[Process completed]" anyway, and an
# extra prompt after a deliberate Ctrl+C reads as though something went wrong.
if [ "$status" -ne 0 ]; then
    echo
    echo "Auto Agent stopped with an error (code $status). The reason is above."
    read -r -p "Press Return to close this window. " _
fi

exit $status
