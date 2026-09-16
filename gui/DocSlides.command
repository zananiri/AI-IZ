#!/usr/bin/env bash
# Double-click entry point for the docslides launcher GUI on macOS.
# Prefers the project venv's python; falls back to python3 on PATH.
#
# First run: macOS may refuse to execute this until you right-click it and
# choose "Open" once (Gatekeeper), or run: chmod +x gui/DocSlides.command

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd .. && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

if [ -x "$VENV_PY" ]; then
  "$VENV_PY" launcher.py
else
  python3 launcher.py
fi
status=$?

if [ $status -ne 0 ]; then
  echo
  echo "Launcher exited with an error (code $status). If this is the first run, try:"
  echo "  ./scripts/setup.sh"
  echo "to create the virtual environment first."
  read -n 1 -s -r -p "Press any key to close..."
fi
