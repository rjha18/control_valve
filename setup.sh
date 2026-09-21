#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "Python 3.12 is required. Set PYTHON_BIN to a compatible executable." >&2
    exit 1
}

"$PYTHON_BIN" -m venv autogen_venv
autogen_venv/bin/python -m pip install --upgrade pip
autogen_venv/bin/python -m pip install -r autogen_requirements.txt
autogen_venv/bin/python -m playwright install chromium

echo "Setup complete."
echo "Activate the environment with: source autogen_venv/bin/activate"
echo "Copy .env.example to .env, add the required keys, and follow README.md."