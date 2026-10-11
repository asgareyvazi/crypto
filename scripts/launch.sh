#!/usr/bin/env bash
# Launch script for ICBT on Linux/macOS.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: Python 3 not found. Set PYTHON=/path/to/python3 and retry." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -e '.[dev]' >/dev/null

case "${1:-gui}" in
  cli)
    shift
    icbt "$@"
    ;;
  gui)
    HOST="${HOST:-127.0.0.1}"
    PORT="${PORT:-8765}"
    echo "ICBT dashboard starting on http://$HOST:$PORT"
    echo "Press Ctrl+C to stop."
    icbt gui --host "$HOST" --port "$PORT"
    ;;
  *)
    icbt "$@"
    ;;
esac
