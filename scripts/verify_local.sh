#!/usr/bin/env bash
# CPU-only logic checks (no simulator, no 7B): unit tests and synthetic/real-input smokes.
# Needs a Python with torch, numpy and pytest: JEM_PYTHON=/path/to/python bash scripts/verify_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JEM_PYTHON=${JEM_PYTHON:-python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$JEM_PYTHON" -m pytest -q
"$JEM_PYTHON" -m jem.cli synthetic-smoke
if compgen -G "data/samples/*.npz" > /dev/null; then
  "$JEM_PYTHON" -m jem.cli real-data-smoke
fi
"$JEM_PYTHON" scripts/necessity_gate_sorting.py --v2-only > runs/necessity-gate-memories.txt
