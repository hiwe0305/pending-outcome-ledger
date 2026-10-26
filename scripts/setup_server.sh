#!/usr/bin/env bash
# One-time setup on a rented 1x A100/H100 80GB server (Linux x86_64, NVIDIA driver for CUDA 12.1+,
# Vulkan for SAPIEN rendering). Creates .venv, pinned sources in external/, the 7B checkpoint,
# then runs the unit tests and preflight. Safe to re-run.
#
#   bash scripts/setup_server.sh            # full setup
#   SKIP_WEIGHTS=1 bash scripts/setup_server.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python3.10}   # MIKASA requires <3.12
VENV=${VENV:-$ROOT/.venv}
EXT=${EXT:-$ROOT/external}

pin() { "$PYTHON" -c "import json;print(json.load(open('configs/manifest.json'))['sources']['$1']['revision'])"; }
clone_at() {  # repo dir revision
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" fetch -q --all
  git -C "$2" checkout -q "$3"
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] || { echo "revision mismatch in $2"; exit 1; }
}

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

# 1) Simulator: MIKASA pins torch 2.2.1, numpy 1.23.5, mani_skill 3.0.0b15.
mkdir -p "$EXT"
clone_at https://github.com/CognitiveAISystems/MIKASA-Robo "$EXT/MIKASA-Robo" "$(pin mikasa)"
python -m pip install -q torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q -e "$EXT/MIKASA-Robo"

# 2) OpenVLA-OFT: pinned source checkout + the bidirectional-attention Transformers fork.
#    TensorFlow/dlimp (RLDS loaders) are not needed by jem and are skipped (--no-deps).
clone_at https://github.com/moojink/openvla-oft "$EXT/openvla-oft" "$(pin oft)"
python -m pip install -q -r configs/requirements-oft.txt
python -m pip install -q -e "$EXT/openvla-oft" --no-deps

# 3) This project.
python -m pip install -q -e '.[test,audit]'
python scripts/build_planner_primitives.py "$EXT/MIKASA-Robo"

# 4) Pinned initialization (~15 GB).
if [ -z "${SKIP_WEIGHTS:-}" ]; then
  python scripts/download_checkpoint.py
fi

# 5) Checks that need no GPU work beyond a CUDA query.
python -m pytest -q
python -m jem.cli preflight --output runs/preflight.json
echo "Setup done. Next: docs/RUNBOOK.md, stage 1 (simulator checks)."
