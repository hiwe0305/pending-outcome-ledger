#!/usr/bin/env bash
# Simulation venv (.venv-sim): ManiSkill 3.0.0b15 + MIKASA-Robo at the pinned commit, CUDA torch 2.2.1
# (ManiSkill calls torch.cuda.synchronize() when rendering, so a CPU-only torch does not work).
# Verified locally on 2026-10-26 (RTX 3050, driver 595). Needs an NVIDIA driver with Vulkan.
#
#   bash scripts/pol/setup_sim_env.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MIKASA_COMMIT=3d5a21a99e5df2bcc34c612040b269e394b09438
PY=.venv-sim/bin/python
mkdir -p "$ROOT/external"
[ -d "$ROOT/external/MIKASA-Robo/.git" ] || git clone -q https://github.com/CognitiveAISystems/MIKASA-Robo "$ROOT/external/MIKASA-Robo"
git -C "$ROOT/external/MIKASA-Robo" checkout -q "$MIKASA_COMMIT"
cd "$ROOT"
[ -x "$PY" ] || uv venv -q -p 3.10 .venv-sim
uv pip install -q -p "$PY" torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -q -p "$PY" --prerelease=allow --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match -e external/MIKASA-Robo mplib pytest pyyaml scikit-learn pillow
uv pip install -q -p "$PY" -e . --no-deps
"$PY" -c "import torch, mani_skill, mplib; assert torch.cuda.is_available(), 'CUDA torch required'; print('sim venv ready', torch.__version__)"
