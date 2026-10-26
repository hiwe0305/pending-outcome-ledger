#!/usr/bin/env bash
# Policy venv (.venv-policy): openpi PyTorch at the pinned commit, patched transformers, this project.
# Verified locally on 2026-10-26 (RTX 3050, torch 2.7.1+cu126, transformers 4.53.2, jax 0.5.3).
#
#   bash scripts/pol/setup_policy_env.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
mkdir -p "$ROOT/external"
[ -d "$ROOT/external/openpi/.git" ] || git clone -q https://github.com/Physical-Intelligence/openpi.git "$ROOT/external/openpi"
git -C "$ROOT/external/openpi" fetch -q origin "$OPENPI_COMMIT" || true
git -C "$ROOT/external/openpi" checkout -q "$OPENPI_COMMIT"

# evdev (lerobot -> pynput -> evdev, keyboard teleop only) needs a C compiler; it is not used here.
# UV_LINK_MODE=copy keeps the transformers patch below out of the shared uv cache (openpi README warning).
( cd "$ROOT/external/openpi" && GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$ROOT/.venv-policy" \
    uv sync --frozen --no-install-package evdev )
SP=$("$ROOT/.venv-policy/bin/python" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp -r "$ROOT/external/openpi/src/openpi/models_pytorch/transformers_replace/"* "$SP/"
"$ROOT/.venv-policy/bin/python" -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly()"
uv pip install -q -p "$ROOT/.venv-policy/bin/python" -e "$ROOT" --no-deps
uv pip install -q -p "$ROOT/.venv-policy/bin/python" scikit-learn pytest

# pi0's SigLIP only (partial restore, ~5 GB RAM), then check it against openpi's JAX SigLIP.
JAX_PLATFORMS=cpu "$ROOT/.venv-policy/bin/python" "$ROOT/scripts/pol/extract_siglip.py"
JAX_PLATFORMS=cpu "$ROOT/.venv-policy/bin/python" "$ROOT/scripts/pol/verify_siglip.py"
echo "policy venv ready"
