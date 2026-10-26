#!/usr/bin/env bash
# Staged experiment pipeline for one GPU (see docs/RUNBOOK.md). Each stage writes under runs/
# and stops on failure; gates in configs/manifest.json must be set to "verified" by a human
# after reviewing the stage outputs, otherwise training refuses to start.
#
#   bash scripts/run_pipeline.sh check       # simulator + dummy runner checks
#   bash scripts/run_pipeline.sh collect     # belief-expert demos for every task
#   bash scripts/run_pipeline.sh probe       # shortcut probes on collected demos
#   bash scripts/run_pipeline.sh smoke       # real 7B forward/backward + VRAM profile
#   bash scripts/run_pipeline.sh train  VARIANT SEED
#   bash scripts/run_pipeline.sh eval   VARIANT SEED
#   bash scripts/run_pipeline.sh report REFERENCE CANDIDATE
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# shellcheck disable=SC1091
source "${VENV:-$ROOT/.venv}/bin/activate"
EXT=${EXT:-$ROOT/external}
CKPT=${CKPT:-$(ls -d checkpoints/hf/models--openvla--openvla-7b/snapshots/* 2>/dev/null | head -1)}
TASKS=${TASKS:-"BatteriesSortHard-3-VLA-v0 BatteriesSortHard-6-VLA-v0 BatteriesSortFlash-3-VLA-v0 BatteriesSortCompound-3-VLA-v0"}
EPISODES=${EPISODES:-250}
EPOCHS=${EPOCHS:-3}

case "${1:-}" in
  check)
    python scripts/check_batteries_sort.py BatteriesSortHard-3-VLA-v0
    python scripts/official_dummy.py --output runs/official-dummy.json ;;
  collect)
    for t in $TASKS; do
      python scripts/collect_batteriessort.py --task "$t" --episodes "$EPISODES" --out data/batteriessort
    done ;;
  probe)
    for t in $TASKS; do python scripts/probe_shortcuts.py "data/batteriessort/$t" | tee "runs/probe-$t.json"; done ;;
  smoke)
    sample=$(ls data/batteriessort/*/000000.npz | head -1)
    for v in B0 EEM KEM; do
      python scripts/oft_smoke.py --source "$EXT/openvla-oft" --checkpoint "$CKPT" --sample "$sample" \
        --variant-config "configs/variants/$v.json" --backward --output "runs/oft-smoke-$v.json"
    done ;;
  train)
    v=${2:?variant}; s=${3:-17}
    data=(); for t in $TASKS; do data+=("data/batteriessort/$t"); done
    python -m jem.cli train --variant-config "configs/variants/$v.json" --data "${data[@]}" \
      --source "$EXT/openvla-oft" --checkpoint "$CKPT" --certificate configs/label-certificate.json \
      --seed "$s" --epochs "$EPOCHS" --output "runs/train/$v-s$s" ;;
  eval)
    v=${2:?variant}; s=${3:-17}
    # shellcheck disable=SC2086
    python scripts/evaluate_oft.py --source "$EXT/openvla-oft" --checkpoint "$CKPT" \
      --run "runs/train/$v-s$s" --tasks $TASKS --output "runs/eval/$v-s$s" ;;
  report)
    ref=${2:-KEM}; cand=${3:-EEM}
    python - "$ref" "$cand" <<'PY'
import json, sys, glob
rows = []
for f in glob.glob("runs/eval/*/records.json"):
    rows += json.load(open(f))
json.dump(rows, open("runs/eval/all-records.json", "w"))
PY
    python -m jem.cli report runs/eval/all-records.json --reference "$ref" --candidate "$cand" \
      --output "runs/report-$ref-vs-$cand.json" ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
