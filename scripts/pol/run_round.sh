#!/usr/bin/env bash
# One data round on the remote: collect -> features -> Stage A (POL, Markov) -> gates. Resumable: rerun the
# same command and finished chunks / episodes are skipped. Stop cleanly after the current chunk with
#   touch logs/$ROUND/STOP
# Launch (never in the foreground of an ssh session):
#   setsid nohup bash scripts/pol/run_round.sh > /dev/null 2>&1 < /dev/null &
# Settings (env): ROUND, TASKS, CHUNKS (chunks of CHUNK kept episodes per task), WORKERS, SEED_BASE, STEPS_POL,
#   PCA_FROM (path of an existing pca.npz to reuse instead of fitting a new PCA), DRY_RUN=1 (print every command, run none).
set -u
cd "$(dirname "$0")/../.."
ROUND=${ROUND:-round1}
TASKS=${TASKS:-"SealedBin-3 GatherInject-3 BatteriesSort-3"}
CHUNKS=${CHUNKS:-3}
CHUNK=${CHUNK:-100}
WORKERS=${WORKERS:-6}          # collect.py caps this by free RAM/VRAM (0.5 GB per worker, 1 GB kept free)
SEED_BASE=${SEED_BASE:-10000}  # chunk k of a task uses seeds SEED_BASE + 1000 k ...; pilot seeds are < 1000
STEPS_POL=${STEPS_POL:-5000}
FEAT=data/features_$ROUND      # new PCA fit on this round's train episodes, so features live apart from the pilot's
RUNS=runs/$ROUND
LOG=logs/$ROUND
PCA_ARG=(--fit-pca); [ -n "${PCA_FROM:-}" ] && PCA_ARG=(--pca "$PCA_FROM")
if [ "${DRY_RUN:-0}" = 1 ]; then LOG=$(mktemp -d); RUNS=$LOG; fi   # dry run touches nothing in the repo
mkdir -p "$LOG" "$RUNS"
exec 3>&1
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN: $*" >&3; else "$@"; fi; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/progress.log"; }
stop_requested() { [ -e "$LOG/STOP" ] && say "STOP file found, stopping before: $1" && exit 0; }

say "start round=$ROUND tasks=[$TASKS] chunks=$CHUNKS x $CHUNK workers<=$WORKERS commit=$(git rev-parse --short HEAD)"
df -h data | tail -1 | tee -a "$LOG/progress.log"

# ---------------------------------------------------------------- collect (chunks interleaved over tasks)
for k in $(seq 0 $((CHUNKS - 1))); do
  for t in $TASKS; do
    s=$((SEED_BASE + 1000 * k))
    [ -e "data/pol/$t/summary-train-$s.json" ] && continue
    stop_requested "collect $t chunk $k"
    say "collect $t chunk $k (seeds from $s)"
    run .venv-sim/bin/python scripts/pol/collect.py --task "$t" --episodes "$CHUNK" --seed-start "$s" --split train \
      --out data/pol --workers "$WORKERS" >> "$LOG/collect_$t.log" 2>&1
    say "  $(tail -1 "$LOG/collect_$t.log")  disk: $(df -h data | tail -1 | awk '{print $4}') free"
  done
done

# ---------------------------------------------------------------- features (skips episodes already done)
stop_requested "features"
say "features -> $FEAT"
run .venv-policy/bin/python scripts/pol/precompute_features.py --data data/pol --tasks $TASKS --out "$FEAT" --batch 16 \
  "${PCA_ARG[@]}" \
  >> "$LOG/features.log" 2>&1 || { say "features FAILED (exit $?)"; exit 1; }
say "features done: $(du -sh "$FEAT" | cut -f1)"

# ---------------------------------------------------------------- Stage A + gates
stop_requested "stage A"
say "stage A Markov"
run .venv-policy/bin/python -m pol.train_stage_a --variant Markov --data data/pol --features "$FEAT" --tasks $TASKS \
  --out "$RUNS/stage_a/Markov" --batch-size 128 > "$LOG/stage_a_Markov.log" 2>&1 || say "Markov FAILED (exit $?)"
say "stage A POL ($STEPS_POL steps)"
run .venv-policy/bin/python -m pol.train_stage_a --variant POL --data data/pol --features "$FEAT" --tasks $TASKS \
  --out "$RUNS/stage_a/POL" --steps "$STEPS_POL" --batch-size 128 > "$LOG/stage_a_POL.log" 2>&1 || say "POL FAILED (exit $?)"
say "gates"
run .venv-policy/bin/python -m pol.gates --pol "$RUNS/stage_a/POL" --markov "$RUNS/stage_a/Markov" --data data/pol \
  --features "$FEAT" --tasks $TASKS --out "$RUNS/gates.json" > "$LOG/gates.log" 2>&1
say "done, gates exit=$?"
