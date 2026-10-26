# Runbook: 1× A100/H100 80GB server

Each stage ends with outputs under `runs/` that a person reviews before setting the
matching gate in `configs/manifest.json` to `"verified"`. Training refuses to start
while required gates are pending. Nothing here downloads or launches anything by itself.

## Stage 0 — setup (≈1 h, mostly downloads)

```bash
git clone/rsync this folder to the server (exclude data/, runs/, .venv/)
bash scripts/setup_server.sh
```

Creates `.venv` (Python 3.10, torch 2.2.1 cu121), clones MIKASA-Robo and openvla-oft at
the pinned revisions into `external/`, installs the Transformers OFT fork, generates
`src/jem/envs/mikasa_primitives.py`, downloads `openvla/openvla-7b` at the pinned
revision, runs `pytest` and `jem.cli preflight`.

Known risk: OFT pins torch 2.2.0, MIKASA pins 2.2.1. Setup uses 2.2.1 and installs OFT
with `--no-deps`; Stage 3 is where an incompatibility would show.

## Stage 1 — simulator (gate `G2_env`)

```bash
bash scripts/run_pipeline.sh check
```

`check_batteries_sort.py` must exit 0 (registration, observation schema, randomized
quality, no evaluator fields in observations); `official_dummy.py` must complete an
episode. **The BatteriesSort env was written without a simulator; expect to fix
geometry (bin/catch tray/hood offsets) here.** Render a few frames and confirm the hood
hides seating and the lamp shutter covers the lamp.

## Stage 2 — demonstrations and probes (gates `G1_data`, `G3_labels`)

```bash
EPISODES=20 bash scripts/run_pipeline.sh collect   # pilot first
bash scripts/run_pipeline.sh probe
EPISODES=250 bash scripts/run_pipeline.sh collect
```

Check `summary.json` success rates of the belief-expert per task and the strata
(spill / occluded / clean) in `*.log.json`. Probes must report AUC ≤ 0.6. Then write
`configs/label-certificate.json`:

```json
{"review_status": "verified", "source_kind": "env_log", "dataset_revision": "<manifest dataset revision>",
 "observability_verified": true, "label_schema": "eem-env-log-v1",
 "evidence_files": [{"path": "runs/probe-BatteriesSortHard-3-VLA-v0.json", "sha256": "..."}]}
```

## Stage 3 — 7B interface and resources (gates `backbone_injection`, `resource_profile`)

```bash
bash scripts/run_pipeline.sh smoke
```

Records peak VRAM and latency for B0/EEM/KEM forward+backward. If peak VRAM > 75 GB,
enable gradient checkpointing in the OFT language model before training.

## Stage 4 — training and evaluation

Order (each line ≈ one GPU job; budget fixed in `configs/common.json` before any test run):

```bash
for v in B0 KEM EEM; do bash scripts/run_pipeline.sh train $v 17; bash scripts/run_pipeline.sh eval $v 17; done
# then, if H2/H5 hold on seed 17:
for v in GRU MU B2MLP EEM-VOI EEM-callonly EEM-nogate; do ...; done
for s in 29 43; do for v in KEM EEM; do ...; done; done
bash scripts/run_pipeline.sh report KEM EEM
```

Fill the paper tables from `runs/report-*.json` (see `latex-paper/eem-paper/README.md`).

## Local CPU checks (no GPU)

```bash
JEM_PYTHON=/path/to/python bash scripts/verify_local.sh
```
