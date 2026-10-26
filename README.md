# Pending-Outcome Ledger (POL)

Code for **"Not Yet Seen: Pending-Outcome Memory for Vision-Language-Action Policies"**: a fixed-capacity
memory for π0 that keeps the outcome of a gripper action as a posterior over learned outcome hypotheses
while it cannot be seen, knows whether it *could* be seen, and resolves it when the gripper returns to
the same place. Evaluated on the Unrevealed-Outcome Suite (ManiSkill3 tasks built from MIKASA-Robo-VLA).

- Design: [docs/specs/2026-10-26-pol-design.md](docs/specs/2026-10-26-pol-design.md)
- Implementation plan and status per task: [docs/plans/2026-10-26-pol-implementation.md](docs/plans/2026-10-26-pol-implementation.md)

## Status (2026-10-26)

| Part | State |
|---|---|
| Core (config, gripper events, ledger, outcome model + EM, 13 memory variants) | done, tested on CPU |
| U-Suite envs: `BatteriesSort-{3,6}`, `SealedBin-{3,5}`, `GatherInject-{3,5}`; belief-experts; collector; revealed test | done, run locally (RTX 3050) |
| π0 SigLIP extraction (verified against openpi JAX), feature store, Stage A/B datasets, causal memory replay | done, run locally |
| Stage A training loop + gates, π0-with-memory model, Stage B training, policy server, evaluation, report, RoboMME adapter, orchestration | **not yet written** (plan tasks 11–19) |

No model results exist yet.

## Environments

Simulation and policy need different torch versions, so there are separate venvs:

| venv | Python | Purpose | Setup |
|---|---|---|---|
| `.venv-sim` | 3.10 | ManiSkill 3.0.0b15 + MIKASA @ `3d5a21a` (torch 2.2.1+cu121), envs, collection | see `docs/plans` Task 7 / `scripts/setup_server.sh` (sim part) |
| `.venv-policy` | 3.11 | openpi @ `215abfb` (PyTorch π0, torch 2.7.1, patched transformers 4.53.2), features, training | `bash scripts/pol/setup_policy_env.sh` |
| `.venv-cpu` | 3.11 | unit tests only (torch CPU, numpy, pytest, pyyaml, scikit-learn) | `uv venv -p 3.11 .venv-cpu` + those packages |

## Layout

| Path | Content |
|---|---|
| `src/pol/` | POL package: `events`, `ledger`, `outcome_model`, `memories/` (POL + baselines), `features`, `data`, `labels`, `episodes`, `config` |
| `src/jem/envs/` | U-Suite envs (`batteries_sort.py`, `bins.py`), failure injection, belief-experts, motion primitives |
| `scripts/pol/` | collection, env checks, revealed test, budget calibration, SigLIP extraction/verification, feature precompute, smoke test |
| `configs/pol/` | task, stage A/B, eval and variant configs |
| `tests/pol/` | CPU tests (`.venv-cpu/bin/python -m pytest tests/pol`) |
| `src/jem/` (other modules), `TODO.md`, `docs/architecture.md`, `scripts/*.py` | the earlier EEM version (OpenVLA-OFT), kept for reference, not used by POL |

## Quick local run

```bash
# collect a few episodes (sim venv; at most 2 sim processes on a 14 GB machine)
.venv-sim/bin/python scripts/pol/collect.py --task SealedBin-3 --episodes 4 --workers 1 --out data/pol
# features on the GPU, then the smoke test (policy venv)
.venv-policy/bin/python scripts/pol/precompute_features.py --data data/pol --tasks SealedBin-3 --out data/features --fit-pca --pca-frames 2000
.venv-policy/bin/python scripts/pol/smoke.py --data data/pol --features data/features --tasks SealedBin-3
```

Upstream code used here: MIKASA-Robo (MIT; generated motion primitives in `src/jem/envs/mikasa_primitives.py`),
openpi (Apache-2.0; installed, not vendored), ManiSkill3.
