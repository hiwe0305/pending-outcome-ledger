# Handoff — POL (Pending-Outcome Ledger), state on 2026-10-27

Read first: [README](../README.md), design [specs/2026-10-26-pol-design.md](specs/2026-10-26-pol-design.md),
plan with per-task status [plans/2026-10-26-pol-implementation.md](plans/2026-10-26-pol-implementation.md).
Paper draft (not in this repo, anonymous): `~/project/wm-wam-review/latex-paper/eem-paper/main.tex` on the dev laptop.

## Where things run

| Machine | Path | Notes |
|---|---|---|
| Dev laptop (RTX 3050 4 GB, 14 GB RAM) | `~/project/wm-wam-review/code/joint-execution-memory` | git checkout of this repo; venvs `.venv-cpu` (tests), `.venv-sim`, `.venv-policy`. **Max 2 sim processes** (RAM) |
| Remote PC `ssh rtx3060` (ubuntu@192.168.1.8; 64 cores, 94 GB RAM, RTX 3060 12 GB) | `/home/ubuntu/workspace/pending-outcome-ledger` | same repo; `.venv-sim`, `.venv-policy` built and verified. **Isaac Sim of the user holds ~8 GB VRAM — do not touch it**; use `--batch-size 128` for Stage A and the collector's VRAM cap. Long jobs: launch with a script + `setsid nohup`, never `pkill -f` a pattern that also matches your own ssh command line |
| GitHub | https://github.com/hiwe0305/pending-outcome-ledger (public) | push from the laptop; `git pull` on the remote |

## Done (plan tasks 1–11)

Core (events, ledger, outcome model, 13 memory variants), U-Suite envs + experts + collector, revealed test,
pi0 SigLIP extraction (verified vs openpi JAX), features, Stage A/B datasets, causal replay, Stage A trainer
for POL and all baselines (`python -m pol.train_stage_a`), gates (`python -m pol.gates`), live logging
(stdout + `<out>/train_log.jsonl`; W&B falls back to offline). Tests: `.venv-cpu/bin/python -m pytest tests/pol` → 93 passed.
BatteriesSort collection yield fixed: 13 % → 67 % (bin walls, low placement, spill observed as an empty grasp).
GatherInject labels fixed 2026-10-27 (commit 81f9c8f): a slip drops the cube mid-transport and the FSM reports
`R0`, not `R+`; an `R0` now takes the outcome of a cube released after the previous event.

## Outcome-model collapse: fixed 2026-10-27 (commit 2582a5b)

Pilot (3000 steps, batch 128) collapsed: `q_entropy` = log 6, `p_failed_given_z` ≈ 0.166 for all z, `omega0` ≈ 1e-6.
Cause: `u(c,z)` saw the context, so InfoNCE matched each event's own view through c (instance discrimination), z unused.
Fix (spec §6.2, plan Task 4): `u_z` from z + event kind only; views as residuals from `scene_mean(own pre-window)`;
E-step targets from the views only (prior distilled); IIC `−lam_i·I(z1;z2)` between v1/v2 posteriors.
Regression test `test_em_uses_outcome_when_context_identifies_each_event`; 91 tests pass.
Retrain on the remote (old run kept as `runs/stage_a/POL_collapsed_215392d`, `runs/gates_collapsed_215392d.json`):
- train `q_entropy` 1.72 → 0.05–0.28, `z_used` 6, `view_mi` ≈ 1.5 (max 1.79); val NMI 0.30.
- `p_failed_given_z` = [0, .286, 0, 0, 0, .588]. On train, all 13 BatteriesSort failures fall in z=5 (with 9 ok);
  but z also encodes task/kind (NMI(z, family:kind) = 0.31; z=1 ≈ SealedBin R+, z=3 ≈ GatherInject), and the
  SealedBin failures share z=1 with 19 successes. GatherInject had no labelled failures (label bug, fixed since).
- Gates: exit 2. N1 pass (but no revealed-set control), G1 fail (unrevealed AUROC 0.667 = pol_fake, n=8, 2 failures),
  G2 fail, AUROC(−ω) = 0.11 (n=11; mostly noise, see round 1); `omega0` 1.7e-5.

## Data round 1: done 2026-10-27 — gates fail, mainly because the failures are visible

Plan and settings: [plans/2026-10-27-data-round1.md](plans/2026-10-27-data-round1.md); one resumable script
`scripts/pol/run_round.sh` (collect → features → Stage A Markov + POL → gates). Ran 10:55–21:35 at commit 1caa9cd
(labels from 81f9c8f): 3 chunks × 100 kept episodes per task (seeds 10000+), 3 collector workers (VRAM cap).
Collect 4.5 h (BatteriesSort kept 74–81 %), features 5.8 h (938 episodes, 6.0 GB, `data/features_round1`),
Markov 1.5 min, POL 5000 steps 20 min. Outputs on the remote: `runs/round1/{stage_a,gates.json}`, `logs/round1/`.
Stage A: 7814 train / 848 val events.

Gates (`runs/round1/gates.json`, exit 2 — N1, G1, G2 all fail):

| | POL | POL, fake actions | Markov | needed |
|---|---|---|---|---|
| unrevealed (n=158, 33 failures) | 0.926 | 0.927 | **0.956** | N1: POL − Markov ≥ 0.10, Markov ≤ 0.75; G1: POL ≥ 0.80, fake ≤ 0.60 |
| revealed, GatherInject (n=85, 15 failures) | 0.403 | 0.406 | 0.880 | \|POL − Markov\| ≤ 0.03 |
| −ω, occluded vs visible (n=217) | 0.440 | | | G2 ≥ 0.80 |

Per task (val, labelled events with a re-encounter; ad-hoc analysis, AUROC SE ≈ 0.07 at 14–19 failures):

| task | n / failures | POL | Markov | note |
|---|---|---|---|---|
| BatteriesSort | 86 / 19 | 1.00 | 1.00 | only `spill` counts as failed (`labels.FAILED`); a spill is an empty grasp, visible |
| SealedBin | 72 / 14 | 0.62 | 0.78 | `visible_out` (5): Markov mean 0.78; `hidden_out` (9): 0.35 vs 0.20 for `in` |
| GatherInject | 85 / 15 | 0.40 | 0.88 | event kind alone (`R0`) gives 0.83 |

Reading:
- N1 fails for the benchmark, not the model: Markov sees the frame just after the re-encounter (`data.py`,
  `ta = tr + 8`), where most labelled failures are visible. The pooled 0.93–0.96 is inflated by BatteriesSort
  (1.0) and the different base rates of the tasks. Truly silent failures in val: ~9 (`hidden_out`).
  `seated_broken` (32 in val, the broken battery shown only by the lamp) is *not* counted as a failure.
- POL ≈ POL with fake actions everywhere: the posterior does not use the action.
- z still tracks task more than outcome: NMI(z, task) = 0.22, NMI(z, task:kind) = 0.29 (train and val alike);
  BatteriesSort never uses z=1; val NMI with the outcome falls during training 0.27 → 0.10. Not collapsed
  (train `q_entropy` 0.1–0.7, 5–6 z used, `view_mi` 0.55–0.9 vs 1.5 on the pilot).
- ω: AUROC 0.44 with n=217, i.e. chance (the pilot's inverted 0.11 was noise); `omega0` 3.4e-3.

## Next (decision needed before more compute)

Spec §10.1 says N1/G1 fail → stop, Plan B. Before that, decide whether the gate measures the right thing:
1. Failure definition: count `seated_broken` as failed for BatteriesSort, and/or restrict the unrevealed set to
   events whose outcome is hidden at the decision frame (`observable == False`, `hidden_out`). Changes the spec.
2. If the gate is redefined: rerun only `pol.gates` on round 1 (no new data needed; ~1 min), then judge POL
   on the silent subset — val has few such events (~9 `hidden_out` + 32 `seated_broken`).
3. Model, independent of 1–2: posterior ignores the action (fake ≈ real), z encodes task; ω at chance (G2).

## Remaining plan tasks

12 π0 with memory tokens · 13 Stage B loop · 14 policy server · 15 eval + metrics · 16 report into the paper ·
17 RoboMME · 18 orchestration/runbook · 19 paper sync. Stage B at full scale needs an 80 GB GPU.

## Useful commands (remote)

```bash
cd /home/ubuntu/workspace/pending-outcome-ledger && git pull
# whole round, unattended and resumable (settings by env, see the script header; stop: touch logs/$ROUND/STOP)
ROUND=round2 SEED_BASE=20000 setsid nohup bash scripts/pol/run_round.sh > /dev/null 2>&1 < /dev/null &
tail -f logs/round2/progress.log
# single steps
T="SealedBin-3 GatherInject-3 BatteriesSort-3"
.venv-sim/bin/python scripts/pol/collect.py --task SealedBin-3 --episodes 100 --workers 6 --out data/pol
.venv-policy/bin/python scripts/pol/precompute_features.py --data data/pol --tasks $T --out data/features --batch 16
PYTHONUNBUFFERED=1 .venv-policy/bin/python -m pol.train_stage_a --variant POL --data data/pol --features data/features --tasks $T --out runs/stage_a/POL --steps 3000 --batch-size 128
.venv-policy/bin/python -m pol.gates --pol runs/stage_a/POL --markov runs/stage_a/Markov --data data/pol --features data/features --tasks $T --out runs/gates.json
tail -f logs/stage_a_POL.log    # live log
```
W&B: works on the remote again (round 1 synced to project `pol`); the code still falls back to offline on errors.
