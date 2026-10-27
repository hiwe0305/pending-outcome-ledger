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
(stdout + `<out>/train_log.jsonl`; W&B falls back to offline). Tests: `.venv-cpu/bin/python -m pytest tests/pol` → 90 passed.
BatteriesSort collection yield fixed: 13 % → 67 % (bin walls, low placement, spill observed as an empty grasp).

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
  SealedBin failures share z=1 with 19 successes. GatherInject has no labelled failures.
- Gates: exit 2. N1 pass (but no revealed-set control), G1 fail (unrevealed AUROC 0.667 = pol_fake, n=8, 2 failures),
  G2 fail and **inverted**: AUROC(−ω) = 0.11 (n=11), i.e. ω is higher on occluded events; `omega0` still 1.7e-5
  (ω is prior-weighted hypothesis disagreement, and the prior is now sharp, prior_nll ≈ 0.39).

## Next

1. Collect ~100 episodes/task so the gates have power (val now has 8 unrevealed events, 0 BatteriesSort failures).
2. Look at ω: sign inversion and the tiny scale under a sharp prior (G2 depends on it).
3. Check whether z keeps encoding task after more data; if so, condition `scene_mean` also on the action, or hold
   the task out of the clusters (e.g. per-family hypotheses).

## Remaining plan tasks

12 π0 with memory tokens · 13 Stage B loop · 14 policy server · 15 eval + metrics · 16 report into the paper ·
17 RoboMME · 18 orchestration/runbook · 19 paper sync. Stage B at full scale needs an 80 GB GPU.

## Useful commands (remote)

```bash
cd /home/ubuntu/workspace/pending-outcome-ledger && git pull
T="SealedBin-3 GatherInject-3 BatteriesSort-3"
.venv-sim/bin/python scripts/pol/collect.py --task SealedBin-3 --episodes 100 --workers 6 --out data/pol
.venv-policy/bin/python scripts/pol/precompute_features.py --data data/pol --tasks $T --out data/features --batch 16
PYTHONUNBUFFERED=1 .venv-policy/bin/python -m pol.train_stage_a --variant POL --data data/pol --features data/features --tasks $T --out runs/stage_a/POL --steps 3000 --batch-size 128
.venv-policy/bin/python -m pol.gates --pol runs/stage_a/POL --markov runs/stage_a/Markov --data data/pol --features data/features --tasks $T --out runs/gates.json
tail -f logs/stage_a_POL.log    # live log
```
W&B: the API key in the remote `~/.netrc` is invalid (401); the user must run `wandb login --relogin` there.
