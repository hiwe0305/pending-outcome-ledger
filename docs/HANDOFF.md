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

## Open problem (do this next)

Pilot gates (remote, `runs/gates.json`): N1/G1/G2 all fail, exit 2. Two causes:
1. Too little val data (8 unrevealed events, 2 failures) — AUROCs are noise.
2. **The outcome model degenerates**: `q_entropy` stays at log 6 = 1.791 for all 3000 steps,
   `p_failed_given_z` ≈ 0.166 for every z, `omega0` ≈ 1e-6. The contrastive score
   `s(v,z,c) = <u(c,z), f(v)>` identifies the event's own view through the context c alone
   (instance discrimination), so z is unused; identical mixture components are an optimum and
   `usage_kl` cannot prevent it (the marginal of uniform posteriors is uniform).

Proposed fix (not yet done):
- add a failing unit test: synthetic data where the context identifies each event -> current code gives uniform q;
- remove the context from the hypothesis embedding `u_z` (z + event kind only; context only in the prior);
- add an IIC-style mutual-information term (minimise per-sample posterior entropy, keep the marginal uniform);
- retrain on the remote: `q_entropy` must drop clearly below 1.79 and `p_failed_given_z` must separate before
  looking at gates; then collect ~100 episodes/task so gates have power.

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
