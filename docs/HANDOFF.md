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
(stdout + `<out>/train_log.jsonl`; W&B falls back to offline). Tests: `.venv-cpu/bin/python -m pytest tests/pol` → 105 passed.
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

## Data round 1: done 2026-10-27 — the benchmark needs memory, POL does not find the outcome yet

Plan and settings: [plans/2026-10-27-data-round1.md](plans/2026-10-27-data-round1.md); one resumable script
`scripts/pol/run_round.sh` (collect → features → Stage A Markov + POL → gates). Ran 10:55–21:35 at commit 1caa9cd
(labels from 81f9c8f): 3 chunks × 100 kept episodes per task (seeds 10000+), 3 collector workers (VRAM cap).
Collect 4.5 h (BatteriesSort kept 74–81 %), features 5.8 h (938 episodes, 6.0 GB, `data/features_round1`),
Markov 1.5 min, POL 5000 steps 20 min. Stage A: 7814 train / 848 val events. Outputs on the remote:
`runs/round1/stage_a/{POL,Markov}`, `runs/round1/gates.json`, `logs/round1/`.

### Gates fixed to the spec (commits b6d10d0, 0607fe7, 7929aa8)

The first gates (`runs/round1/gates_failedlabel_75638cd.json`, Markov `stage_a/Markov_failedlabel_75638cd`)
scored "failed" = BatteriesSort spill / any cube out of the bin. Those are visible at the re-encounter (a spill is
an empty grasp, a side deflection lies on the table), so Markov reached 0.96 and POL 0.93 (BatteriesSort 1.0 for
both). Fixes, all in `gates.py` / `labels.py`, spec §10.1 updated:
- target = `labels.GATE_TARGET`, the outcome hidden at decision time (spec §4): BatteriesSort **broken vs
  working**, SealedBin **behind the bin vs in**, GatherInject (revealed control) any failure vs in. The Markov
  head is trained on the same target (retrained on round 1, POL unchanged: it never sees labels).
- G1 control = **shuffled evidence** (v1, v2 from another event of the same task and kind). Fake actions cannot
  fail a correct model, since the injected outcomes do not depend on the action; they are still reported.
- group AUROC = **mean of the per-task AUROCs**; pooled, the shuffled control still got 0.71 from the task base
  rates (48 % broken vs 13 % behind the bin). P(target | z) is calibrated per task, on train events with a
  re-encounter (calibrated on all events it scored SealedBin at 0.24; pooled over tasks, 0.35).

Gates now (`runs/round1/gates.json`, exit 2 — N1, G1, G2 fail):

| | POL | POL, shuffled evidence | Markov | needed | |
|---|---|---|---|---|---|
| unrevealed (2 tasks, n=134, 41 positive) | 0.664 | 0.529 | 0.479 | N1: POL − Markov ≥ 0.10 ✓, Markov ≤ 0.75 ✓; G1: POL ≥ 0.80 ✗, shuffled ≤ 0.60 ✓ | |
| revealed, GatherInject (n=85, 15) | 0.596 | 0.589 | **0.921** | N1: \|POL − Markov\| ≤ 0.03 ✗ | |
| −ω, occluded vs visible (n=217) | 0.440 | | | G2 ≥ 0.80 ✗ | |

| task | n / positive | POL | POL, shuffled | Markov | linear probe on v1 (ad hoc) |
|---|---|---|---|---|---|
| BatteriesSort, broken vs working | 67 / 32 | 0.564 | 0.501 | 0.294 | 0.93 |
| SealedBin, behind vs in | 67 / 9 | 0.764 | 0.557 | 0.664 | 0.78 |
| GatherInject, failure vs in | 85 / 15 | 0.596 | 0.589 | 0.921 | — |

AUROC SE ≈ 0.07 (BatteriesSort, GatherInject), ≈ 0.1 (SealedBin, 9 positives). Fake actions ≈ real everywhere.

Reading:
- **The benchmark needs memory.** Markov does not detect the hidden outcome at decision time (0.29 / 0.66).
  The BatteriesSort Markov head is *below* chance (0.29): one head trained across tasks with different targets;
  a per-task logistic regression on the same input gives 0.41. Either way the N1 gain is real but inflated.
- **The evidence is in the features.** Ad-hoc probes (logistic regression, train → val, BatteriesSort): broken vs
  working is readable **only at the release frame** (AUROC 0.93 at t, 0.42–0.44 at t−8…t−4 and t+12, ~0.47
  far before): the lamp flash, then gone — the intended memory problem; the battery itself does not give it away.
  The same probe on events labelled lamp-occluded still gives 0.93: the shutter does not hide the flash from
  the cameras (or `lamp_occluded` is wrong), so BatteriesSort observability labels, and G2 there, mean nothing yet.
- **POL does not pick it up.** BatteriesSort 0.56 vs probe 0.93; SealedBin 0.76 ≈ probe 0.78 (few positives);
  GatherInject 0.60 where the failure is plainly visible (Markov 0.92). Likely cause: the IIC term I(z1; z2)
  rewards only what v1 and v2 share, and a silent outcome is by definition absent from v2 (the lamp is off at
  the re-encounter), so z settles on scene/task (NMI(z, task) = 0.22; val NMI with the outcome falls 0.27 → 0.10
  during training; train `q_entropy` 0.1–0.7, `view_mi` 0.55–0.9). The flash is one frame inside v1's first
  time block, small next to scene variation.
- ω: AUROC 0.44 (n=217), chance; `omega0` 3.4e-3. The pilot's inverted 0.11 (n=11) was noise.

### v1-only IIC tried (commit 657069a): no gain — the encoder has the outcome, the clustering spends z on task

`lam_i1 = 1` adds IIC between two augmentations of v1 (time-block / token dropout, noise), next to the v1↔v2
term. Synthetic test with the outcome only in v1's first block: NMI 0 → 1.0. Retrained POL on round 1 (5000
steps, 30 min; old run kept as `runs/round1/stage_a/POL_v1v2_7929aa8`, `runs/round1/gates_v1v2_7929aa8.json`):

| per task (val) | POL v1↔v2 | POL + v1 IIC | shuffled (v1 IIC) | Markov | probe raw v1 | probe f1 embedding (v1↔v2 / v1 IIC) |
|---|---|---|---|---|---|---|
| BatteriesSort, broken vs working | 0.564 | 0.496 | 0.586 | 0.294 | 0.932 | **0.968 / 0.900** |
| SealedBin, behind vs in (9 pos.) | 0.764 | 0.688 | 0.536 | 0.664 | 0.776 | 0.755 / 0.682 |
| GatherInject, failure vs in | 0.596 | 0.740 | 0.827 | 0.921 | — | — |

Gates still exit 2 (unrevealed POL 0.59, revealed 0.74 vs Markov 0.92, −ω 0.52). With v1 IIC, z encodes task even
more: val NMI(z, task) 0.22 → 0.48, NMI(z, task:kind) 0.29 → 0.59; val BatteriesSort events fall in 2 z, SealedBin
in 1. GatherInject's gain is context (shuffled 0.83 > 0.74).

Reading: the learned v1 embedding f1 separates broken from working at 0.97 (linear probe), so the encoder keeps
the evidence; it is lost when f1 is assigned to the M = 6 hypotheses. These are shared over all tasks and kinds,
and the IIC marginal term (uniform z over the batch) is maximised by spending z on task:kind, the largest
variation, leaving one or two z per task. Config default is now `lam_i1 = 1.0`; it did not help on round 1.

### Clustering within task:kind tried (commit 0ed9bc8): z leaves the task, but finds no outcome

InfoNCE candidates and inference bank views from the event's own task:kind group, IIC as I(z1; z2 | group)
(both terms, `lam_i = lam_i1 = 1`). Same commit fixes the inference bank: it was `items_tr[:4096]`, ordered by
task, and held **no BatteriesSort view** on round 1 (now a random sample). Synthetic test: unconditional NMI(z,
task) = 1 and within-task 0 → within-task 1.0. Retrained on round 1 (5000 steps, 32 min; previous run kept as
`runs/round1/stage_a/POL_v1iic_657069a`, `gates_v1iic_657069a.json`):

| val | POL v1↔v2 | + v1 IIC | + within task:kind | Markov | probe raw v1 |
|---|---|---|---|---|---|
| BatteriesSort, broken vs working (POL) | 0.564 | 0.496 | **0.487** | 0.294 | 0.932 |
| BatteriesSort, probe on f1 embedding | 0.968 | 0.900 | 0.760 | | |
| SealedBin, behind vs in (POL; 9 pos.) | 0.764 | 0.688 | **0.496** | 0.664 | 0.776 |
| SealedBin, probe on f1 embedding | 0.755 | 0.682 | 0.695 | | |
| GatherInject, failure vs in (POL) | 0.596 | 0.740 | 0.562 | 0.921 | — |
| val NMI(z, task) | 0.22 | 0.48 | **0.02** | | |

Gates exit 2 (unrevealed POL 0.49 = shuffled 0.50, −ω 0.47). z is now spread over all 6 values inside each task
(train `q_entropy` 0.02, `view_mi` 1.42 of 1.79 max), and val NMI with the outcome is 0.09: the clusters follow
another within-task variation that v1 and v2 share (likely slot / object position), not the outcome. The encoder
also keeps less of the outcome (f1 probe 0.97 → 0.76). The bank fix and the conditioning landed together, so
their separate effects are not measured.

Reading across the three objectives: the outcome is linearly readable from the frozen features (0.93), but each
unsupervised clustering picks the largest variation available to it — task, then within-task position — and a
one-frame lamp flash is never the largest. Config defaults now: `lam_i1 = 1.0`, conditioning always on.

### Scene residual from the pre-window + the view's actions, per time block (commit f83853f): no gain

`scene_mean` takes `[pooled, act1]` (`act2` for the re-encounter: the 16 actions observed at the view's frames,
also passed by the Stage B producer) and predicts each of the 4 time blocks (flag `model.scene_act`, default on;
grouping and `lam_i1 = 1` kept). Synthetic test (hidden slot moves both views by 3x the outcome, the actions show
the slot): pre-window only → NMI(z, slot) 1, NMI(z, outcome) 0; with the actions → outcome 1, slot 0. Retrained on
round 1 (5000 steps, 34 min; previous run kept as `runs/round1/stage_a/POL_grouped_0ed9bc8`, its gates in
`logs/round1/gates_grouped.log` — `runs/round1/gates.json` is now this run):

| val | + within task:kind (0ed9bc8) | + residual from actions (f83853f) | probe raw v1 |
|---|---|---|---|
| BatteriesSort, broken vs working (POL) | 0.487 | **0.471** | 0.932 |
| BatteriesSort, probe on f1 embedding | 0.760 | 0.861 | |
| SealedBin, behind vs in (POL; 9 pos.) | 0.496 | **0.533** | 0.776 |
| SealedBin, probe on f1 embedding | 0.695 | 0.538 | |
| GatherInject, failure vs in (POL) | 0.562 | 0.517 | — |
| val NMI(z, task) | 0.02 | 0.05 | |

Gates exit 2 (unrevealed POL 0.50, shuffled 0.45, −ω 0.45). Train `view_mi` is still 1.39 (of 1.79), `q_entropy`
0.03, val NMI(z, outcome) 0.09: v1 and v2 still agree strongly on a variation that neither the pre-window pooled
features nor the actions predict, and z follows it, not the outcome. Not checked what it is; candidates are what
the re-encounter shares with the event by construction (same place within rho, same object), readable in the
32×256 tokens but not in the 2×1152 pooled pre-window.

Code note: the scene head changed shape, so checkpoints from before f83853f (`POL_v1v2_*`, `POL_v1iic_*`,
`POL_grouped_*`) load only at their own commits.

### What z encodes within a task (diagnostic on the f83853f checkpoint, CPU)

NMI(argmax q, variable) within each task:kind group, size-weighted per task, against a shuffled-variable chance
level (5 permutations); evaluator-only variables from the episodes and sim logs. Train split (n = 3570 / 2121 /
2123 events), by kind where it matters:

| task | strongest variables (NMI, chance ≈ 0.00–0.01) | outcome |
|---|---|---|
| BatteriesSort R+ | where the battery goes: tester socket vs back to its slot **0.39**; kind of the re-encounter event 0.31; re-encounter gap 0.17 | broken vs working ≈ chance (gate 0.47) |
| SealedBin R+ / R0 | release position on a 3×3 x–y grid **0.24 / 0.26** | 0.01 |
| GatherInject R+ / R0 | release position on a 3×3 x–y grid **0.27 / 0.24** | 0.00 |
| all G+ | nothing above 0.08 (x–y grid) | — |

Object id, event ordinal, time in episode, height and re-encounter distance are at chance. Val (n = 396 / 220 /
232) shows the same order for BatteriesSort (socket vs slot 0.39) and an x–y grid above chance elsewhere, with
chance levels of 0.07–0.13 at that size. (The `outcome` row in the log counts events without an outcome as a
category, so for BatteriesSort R+ it mirrors socket vs slot; within inserts z does not separate broken/working.)

Reading: within a task:kind group z clusters **where** the release happens (bin / place in x–y, or the task
phase: insert into the tester vs put back), which the re-encounter shares with the event by construction (same
place within rho). Neither the 2×1152 pooled pre-window nor the actions remove it, because the scene head never
sees the end-effector position. Grasp clusters match none of the measured variables.
Log: `logs/round1/zdiag.log` on the remote; script `/tmp/pol_zdiag.py`.

### End-effector positions in the scene head (commit 2f94740): no change, and the residual removes nothing

`scene_mean` also takes `pos1` / `pos2` (proprio[:3] at the 16 frames of each view; flag `model.scene_pos`).
Synthetic test: hidden place shared by both views, shown only by the positions → NMI(z, outcome) 0 → 1. Retrained
on round 1 (5000 steps, 33 min; previous run kept as `runs/round1/stage_a/POL_sceneact_f83853f`,
`runs/round1/gates_sceneact_f83853f.json`):

| val | + residual from actions (f83853f) | + positions (2f94740) | probe raw v1 |
|---|---|---|---|
| BatteriesSort, broken vs working (POL) | 0.471 | **0.504** | 0.932 |
| BatteriesSort, probe on f1 embedding | 0.861 | 0.896 | |
| SealedBin, behind vs in (POL; 9 pos.) | 0.533 | **0.502** | 0.776 |
| SealedBin, probe on f1 embedding | 0.538 | 0.743 | |
| GatherInject, failure vs in (POL) | 0.517 | 0.557 | — |

Gates exit 2 (unrevealed POL 0.50, shuffled 0.42, −ω 0.47). The z diagnostic is unchanged
(`logs/round1/zdiag_scenepos.log`): SealedBin / GatherInject R+ still follow the x–y place (0.25 / 0.30),
BatteriesSort R+ insert vs put-back (0.38).

Why (`logs/round1/placeprobe.log`, `/tmp/pol_placeprobe.py`): a linear probe of the 3×3 place cell of R+ events
(train → val) reads it **as well from the residual as from raw v1** in both checkpoints — SealedBin 0.92 / 0.92,
GatherInject 0.95 / 0.96, BatteriesSort 0.75 / 0.75 (chance 0.17–0.31) — and val scene MSE moves by 2 %
(0.0289 → 0.0284). The scene head (MLP, 256 hidden → 4×32×256, MSE) barely fits the view beyond its mean, so
`v − scene_mean` keeps the place, the motion and the task signature; the action and position inputs
(f83853f, 2f94740) were never used. The synthetic tests pass because there the nuisance is a large, clean,
2-valued pattern.

### Closed-form ridge scene head, prototype only (no model change): place and outcome do not come apart

Before touching the model: ridge regression (closed form, standardised inputs, pooled reduced to 384 PCs) from the
inputs to v1's per-block tokens, fit on train; on val, linear probes on the residual's block means — the 3×3 place
cell of R+ events (want: chance, 0.17–0.31) and the gate target (want: raw-v1 level). `logs/round1/ridgeproto.log`,
`/tmp/pol_ridgeproto.py`:

| scene prediction | val MSE | SB place | GI place | BS place | SB target | BS target |
|---|---|---|---|---|---|---|
| none (raw v1) | 0.315 | 0.92 | 0.95 | 0.75 | 0.79 | 0.92 |
| train mean | 0.144 | 0.92 | 0.96 | 0.74 | 0.79 | 0.92 |
| (MLP scene head, f83853f / 2f94740) | 0.026–0.029 | 0.92 | 0.96 | 0.75 | | |
| ridge pooled, λ=1 / 0.1 / 0.01 | 0.044 / 0.012 / 0.011 | 0.91 / 0.85 / 0.67 | 0.97 / 0.93 / 0.73 | 0.75 / 0.68 / 0.65 | 0.76 / 0.69 / 0.66 | 0.92 / 0.93 / 0.93 |
| ridge pooled+act+pos, λ=0.01 | 0.009 | 0.67 | 0.74 | 0.68 | 0.64 | **0.65** |
| + Fourier(pos), + task one-hot, λ=0.01 | 0.009 | 0.69–0.70 | 0.70 | 0.64–0.65 | 0.65 | 0.62 |
| act+pos+Fourier+task only, any λ | 0.024–0.033 | 0.90 | 0.94–0.97 | 0.75 | 0.76–0.77 | 0.92–0.93 |

Reading:
- The MLP scene head was underfitting (val MSE 3× the ridge's), which is why the action/position inputs did nothing.
- No setting reaches the criterion. Fitting the scene harder lowers the place only to 0.65–0.74 (still 3–4× chance)
  and lowers the outcome **at the same rate**: SealedBin behind/in 0.79 → 0.65 (behind vs in the bin is itself a
  place of the cube), BatteriesSort broken/working 0.92 → 0.62–0.65 once actions and positions join the pooled
  pre-window (pooled alone keeps it at 0.93).
- So on these features, "where" and "what happened" are not separable by regressing out the inputs: a residual
  that hides the place also hides the outcome. The one trade-off that keeps BatteriesSort's outcome — ridge on the
  pooled pre-window only, λ=0.01 — removes little of the place (0.75 → 0.65).

## Next (decision)

Six steps (four objectives, a stronger residual, this prototype): the unsupervised outcome clustering does not
separate the silent failures from what the re-encounter shares by construction, and the residual route cannot
remove one without the other.

1. Plan B per spec §10.1, or weak supervision (a few labelled events per task) — which changes the paper's claim.
2. Last cheap try on the residual route, BatteriesSort only: ridge on the pooled pre-window (λ=0.01) as the scene
   head, retrain, gates. Expected small effect (place 0.75 → 0.65).
3. Independent: BatteriesSort occluder check (G2), per-task Markov head (N1), the Stage B producer should pass
   the task:kind group to the outcome model (now whole bank).

Probe scripts (not in the repo, lost on reboot): `/tmp/pol_{breakdown,silent,leak,leak2,embprobe,embprobe_act,embprobe_pos,zdiag,placeprobe,ridgeproto}.py`
on the remote; probe logs `logs/round1/embprobe_{grouped,sceneact}.log`.

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
