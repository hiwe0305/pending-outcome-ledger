# Architecture

Research contract: [`latex-paper/joint-outcome-proposal/PROPOSAL.md`](../../../latex-paper/joint-outcome-proposal/PROPOSAL.md).

## Data flow at one policy call

```
call-time obs (top+wrist RGB 128x128, proprio 7D, language)
  └─ OFT backbone encode ── spatial tokens ───────────────────────────────┐ direct path
       └─ RoleAttention(N=6 roles + null, query = previous roles + context) │
            roles z_t ─┐                                                    │
executed prefix a_{1..T}, dt_{1..T}  (from the action ledger)              │
frames after each executed action ── EvidenceEncoder (64x64 CNN) e_{1..T}  │
            └─ memory.update(prior roles, z_t, a, dt, e) ─ belief b_t, tokens│
                  └─ EpistemicGate → mode m ∈ {act, verify(i)} → mode token │
                        memory tokens (+ mode token) ── OFT decode ◄───────┘ → action chunk (K=8, L1 head)
```

## Modules (`src/jem/`)

| Module | Role |
|---|---|
| `data.py` | Strict observation allowlist, `Episode` (npz), `calls()` → `Call(t, observation, executed, dt, frames, target, mask)`, episode split, train-only normalizers, task lists (`TASKS` released MIKASA, `SORT_TASKS` BatteriesSort) |
| `backbone.py` | Pinned OpenVLA-OFT loader (refuses unpinned source/fork/checkpoint), differentiable encode/decode with memory prefix, LoRA recipe |
| `memory.py` | `MEMORIES`: GRU, B2 (linear direct joint), B2MLP, **EEM** (identifiable chain-rule kernel), KEM (EventVLA-style keyframe FIFO), MU (recurrent memory tokens). Joint filters are log-space with structural masks |
| `gate.py` | `EpistemicGate` (learned, EEM-IL) and analytic `voi_verify`/`voi_mode` (EEM-VOI) |
| `policy.py` | `JointPolicy(backbone, variant, evidence, gate)`; shared frontend built before variant modules so equal seeds give equal frontends |
| `supervision.py` | Shared auxiliary contract: joint/marginal NLL, gate CE, KEM keyframe BCE; label certificate check (schema `eem-env-log-v1`) |
| `training.py` | Episode-causal training, burn-in 1 call, TBPTT 4 calls, teacher-forced mode/keyframes |
| `environment.py` | Official MIKASA runner integration; ledger of predicted/dispatched/executed actions and the frames observed after each executed action |
| `evaluation.py` | 50 episodes/task on the fixed seed stream; paired train-seed × eval-seed bootstrap; refuses pairs with different provenance |
| `envs/batteries_sort.py` | BatteriesSort env (Hard/Flash/Compound × 3/6). **Needs the simulator; untested here** |
| `envs/expert.py` | Belief-expert decisions from observable evidence only (pure Python, tested) |
| `envs/mikasa_primitives.py` | Generated from the pinned MIT MIKASA planner by `scripts/build_planner_primitives.py` |
| `fixtures.py` | Tiny backbone and synthetic episodes for CPU tests only |

## Belief state

Per role i: joint over 12 states `s = 3·l + q`, `l ∈ {slot, held, socket, elsewhere}`,
`q ∈ {untested, working, broken}` (evidence observed by the model, not hidden quality).
Tested evidence never returns to untested (structural mask). Role i is bound to tray
slot i (canonical slot binding); labels use the same binding.

EEM transition per executed microstep: `log K(l',q'|s) = log ρ(q'|s,x,a) + log D(l'|s,q',x,a)`,
ρ restricted to reachable q'. Correction after every microstep with an evidence frame and
once with the call-time roles: `log b ← log b + c(·) − logsumexp`.

## Fairness rules enforced in code

- Same frontend initialization, cadence, evidence stream, labels and auxiliary losses for
  every memory variant (`tests/test_data_training.py`).
- Only executed actions enter memory; future targets never do (`test_action_depends_only_on_available_history`).
- Evaluation pairs must share benchmark, initialization, label hash, train IDs, chunk size,
  tuning budget and code hash (`evaluation.check_provenance`).
