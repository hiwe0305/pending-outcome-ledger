# POL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full POL experiment pipeline (U-Suite envs, data, Stage A, π0 Stage B, 12 memory variants, policy server, eval, report into `results.tex`) as specified.

**Architecture:** New package `src/pol/` next to the existing `src/jem/`. Memory is computed outside the policy by `MemoryProducer`s (Stage A, features only) and fed to a `PI0WithMemory` model as K suffix tokens (Stage B). Simulation (Python 3.10, torch 2.2.1) and policy (Python 3.11, torch 2.7.1, openpi) run in separate venvs and talk over the openpi websocket protocol.

**Tech Stack:** PyTorch, openpi @ `215abfb217dbac7d5f1273282331b9b1866c0479` (PyTorch π0), ManiSkill3 3.0.0b15 + MIKASA @ `3d5a21a99e5df2bcc34c612040b269e394b09438`, wandb, numpy, pyarrow, scikit-learn (metrics only), pytest.

**Spec:** [`docs/specs/2026-10-26-pol-design.md`](../specs/2026-10-26-pol-design.md)

## Global Constraints

- Never feed `oracle_info`, `working_mask`, sim object IDs or `*.log.json` content to any variant except Oracle; label-trained heads (Markov, AGM-style) may read logs **only at their own training time**.
- Every memory variant: same frozen features, same Stage B recipe, memory tokens `[K, 50]` + mask `[K]`.
- Entry layout (50): PE(x_ee) 12 | kind one-hot 3 | softmax ℓ padded to 8 | ω 1 | H 1 | age 1 | ctx summary 24.
- Defaults: M=6, τ=0.1, η=0.3 nat, ρ=0.06 m, K=8, H=16, K_exec=8, flow steps 10, eval episodes 50, eval seed start 4242424242, equivalence ±0.03.
- Stage A: AdamW lr 3e-4 wd 0.01, batch 512, 20k steps, warmup 1k, cosine, λ_g=1.0, λ_u=0.1, 32 hard + 31 in-batch negatives.
- Stage B: PaliGemma frozen; expert lr 5e-5, token MLP lr 1e-4, AdamW (0.9, 0.95), eps 1e-8, wd 1e-10, clip 1.0, warmup 1000, cosine to 10 %, batch 64, 20k steps, fp32 master weights for trainable params, checkpoint pick among {10k,15k,20k} by val flow loss.
- Injection rates and step budgets are read from `configs/pol/tasks/*.yaml` and frozen before Stage B (hash goes into provenance).
- W&B: project `pol`, entity from env; `WANDB_MODE=offline` must work; log every 20 steps.
- No git in this checkout: "commit" steps are replaced by running the full local test suite.

## Review Focus

1. Episode with **zero gripper events** (robot never grasps): ledger tokens must be all-masked and π0 must still run — test in Task 3 and Task 12.
2. **More events than K** with all entries pending: eviction must still pick one deterministically and never evict the entry just written — test in Task 3.
3. Re-encounter **with no pending entry in radius ρ**: association weights must be all zero and no entry is updated — test in Task 3.
4. **Server restart / new episode without `reset`** (episode_id changes): server must reset memory state, not carry it over — test in Task 14.
5. Memory precompute must use **only frames ≤ t** for the call at t (causality) — test in Task 10 by perturbing frames after t.

---

## Phase 0 — Core logic (CPU, local)

### Task 1: Config and provenance

**Files:** Create `src/pol/__init__.py`, `src/pol/config.py`, `configs/pol/stage_a.yaml`, `configs/pol/stage_b.yaml`, `configs/pol/eval.yaml`, `configs/pol/variants/{B0,Frames,Random,Markov,AURA,StateMem,CheckVLA,CheckFIFO,AGM,KEM,MU,POL,Oracle}.yaml`, `configs/pol/tasks/{BatteriesSort-3,BatteriesSort-6,SealedBin-3,SealedBin-5,GatherInject-3,GatherInject-5}.yaml`. Test `tests/pol/test_config.py`.

**Interfaces — Produces:**
- `load_config(path: str | Path, overrides: dict | None = None) -> dict` — YAML, key `_base_` (relative path) merged depth-first, then dotted overrides (`{"train.lr": 1e-4}`).
- `stable_hash(obj) -> str` — sha256 of `json.dumps(obj, sort_keys=True)`, first 16 hex.
- `provenance(cfg: dict, code_root: Path, extra: dict) -> dict` — keys: `config_hash`, `code_hash` (sha256 over `src/pol/**/*.py` sorted), `openpi_commit`, `mikasa_commit`, plus `extra`.

- [ ] Test: `_base_` merge overrides scalars and merges dicts; dotted override sets nested key; `stable_hash` equal for reordered dicts; task yaml values equal the spec table (e.g. `SealedBin-5.injection.deflect == 0.2`, `.deflect_behind_frac == 0.5`, `BatteriesSort-3.lamp_mode == "flash"`, `.deferred is True`, `.occlusion == 0.15`, `.flip == 0.05`).
- [ ] Implement; run `pytest tests/pol/test_config.py -q` → pass.

### Task 2: Gripper-load event detector

**Files:** Create `src/pol/events.py`; test `tests/pol/test_events.py`.

**Interfaces — Produces:**
- `@dataclass(frozen=True) Event(t: int, kind: str, x_ee: np.ndarray)`; `KINDS = ("G+", "R+", "R0")`.
- `@dataclass FsmThresholds(w_open: float, w_empty: float, min_loaded_steps: int = 3)`.
- `class GripperFSM(th)`: `reset()`, `step(t: int, width: float, x_ee) -> Event | None`.
- `detect_events(widths: np.ndarray[T], x_ee: np.ndarray[T,3], th) -> list[Event]`.
- `event_windows(t: int, T: int, pre: int = 8, post: int = 16) -> tuple[slice, slice]` (clipped to [0,T)).
- `reencounter_index(events: list[Event], j: int, rho: float = 0.06) -> int | None` — first k>j with ‖x_k−x_j‖ ≤ ρ.
- `calibrate_thresholds(widths: list[np.ndarray]) -> FsmThresholds` — 1D 3-means (empty, loaded, open), thresholds at the midpoints. *(done)*

- [ ] Tests: synthetic width trace open→close-on-object(0.02)→open gives `["G+","R+"]` at the right t; open→close-empty(0.0)→open gives `["R0"]`; loaded for 2 steps (< min_loaded_steps) gives no `G+`; hysteresis: noise ±0.002 around thresholds produces no extra events; `reencounter_index` returns None when nothing within ρ; windows clip at episode ends.
- [ ] Implement; run tests → pass.

### Task 3: Ledger

**Files:** Create `src/pol/ledger.py`; test `tests/pol/test_ledger.py`.

**Interfaces — Consumes:** `Event` (Task 2). **Produces:**
- `ENTRY_DIM = 50`, `M_MAX = 8`, `SUMMARY_DIM = 24`, `ASSOC_DIM = 64`.
- `@dataclass Entry(x: np.ndarray[3], kind: int, logp: np.ndarray[M], omega: float, t: int, summary: np.ndarray[24], assoc: np.ndarray[64])`; property `entropy`.
- `class Ledger(K: int, M: int, eta=0.3, omega0=0.0, rho=0.06, beta=0.0)`:
  `reset()`; `write(e: Entry) -> int` (slot); `association(x, assoc) -> np.ndarray[n]` (**unnormalised gate** `exp(−d²/2ρ² + β(cos−1))` inside ρ, 0 outside, so one look at a bin can inform several pending drops); `resolve(weights, loglik_v2: np.ndarray[n|1, M])` (ℓ ← ℓ + w·loglik, renormalise in log space); `status(i) -> Literal["resolved","pending_unobservable","pending_ambiguous"]`; `featurize(t_now: int) -> tuple[np.ndarray[K,50], np.ndarray[K] bool]`; `evict_index() -> int`.
- `positional_encoding(x: np.ndarray[3]) -> np.ndarray[12]` (sin/cos at frequencies 2π·{1, 4}/m).

- [ ] Tests: write ≤K entries → all slots masked True in order; featurize layout (slice positions per Global Constraints, softmax sums to 1, padded probs zero); resolve with a decisive loglik makes entropy < η and status "resolved"; ω < ω0 & high entropy → "pending_unobservable"; association: two entries, re-encounter at first entry's x gives weight≈1 on it; nothing in ρ → zeros and no change after `resolve`; K+1 writes with all pending → length stays K and the newest entry is kept; zero writes → mask all False (Review Focus 1–3).
- [ ] Implement everything except `evict_index` body (left to the user, see execution notes); run tests.

### Task 4: Outcome model and EM step

**Files:** Create `src/pol/outcome_model.py`; test `tests/pol/test_outcome_model.py`.

**Interfaces — Produces:**
- `class OutcomeModel(nn.Module)(M=6, d_feat=1152, d_tok=256, n_time=4, n_tok=32, d_ctx=512, d_act=128, tau=0.1, d_assoc=64)`:
  `context(pooled[B,2,1152], act[B,16,7]) -> [B,640]`; `prior_logits(ctx) -> [B,M]`;
  `scene_mean(pooled, which) -> [B,32,256]`; `view_embed(v[B,4,32,256], pooled[B,2,1152], which: int) -> [B,256]` (which ∈ {1,2}; encodes `v − scene_mean(pooled)`, pooled = the view's own pre-window; view 2 also takes `w2[B,16]`);
  `hyp_embed(kind[B] long) -> [B,M,256]` (z and event kind only, **no context**: see "Fix 2026-10-27" below); `scores(f[B',256], u[B,M,256]) -> [B,B',M]` (= ⟨u_z, f⟩/τ);
  `predict_post(ctx) -> [B,M,32,256]`; `omega(ctx, sigma2[32,256]) -> [B]`; `summary(ctx) -> [B,24]`; `assoc_embed(pooled) -> [B,64]` (L2-normalised).
- `em_step(model, batch: dict, lam_g=1.0, lam_u=0.1, lam_a=0.1, lam_i=1.0, lam_i1=0.0) -> dict` with keys `loss, prior_nll, nce1, nce2, recon, scene, usage_kl, view_mi, v1_mi, assoc, q` (q detached `[B,M]`); batch keys `pooled, act, v1, v2, w2, has_v2, pooled2, kind` (pooled2 = pre-window of the re-encounter, for the association embedding and the v2 scene residual; kind = index into `events.KINDS`). Hard negatives come from batch construction (Task 10 sampler groups nearest-context events), not from an index tensor. *(done)*
- Methods `model.set_banks(batch)`, `model.log_posterior_v1(pooled, act, v1, kind) -> [B,M]`, `model.loglik_v2(v2, w2, pooled2, kind_j) -> [B,M]` (pooled2 = pre-window of the re-encounter, kind_j = kind of the original event): inference normaliser is a fixed bank of ≤4096 train view embeddings (stored as buffers, saved with the model). *(done)*
- **Fix 2026-10-27 (pilot collapse: q_entropy = log 6, p(fail|z) flat).** (1) `u_z` without context — with `u(c,z)` the score matched each event's own view through its context alone (instance discrimination), z unused. (2) Views are encoded as residuals from `scene_mean(pooled)` (MSE head, stop-grad into the encoder), otherwise context-free clusters split scenes/tasks instead of outcomes. (3) E-step targets for NCE and prior come from the views only (`softmax(ll1 + ll2)`); the prior is distilled from them — fed back, a prior that sees a balanced nuisance (visibility) locks in a split on it. (4) IIC term `−lam_i·I(z1; z2)` between per-view posteriors of re-encountered events. Regression test: `test_em_uses_outcome_when_context_identifies_each_event`.
- **Fix 2026-10-28 (round 1: BatteriesSort broken vs working 0.56, a v1 probe 0.93).** `I(z1; z2)` rewards only what v1 and v2 share; a silent outcome is absent from v2. `lam_i1` adds IIC between two augmentations of v1 (`augment_view`: time-block and token dropout, feature noise); config `lam_i1 = 1.0`. Test `test_v1_iic_finds_an_outcome_absent_from_the_re_encounter` (outcome only in v1's first time block: NMI 0 with lam_i1 = 0, 1.0 with lam_i = lam_i1 = 1 at 900 steps; the observability case still passes; lam_i1 = 5 or 2 breaks it, NMI 0.31).
- **Fix 2026-10-28 (2): conditional on task:kind.** `collate_events` adds `group = data.event_group(task, kind)`; `_nce` normalises over in-batch candidates of the same group, `grouped_mi` = I(z1; z2 | group), banks keep `bank{1,2}_group` and `log_posterior_v1` / `loglik_v2` take an optional `group` (None = whole bank; the Stage B producer does not pass it yet). The bank is now a random sample of train events: `items_tr[:4096]` was ordered by task and held no BatteriesSort event on round 1. Tests: `test_iic_within_task_and_kind_finds_the_outcome_not_the_task` (unconditional: NMI(z, task) = 1, within-task NMI 0), `test_bank_normaliser_uses_the_event_group_only`.

- [x] Tests: shapes; `q` rows sum to 1; synthetic data with 2 hidden outcomes (v1 informative for half the events, v2 always informative) — after 300 CPU steps NMI(argmax q, truth) > 0.8 and mean ω on informative events > on uninformative ones; `usage_kl` → 0 when q uniform over a batch; gradient does not flow into `q`; context that identifies each event (scene signature also in the views) — NMI > 0.8 and low posterior entropy.
- [ ] Implement; run tests.

### Task 5: Memory producers — simple variants

**Files:** Create `src/pol/memories/{__init__,base,simple}.py`; test `tests/pol/test_memories_simple.py`.

**Interfaces — Produces:**
- `@dataclass StepObs(t: int, pooled: np.ndarray[2,1152], tokens: np.ndarray[32,256] | None, proprio: np.ndarray[7], action: np.ndarray[7], event: Event | None, log: dict | None = None)`. *(done; producers call `reset()` in `__init__`; helper `project50(pooled, assets)` uses `assets["proj50"]`, `assets["proj_mean"]`)*
- `class MemoryProducer` (ABC): attrs `name, K`; `reset()`, `observe(o: StepObs)`, `tokens() -> (np.ndarray[K,50], np.ndarray[K] bool)`; classmethod `bytes_per_episode()` = `K*50*4`.
- `make_memory(name: str, cfg: dict, assets: dict) -> MemoryProducer` registry.
- Variants here: `B0` (all masked), `Frames` (last K call frames: PCA-50 projection of pooled, `assets["proj50"]`), `Random` (p=0.1, seeded), `Oracle` (true outcome one-hot from `o.log["event_outcome"]` into the softmax slot), `Markov` (single slot: `assets["markov"]` classifier P(fail | pooled, proprio) at last event re-encounter).

- [ ] Tests: every variant returns `[K,50]` float32 and bool mask; `B0` mask all False; `Frames` keeps the K most recent; `Random` deterministic per seed; `Oracle` reads only `log` and errors if `log is None`; non-Oracle variants never access `o.log` (pass a dict subclass that raises on access).
- [ ] Implement; run tests.

### Task 6: Memory producers — learned baselines and POL

**Files:** Create `src/pol/memories/pol.py`, `src/pol/memories/learned.py` (KEM, MU, AURA, StateMem, CheckVLA/CheckFIFO, AGM), `src/pol/baseline_models.py` (small torch modules + their Stage-A losses); test `tests/pol/test_memories_learned.py`. *(done)*
**Assets each producer reads** (written by Stage A, Task 11): `outcome_model, sigma2, omega0` (POL); `kem_scorer, kem_threshold`; `mu_net`; `aura_net, aura_threshold`; `statemem_net`; `check_net, check_mu, check_sd, conformal_q, still_threshold`; `verify_head`; all use `proj50, proj_mean`.

**Interfaces — Consumes:** Ledger (T3), OutcomeModel (T4), StepObs (T5). **Produces:**
- `POLMemory(cfg, assets)` — on `G+/R+/R0` event: wait until post window filled, then `Entry` from `log_posterior_v1`, `omega`, `summary`, `assoc_embed`; on an event matching pending entries, `resolve(association, loglik_v2)` after its window. Config flags for ablations: `resolve: bool`, `omega: bool`, `M`, `call_frames_only: bool`, `eviction: "entropy"|"fifo"`, `beta`.
- `KEMMemory` (keyframe scorer, top-K buffer), `MUMemory` (GRU over pooled per call, K tokens = linear readout), `AURAMemory` (write when action-predictor surprise > threshold), `StateMemMemory` (1 slot, low-rank residual of prediction error, rank 8), `CheckVLAMemory(fifo_occluded: bool)` (point predictor k=8, standardized discrepancy, conformal threshold α=0.1 from `assets["conformal_q"]`, keyframe bank + risk token), `AGMMemory` (subgoal pointer from `cfg["subgoals"]`, verify head on post-release pooled).
- `baseline_models.py`: `KeyframeScorer`, `RecurrentMemory`, `ActionPredictor`, `FeaturePredictor`, `VerifyHead`, each with `stage_a_loss(batch) -> Tensor`.

- [ ] Tests: each producer satisfies the T5 contract on a random 300-step stream with events; POL with `resolve=False` leaves entries unchanged after re-encounter; POL entry written only after the post window (causality); CheckVLA with `fifo_occluded=True` stores flagged chunks; byte budget equal across variants for the same K.
- [ ] Implement; run tests.

## Phase 1 — Environments and data (sim venv)

### Task 7: Injection wrapper and BatteriesSort update *(done)*

> As built: `jem.envs.injection.FailureSampler`; env ids `BatteriesSort-{3,6}-VLA-v0` (flash, deferred,
> occlusion 0.15, flip 0.05, spill 0.2, held-out stick 0.05); the catch tray has a visual-only cover, so
> the spill pose change happens between two occluded places (socket under the hood, covered tray).
> Injection draws use `env.inj_rng` reseeded from `_episode_seed` (torch RNG was irreproducible per seed).
> Bin check accepts upright batteries (`z_extra = 2.2 * BATTERY_HALF_HEIGHT`). Check: `scripts/pol/check_envs.py`.

**Files:** Create `src/jem/envs/injection.py`; modify `src/jem/envs/batteries_sort.py` (physical spill via release-pose offset into a chute behind the hood, sticking release, `LAMP_MODE="flash"`, `DEFERRED=True`, probabilities from task yaml); test `tests/pol/test_injection.py` (pure-logic parts) + `scripts/pol/check_envs.py` (sim).

**Interfaces — Produces:** `FailureInjector(env, spec: dict, rng)` wrapper with hooks `on_release(env_idx, obj) -> str | None` (returns failure kind), `on_transport(env_idx, obj)`; writes `info["pol_log"]` = `{events_true, failure_kind, outcome}`; env ids `BatteriesSort-{3,6}-VLA-v0` (new names; old EEM ids stay registered).

- [ ] Logic test: sampled failure frequencies within ±2σ of spec over 10k draws; held-out kind never sampled when `split="train"`.
- [ ] Sim check (`python scripts/pol/check_envs.py BatteriesSort-3-VLA-v0`): reset/step 200 random actions, lamp never lit at extraction time in flash mode, spill lands in chute (pose test), no teleport (velocities continuous).

### Task 8: SealedBin, GatherInject and experts *(done)*

> As built: one module `jem/envs/bins.py`; primitives `jem/envs/bin_primitives.py` (new, follows MIKASA's
> GatherAndRecall planner pattern, no code generation); expert `jem/envs/expert_bins.py`.
> SealedBin evidence per drop is history-dependent by design: unseen backwards bounce 0.1 (lands under a
> visual-only canopy), visible sideways bounce 0.1, lid jiggle cue P=0.9 | bounce, 0.1 | clean, shutter
> hides the cue with P=0.2 (unobservable). Bounces are applied 12 steps after release, when the gripper
> is above the lid. The expert opens the lid only if P(all inside) < 0.9.

**Files:** Create `src/jem/envs/sealed_bin.py`, `src/jem/envs/gather_inject.py`, `src/jem/envs/expert_bins.py`; modify `scripts/build_planner_primitives.py` to also emit GatherAndRecall primitives into `src/jem/envs/mikasa_primitives_gather.py`; tests `tests/pol/test_expert_bins.py`.

**Interfaces — Produces:** env ids `SealedBin-{3,5}-VLA-v0`, `GatherInject-{3,5}-VLA-v0`; rewritten `evaluate()` (cube-in-bin by AABB, flash trigger after all placed attempts, button allowed when expert believes all inside); sliding lid actor opened by handle push; back-screen visual occluder. `BinExpert(model)` with `on_release_window(saw_deflect: bool | None)`, `pending()`, `next_action() -> "continue"|"open_lid"|"sweep_behind"|"retrieve"|"confirm"`.
- [ ] Expert tests (pure Python): visible deflect → immediate `retrieve`; invisible (None) → pending until before confirm, then `open_lid`; no pending → `confirm`.
- [ ] Sim check: 20 expert episodes per task succeed ≥ 90 %; budget = P95 × 1.15 written back to task yaml by `scripts/pol/calibrate_budget.py`.

### Task 9: Collector, revealed test, probes *(done)*

> As built: `scripts/pol/collect.py` keeps episodes with `expert_ok` (expert executed its own decisions),
> not only task successes; workers capped by free RAM (2 GB each, 3 GB reserve). `scripts/pol/check_revealed.py`
> runs both the revealed test and the single-frame probe (no separate probe.py). Re-encounter radius is a
> per-task `rho` (socket 0.06, bin 0.12, disc 0.16). Outcome labels: `pol/labels.py`; episode loading and
> events: `pol/episodes.py`; budgets: `scripts/pol/calibrate_budget.py`. Local yield: SealedBin 8/8,
> GatherInject 1/1, BatteriesSort ~2/9 (MIKASA planner failures, as upstream).

**Files:** Create `scripts/pol/collect.py` (generalises `collect_batteriessort.py`, `--task --episodes --seed-start --workers --split {train,eval}`), `scripts/pol/check_revealed.py`, `scripts/pol/probe.py`.
- Output per episode: `<out>/<task>/<seed:06d>.npz` (jem `Episode` schema) + `.log.json` with `pol_log` per step.
- `check_revealed.py --features ... --events ...` → `runs/revealed/<task>.json`: per failure kind AUROC of logistic regression on (pooled at re-encounter, proprio at re-encounter) and verdict `revealed`/`unrevealed` at 0.9.
- [ ] Run on 20 pilot episodes per task locally (sim venv); JSON schema validated by `tests/pol/test_reports.py`.

### Task 10: Features, events, datasets *(done)*

> As built: `pol/features.py` (`SiglipExtractor`, `PCA`, `FeatureStore`), `scripts/pol/extract_siglip.py`
> (partial orbax restore of `PaliGemma/img` straight from gs://, 412M params, ~5 GB RAM, no 11 GB download),
> `scripts/pol/verify_siglip.py` (matches openpi's JAX SigLIP: max |diff| 1.5e-4), `scripts/pol/precompute_features.py`
> (also writes `proj50.npz` and per-task `fsm.json`). No parquet event table: events are recomputed from proprio
> by the FSM (`pol.data.event_tuples`, `stage_a_items`). `precompute_memory` is `pol.data.replay_memory`;
> `StepObs.action` is the action executed at t-1, so memory at call t never sees the target action[t].
> Local throughput on the RTX 3050: ~10-15 frames/s (bulk extraction belongs on the server).
> Policy venv: `scripts/pol/setup_policy_env.sh` (skips `evdev`, `UV_LINK_MODE=copy`, transformers patch).

**Files:** Create `src/pol/features.py`, `src/pol/data.py`, `scripts/pol/precompute_features.py`, `scripts/pol/extract_events.py`; test `tests/pol/test_data.py`.

**Interfaces — Produces:**
- `SiglipExtractor(ckpt_dir, device)`: `__call__(rgb[N,128,128,6] uint8) -> (pooled[N,2,1152] f16, grid[N,32,1152] f16)` (resize-with-pad 224, 16×16 → 4×4 avg-pool).
- `fit_pca(grid_samples, dim=256, seed=0) -> PCA` (`.transform`, `.save/.load`).
- `FeatureStore(root)`: `pooled(task, seed) -> memmap[T,2,1152]`, `tokens(task, seed) -> dict[int, np.ndarray[32,256]]` (only stored frames).
- `EventTable` parquet columns: `task, seed, j, t, kind, x, y, z, k_reenc, t_reenc`.
- `StageADataset(store, events, split)` → batch dict for `em_step`; `hard_negatives(ctx_emb, kinds, k=32)`.
- `precompute_memory(producer, episode, store) -> np.ndarray[n_calls,K,50], mask` — replays steps 0..T−1, emits tokens at t mod 8 == 0 **after** observing t.
- `StageBDataset(episodes, memory_npz, norm, H=16)` → `{"image": {"base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"}, "image_mask", "state"[32], "actions"[16,32], "memory"[K,50], "memory_mask"[K], "prompt"}` (pad 7→32 with zeros, matching openpi conventions).
- [ ] Tests: causality — perturbing frames > t leaves memory at call t unchanged (Review Focus 5); action pad/unpad round trip; split by seed has no overlap; `hard_negatives` excludes self.

## Phase 2 — Training (policy venv)

### Task 11: Stage A training and gates

**Files:** Create `src/pol/train_stage_a.py` (CLI `python -m pol.train_stage_a --config configs/pol/stage_a.yaml --variant POL --out runs/stage_a/POL`), `src/pol/gates.py` (CLI writes `runs/gates.json`), `src/pol/wandb_utils.py`.
- `wandb_utils.init_run(stage: str, variant: str, cfg: dict, prov: dict, K: int, seed: int)` → run name `{stage}-{variant}-K{K}-s{seed}`, group `{stage}/{variant}`; `log(step, metrics)` every 20 steps.
- Stage A trains POL OutcomeModel or the baseline modules of T6 (same loop, `stage_a_loss`); saves `model.pt`, `sigma2.npy`, `assets.json`; val every 1k: val NCE, and eval-only NMI/ω-AUROC from logs.
- `gates.py` implements spec §10.1 thresholds exactly; exit code 0 = pass, 2 = fail N1/G1, 3 = fail G2 only.
  *Fix 2026-10-28:* scored on `labels.GATE_TARGET` (broken vs working battery, behind the bin vs in; any failure on
  GatherInject) instead of "failed"; the Markov head is trained on the same target; G1 control = shuffled
  evidence (`gates.shuffle_evidence`), fake actions reported only; group scores = mean of per-task AUROCs (`per_task`; pooled kept under `pooled`); P(target | z) calibrated per task on train events with a re-encounter.
- [ ] Tests (`tests/pol/test_gates.py`): synthetic scores reproduce each pass/fail branch; W&B offline smoke (`WANDB_MODE=disabled`) for 50 steps on synthetic data.

### Task 12: π0 with memory tokens

**Files:** Create `src/pol/pi0_memory.py`; test `tests/pol/test_pi0_memory.py` (policy venv, openpi `dummy` gemma variants, CPU).

**Interfaces — Produces:**
- `memory_suffix_att(K: int, has_state: bool, H: int) -> list[int]` = `[1]+[0]*(K-1)` + `[1]` (state) + `[1]+[0]*(H-1)`.
- `class PI0WithMemory(PI0Pytorch)(config, K: int, entry_dim: int = 50)`: `embed_suffix(state, x_t, t, memory, memory_mask)`, `forward(observation, actions, memory, memory_mask, noise=None, time=None) -> Tensor[B,H,32]`, `sample_actions(device, observation, memory, memory_mask, noise=None, num_steps=10)`, `denoise_step(..., memory, memory_mask)`; `trainable_parameters() -> list[nn.Parameter]` after `freeze_backbone()`.
- `load_pi0_with_memory(ckpt_dir, K, action_horizon=16) -> PI0WithMemory` (loads `model.safetensors`, `strict=False` only for the new memory MLP keys — assert no other missing keys).
- [ ] Tests: with all-False mask, output equals base `PI0Pytorch` output on the same weights/noise (memory is a no-op when empty — Review Focus 1); attention: prefix tokens cannot attend memory (check 2D mask); gradients reach memory MLP and expert, none reach paligemma; `sample_actions` shape `[B,16,32]`.

### Task 13: Stage B training loop

**Files:** Create `src/pol/train_stage_b.py` (CLI `--config configs/pol/stage_b.yaml --variant POL --seed 17 --memory runs/memory/POL-K8 --out runs/stage_b/POL-K8-s17 [--resume]`), `src/pol/optim.py`.
- `optim.MasterWeightAdamW(params_groups, betas, eps, wd)` — fp32 master copies, `step()` copies back to bf16.
- Loop: grad checkpointing on; cosine LR with warmup per group; val flow loss every 2k on val split; save trainable state every 5k + at {10k,15k,20k}; `best.json` picks the lowest val among those; W&B metrics `loss, grad_norm, lr/*, step_time, samples_per_s, vram_gb, val/loss`; camera image at step 0; resume restores model/optimizer/RNG/step.
- [ ] Tests: `MasterWeightAdamW` on a bf16 toy model matches fp32 AdamW within 1e-3 after 100 steps; resume test on a dummy-gemma model (CPU): 10 steps, stop at 5 and resume, identical loss sequence.

## Phase 3 — Serving, evaluation, reporting

### Task 14: Policy server and protocol

**Files:** Create `src/pol/serve.py` (policy venv), `src/pol/protocol.py` (shared, numpy only), `src/pol/eval_client.py` (sim venv).
- `protocol.Request` keys: `episode_id, reset, images{top,wrist}[128,128,3] uint8, state[7], prompt, chunk_frames[n,2,128,128,3], chunk_proprio[n,7], chunk_t0`; `Response`: `actions[16,7], memory_debug{posteriors, omega, status}, latency_ms`.
- `MemoryPolicy(model, producer_factory, extractor, pca, norm).infer(req) -> resp` — resets state when `reset` or `episode_id` changes (Review Focus 4); serves via `openpi.serving.websocket_policy_server.WebsocketPolicyServer`.
- [ ] Tests: round trip with a fake model; episode_id change resets; latency field present.

### Task 15: Evaluation runner and metrics

**Files:** Create `scripts/pol/run_eval.py` (sim venv: `--task --variant --train-seed --episodes 50 --workers 8 --host --port --out runs/eval/<variant>-K<K>-s<seed>`), `src/pol/metrics.py`, `scripts/pol/make_cma_pairs.py`.
- Record per episode (JSON lines): `kind="model_evaluation", task, variant, training_seed, episode_seed, success_once, stratum, rechecks, unnecessary_rechecks, failures[{kind, t, recovered, t_recover}], events[{t, outcome_true, posterior, omega, observable_true}], latency_ms, provenance`.
- `metrics.py`: `success_rate`, `recovery_rate`, `auroc(scores, labels)`, `nmi(a, b)`, `ece(p, y, bins=10)`, `brier`, `cma_switch_rate(pairs)`; reuse `jem.evaluation.paired_report` with `SHARED_PROVENANCE` extended by `feature_hash, stage_a_hash, memory_bytes`.
- W&B: `wandb.Table` per episode, 3 videos per task, summary SR + CI.
- [ ] Tests: metrics against sklearn on random data; paired report refuses mismatched `memory_bytes`.

### Task 16: Report into the paper

**Files:** Create `src/pol/report.py` (CLI `--runs runs --tex ../../latex-paper/eem-paper/results.tex --fig ../../latex-paper/eem-paper/figures/capacity.pdf`); replace `latex-paper/eem-paper/fill_results.py` with a 5-line shim calling it.
- Writes every key listed in `latex-paper/eem-paper/README.md`; task codes `bs3, bs6, sb5, inj (= mean GI3,GI5), cnt`; `unrev` = macro over bs3, bs6, sb5; `rev` = inj.
- [ ] Test: synthetic records → `results.tex` contains `\setres{sr.POL.bs3}{0.80}` etc.; missing data keeps `[?]`.

### Task 17: RoboMME-Counting

**Files:** Create `scripts/pol/convert_robomme.py` (LeRobot v3 → jem `Episode`, 256² resized to 128² for features only; actions 8D), `configs/pol/tasks/RoboMME-{BinFill,PickXTimes,SwingXTimes}.yaml`, `src/pol/robomme_adapter.py` (maps their eval client observation dict → `protocol.Request`), `scripts/pol/setup_robomme_env.sh`.
- [ ] Test: adapter converts a recorded sample dict; converter round-trips one parquet episode (frame count and timestamps match metadata).

### Task 18: Setup, download, orchestration, runbook

**Files:** Create `scripts/pol/{setup_sim_env.sh,setup_policy_env.sh,download_assets.sh,run_all.sh}`, `docs/RUNBOOK-POL.md`.
- `setup_policy_env.sh`: uv venv py3.11, openpi at pinned commit, `transformers==4.53.2` + copy `transformers_replace`, `pip install -e .[pol]`; `download_assets.sh`: `pi0_base` via `gsutil`/openpi downloader → `convert_jax_model_to_pytorch.py` → `checkpoints/pi0_base_pytorch`, hashes to `runs/assets-manifest.json`.
- `run_all.sh <phase>` with phases `collect | features | stage_a | gates | memory | stage_b | eval | report`; phase `stage_b` launches two runs at a time (`CUDA_VISIBLE_DEVICES=0` / `1`) from a queue file `configs/pol/queue_p0.txt`.
- [ ] Dry-run `bash scripts/pol/run_all.sh --dry-run all` prints the full command list without executing.

### Task 19: Paper sync (spec §12b)

**Files:** Modify `latex-paper/eem-paper/main.tex`, `README.md`.
- [ ] Apply the five edits of spec §12b and the POMDP-distinction paragraph; rebuild with `./build.sh`; zero undefined references.

---

## Execution notes
- Order: T1→T6 local CPU (this machine), T7–T9 need the sim venv (local RTX 3050 can run ManiSkill), T10–T13 policy venv (T12/T13 tests use openpi `dummy` variants on CPU), T14–T18, T19 last.
- `Ledger.evict_index` (T3) is written by the user (learning mode); its tests define the contract.
