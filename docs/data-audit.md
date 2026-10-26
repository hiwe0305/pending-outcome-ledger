# Audit of the released MIKASA BatteriesChecker data (negative-control tasks)

Scope note (2026-10-26): this audit covers the released MIKASA LeRobot data used for the
negative control and normalizer fitting. EEM training data come from the BatteriesSort
collector (docs/RUNBOOK.md, stage 2).

> Correction (2026-10-26): the source metadata episode-length order does not
> match Parquet episode-index order for any of the four BatteriesChecker tasks,
> although their length multisets match. Index-based sample seed associations
> below are unverified and must not be used for replay or label generation.
> See [task suitability audit](joint-task-suitability-2026-10-26.vi.md).

Audit date: 2026-10-14. Main specification: PROPOSAL.md revision 2. No original
research documents or Paper Desk code were modified. No applicable AGENTS.md
was found in workspace ancestors/code. No secrets were read. Machine-readable
pins: `configs/manifest.json`; selected source hashes: `configs/source-hashes.json`.

## Exact versions

| Source | Revision |
|---|---|
| CognitiveAISystems/MIKASA-Robo, vla | `3d5a21a99e5df2bcc34c612040b269e394b09438` |
| moojink/openvla-oft, main | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| moojink/transformers-openvla-oft | `bc339d9ad707454c0c115970db43c260067c61ab` |
| mikasa-robo/mikasa-robo-vla-lerobot | `fa5417a266d1cb87ed7715c3dd2d0e4edc067b04` |
| Candidate openvla/openvla-7b | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| shihao1895/MemoryVLA, openvla-codebase | `d732ea9072bc063399ccc817aed74ab172eb50be` |

Checkpoint config/card downloaded, **weights not downloaded**. Candidate is
original OpenVLA initialization, not a MIKASA-trained policy. Final initialization
is not locked until actual compatibility/license audit. OFT/dataset/MIKASA MIT
license records saved; original OpenVLA card identifies MIT and Llama-2 lineage;
upstream license review remains. MemoryVLA LICENSE at root returned 404; do not
infer that no license exists. E1 adaptation has not begun.

## Gates

| Gate | Status | Evidence / limitation |
|---|---|---|
| G1: one episode per task | PASS within sample scope | Real parquet + both decoded videos; visual contact sheet reviewed; arrays/time alignment checked |
| G2: official simulator/action interface | PARTIAL / NOT VERIFIED | Exact runner helpers pass fixture tests; simulator package/assets unavailable locally |
| G3: trustworthy L/Q/E targets | NOT VERIFIED | No such columns in released parquet; source seeds exist but replay/video-label alignment unverified |
| OFT memory injection | PARTIAL | Exact pinned prompt/prefix helpers and gradient/offset tests pass; real 7B forward pending |
| Resource profile | PARTIAL | CPU fixture counts/time/RSS measured; CUDA driver unavailable, no 7B VRAM measurements |

G1 does not certify all 1000 videos or recovered ground-truth state. G2 fixture
execution is not simulator rollout. G3 has not passed and supervised real memory
training fails closed. Test/val data have not been used to tune any model.

## Actual samples

Every task has 250 episodes in metadata. Four small numeric parquet shards
were downloaded (about 72 MB total); no full video shard was downloaded.
Range-based AV1 decoder read only byte ranges required for episode 0. Saved
samples are RGB uint8 `[T,128,128,6]`, proprio/actions float32 `[T,7]`, timestamps,
language and non-runtime episode ID. They are real demonstration samples.

| Task | Folder | Episode 0 length | Source seed |
|---|---|---:|---:|
| BatteriesCheckerEasy-3-VLA-v0 | batteries_checker_easy_3_vla_v0 | 362 | 9 |
| BatteriesCheckerEasy-6-VLA-v0 | batteries_checker_easy_6_vla_v0 | 731 | 10 |
| BatteriesCheckerHard-3-VLA-v0 | batteries_checker_hard_3_vla_v0 | 684 | 1 |
| BatteriesCheckerHard-6-VLA-v0 | batteries_checker_hard_6_vla_v0 | 1425 | 0 |

Metadata paths: each folder's `meta/info.json`, `meta/tasks.parquet`,
`meta/episodes/chunk-000/file-000.parquet`, `source_rlds_metadata.json`.
Language is stored as pandas index `__index_level_0__` in tasks.parquet.
Episode video paths/offsets use LeRobot v3 metadata (not one video per episode).
`runs/real-sample-extraction.json` records frame counts, byte budgets, paths and
PTS errors. Largest video/parquet timestamp error is `6.103515630684342e-06` s;
FPS=10. All decoded frame counts match episode lengths. Contact sheet
`runs/sample-contact-sheet.png` shows distinct top/wrist views, tray, socket,
button and lamp with occlusion and small lamp appearance; no image-derived
outcome labels were assigned.

Numeric parquet columns are exactly observation.state, action, timestamp,
frame_index, episode_index, index, task_index. Task indices are local 0, so
folder mapping is mandatory. Sources include success_once/reward_sums and
**episode_seeds** in source metadata; these remain evaluator/replay-only.
No simulator snapshots or per-frame battery IDs/L/Q/E were found. Source
collector commit is not recorded in this metadata; seed alone cannot prove
that replay on today's pinned simulator reconstructs collection states.

Episode-rank split at seed17 yields 200/25/25 per task (800/100/100 total),
saved in `configs/episode-splits.json`. Both numeric scalers fit exclusively
800 train episode IDs; original dataset-wide stats are never reused. Numeric
shards include all 250 trajectories per task; only four episodes have local RGB.

## Source evidence for interface and task semantics

All following paths are relative to `third_party/mikasa/` at the exact SHA above:

- `mikasa_robo_vla_envs.csv:45–48`: four IDs, language, horizons 540/1080/1080/2160.
- `docs/source/observation_space.rst:49–75`: top/base then hand/wrist camera packing.
- `docs/source/observation_space.rst:93–185`: absolute TCP xyz metres, quaternion
  converted to roll/pitch/yaw radians, sum of finger qpos metres `[0,.08]`;
  normalized action xyz delta scale .1 m, rotation **vector** norm limit .1 rad,
  gripper -1 closed / +1 open. Action rotation is not proprio Euler angles.
- `mikasa_robo_suite/vla/utils/wrappers.py:193–249`: quaternion->Euler, finger sum.
- `mikasa_robo_suite/vla/utils/apply_wrappers.py:1–35`: canonical wrapper order;
  privileged fields stripped and debug overlays disabled.
- `mikasa_robo_suite/vla/memory_envs/batteries_checker_hard_vla.py:100–117`:
  insertion/return tolerances, button hysteresis constants, lamp afterglow=7.
- Same Hard file `:502–552`: released insertion, stage transition, working_mask
  changes lamp appearance; inserted battery snaps to socket center.
- Same Hard file `:564–607`: TCP/button depth press-ready hysteresis; released
  battery near its original slot advances to confirmation.
- Same Hard file `:609–653`: button only confirms correct stage; all active
  batteries must be checked (including negative ones), checked batteries must
  return home, working discoveries counted. No explicit "robot saw lamp" flag.
- Same Hard file `:705–717`: success terminates episode.
- `.../batteries_checker_easy_vla.py:38–76`: insertion triggers immediate automatic
  return and confirm stage. Insert and return can occur in the same evaluate call.
- `mikasa_robo_suite/vla/benchmarking.py:26–33`: 50 episodes/task, seeds
  4242424242..4242424291, normalized controller, official wrappers.
- Same benchmarking file `:430–512`: queue-only policy calls, reset queue, OR-latch
  success_once, termination/truncation, batch-one action conversion.

Do not use working_mask as Q ground truth. A valid negative test requires
verified insertion + visible result + identity/history alignment. Lamp afterglow
can survive removal; absence of light with no valid test is unknown, not negative.
Q memory does not replace button-confirm/check predicates in the evaluator.

## Environment and model blockers

Default Python3.12 has no torch; reused a local conda Python with torch
(Python3.10.22, torch2.10.0+cu128) for CPU tests. CUDA unavailable, nvidia-smi exit9.
ManiSkill/SAPIEN and installed MIKASA package absent; official-dummy invocation
fails `ModuleNotFoundError: mikasa_robo_suite` before simulator rollout. This
attempt is logged in `runs/official-dummy-attempt.txt`.

OFT source constants default to LIBERO proprio8 if invoked under an unrecognized
command (`third_party/oft/prismatic/vla/constants.py:39–95`); adapter explicitly
sets MIKASA proprio7 and chunk8. OFT `pyproject.toml:50` requires a bidirectional
Transformers fork. Installed stock Transformers is not accepted as OFT. OFT
pins torch2.2.0 while MIKASA pins2.2.1; dependency co-installation is **not verified**.
Resolve one common tested environment on the future machine, record any consistent
patch-version adjustment, and retain the same environment for all variants.

Local sandbox DNS failed; permitted escalated public-source access succeeded.
Downloads had bounded file/range sizes; no credentials or secret files read.
No GPU rental, full dataset fetch, checkpoint fetch, 7B training or paid service.
