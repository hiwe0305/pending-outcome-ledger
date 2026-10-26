"""Collect U-Suite demonstrations with the belief-experts (sim venv).

    python scripts/pol/collect.py --task SealedBin-3 --episodes 500 --seed-start 0 --split train \
        --out data/pol --workers 8

--task is a configs/pol/tasks/<name>.yaml stem. Each kept episode is written as
<out>/<task>/<seed:06d>.npz (jem.data.Episode schema: rgb [T,128,128,6], proprio [T,7], action [T,7]
normalised ee-delta, timestamp, language) and <seed:06d>.log.json (evaluator-only per-step log,
expert evidence and decisions). Episodes are kept when the expert carried out its own decisions
(`expert_ok`), not only when the task succeeded: a Bayes-optimal expert still fails when the evidence
misleads it, and filtering those out would bias the data towards lucky outcomes. Motion-planner
failures are dropped and retried with the next seed, as MIKASA's collector does.

Workers are capped by free RAM (one sim process peaks at ~1.75 GB; 3 GB are kept free) and by free
GPU memory (SAPIEN renders on the GPU; 0.5 GB per worker, 1 GB kept free for other jobs on the card).
"""
import argparse, json, multiprocessing as mp, os, sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
GB_PER_WORKER, GB_RESERVE = 2.0, 3.0
VRAM_PER_WORKER, VRAM_RESERVE = 0.5, 1.0


def available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def free_vram_gb():
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def safe_workers(requested):
    by_ram = int((available_gb() - GB_RESERVE) // GB_PER_WORKER)
    by_vram = int((free_vram_gb() - VRAM_RESERVE) // VRAM_PER_WORKER)
    return max(1, min(requested, by_ram, by_vram))


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class Recorder:
    """Wraps the planner-facing env: VLA observation before each step, normalised ee-delta action, log."""

    def __init__(self, env, log_key, beliefs_ref=None):
        self.env, self.log_key, self.beliefs_ref = env, log_key, beliefs_ref if beliefs_ref is not None else []
        cfg = env.unwrapped.agent._controller_configs["pd_ee_delta_pose"]["arm"]
        self.pos_scale = float(np.max(np.abs([cfg.pos_lower, cfg.pos_upper])))
        self.rot_scale = float(np.max(np.abs([cfg.rot_lower, cfg.rot_upper])))
        self.dt = 1.0 / float(env.unwrapped.control_freq)
        self.mode, self.rows, self.last_obs = 0, [], None
        self.planner_failed = False

    def __getattr__(self, k):
        return getattr(self.env, k)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.rows, self.last_obs = [], obs
        return obs, info

    def step(self, action):
        from transforms3d.euler import quat2euler
        from transforms3d.quaternions import qinverse, qmult

        u = self.env.unwrapped
        prev = _np(u.agent.tcp.pose.raw_pose[0]).copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        nxt = _np(u.agent.tcp.pose.raw_pose[0]).copy()
        dq = qmult(nxt[3:7], qinverse(prev[3:7]))
        delta = np.r_[(nxt[:3] - prev[:3]) / self.pos_scale, np.array(quat2euler(dq)) / self.rot_scale]
        grip = float(np.asarray(action).reshape(-1)[-1])
        log = {k: _np(v)[0].tolist() for k, v in info[self.log_key].items()}
        self.rows.append({"rgb": _np(self.last_obs["rgb"])[0], "proprio": _np(self.last_obs["proprio"])[0],
                          "action": np.clip(np.r_[delta, grip], -1, 1).astype(np.float32), "mode": self.mode,
                          "evidence": [getattr(b, "evidence", None) for b in self.beliefs_ref], "log": log})
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------ bins
def run_bin_episode(env, prim, sealed, rng):
    from jem.envs.bins import AT_TARGET, FAILURE_CODES, VISIBLE_OUT
    from jem.envs.expert_bins import BinBelief, BinModel, decide_after_release, decide_before_confirm

    u = env.unwrapped
    n = int(u.N_CUBES)
    f = u.failures
    model = (BinModel(side=f.p("deflect_side"), behind=f.p("deflect_behind"), cue_hit=f.p("cue_hit"),
                      cue_false=f.p("cue_false")) if sealed else BinModel(side=0.0, behind=0.0, cue_hit=0.0, cue_false=0.0))
    beliefs = [BinBelief(model) for _ in range(n)]
    targets = prim.place_targets()
    decisions, lid_open = [], False

    def evidence(start):
        logs = [r["log"] for r in env.rows[start:]]
        visible = VISIBLE_OUT + ((FAILURE_CODES["deflect_behind"],) if lid_open else ())  # open lid: the exit is seen
        if any(l["failure"] in visible for l in logs):
            return "seen_out"
        if any(l.get("shutter") for l in logs):
            return "occluded"
        return "cue" if any(l.get("lid_jiggle") for l in logs) else "no_cue"

    def place_cube(i, attempts=3):
        for _ in range(attempts):
            if prim.done["value"]:
                return
            start = len(env.rows)
            prim.pick_cube(i)
            prim.place_at(*targets[i])
            prim.hold(int(getattr(u, "BOUNCE_DELAY", 0)) + 4)
            ev = evidence(start)
            beliefs[i].observe_release(ev, lid_open=lid_open)
            decisions.append({"cube": i, "evidence": ev, "p_out": beliefs[i].p_out, "t": len(env.rows)})
            if decide_after_release(beliefs[i]) != "retrieve":
                return

    for i in rng.permutation(n):
        place_cube(int(i))
    choice = decide_before_confirm(beliefs) if sealed else "confirm"
    decisions.append({"before_confirm": choice, "p_out": [b.p_out for b in beliefs], "t": len(env.rows)})
    if choice == "open_lid" and not prim.done["value"]:
        env.mode = 1
        hx, hy = _np(u.handle_xy[0]).tolist()
        prim.press(hx, hy, 0.04)
        lid_open = bool(_np(u.lid_open)[0])
        cx, cy = _np(u.disc_xy[0]).tolist()
        prim.look_at(cx, cy)
        loc = env.rows[-1]["log"]["cube_location"] if env.rows else [AT_TARGET] * n
        for i in range(n):
            beliefs[i].observe_inside(loc[i] == AT_TARGET)
        missing = [i for i in range(n) if loc[i] != AT_TARGET]
        if missing:
            bx, by = cx + u.BEHIND_OFFSET[0], cy + u.BEHIND_OFFSET[1]
            prim.look_at(bx, by)  # the wrist camera looks under the canopy
            for i in missing:
                place_cube(i)
        env.mode = 0
    pressed = False
    if not prim.done["value"]:
        b = int(_np(u.flash_color)[0])
        bx, by = prim.buttons_xy()[b]
        prim.press(bx, by, prim.button_top_z)
        pressed = not prim.done["value"]
    info = u.evaluate()
    return {"success": bool(_np(info["success"])[0]), "expert_ok": pressed, "decisions": decisions}


# ------------------------------------------------------------- batteries
def run_battery_episode(env, prim, rng, max_retests=2):
    from collect_batteriessort import run_episode
    from jem.envs.expert import ObservationModel

    u = env.unwrapped
    model = ObservationModel(spill=u.SPILL_PROB, flip=u.FLIP_PROB, occlusion=u.OCCLUSION_PROB, p_working=u.P_WORKING)
    info = run_episode(env, prim, model, u.DEFERRED, u.LAMP_MODE == "persistent", max_retests, rng)
    n = int(_np(u.active_battery_count)[0])
    checked = int(_np(u.checked_count)[0]) >= n
    pos, _, grasp = u._stack_battery_poses()
    at_slot = (np.linalg.norm(_np(pos[0, :n, :2] - u.tray_slot_positions[0, :n, :2]), axis=-1) <= u.SLOT_RETURN_XY_TOL)
    in_bin = _np(u._in_box(pos, u.bin_pos, u.BIN_HALF_SIZE, z_extra=2.2 * u.BATTERY_HALF_HEIGHT)[0, :n])
    intended_home = [b.p_working >= 0.5 for b in env.beliefs_ref]
    placed_as_intended = all((at_slot[i] if intended_home[i] else in_bin[i]) for i in range(n))
    diag = {"checked_count": int(_np(u.checked_count)[0]), "n": n, "at_slot": at_slot.tolist(), "in_bin": in_bin.tolist(),
            "intended_home": intended_home, "held": _np(grasp[0, :n]).tolist(), "stage": int(_np(u.action_stage)[0]),
            "pos": np.round(_np(pos[0, :n]), 3).tolist(), "steps": len(env.rows), "budget_hit": bool(prim_done(prim))}
    diag["planner_failed"] = bool(env.planner_failed)
    return {"success": bool(_np(info["success"])[0]), "expert_ok": checked and placed_as_intended and not env.planner_failed, "diag": diag,
            "decisions": [{"battery": i, "tests": b.tests, "p_working": b.p_working} for i, b in enumerate(env.beliefs_ref)]}


def prim_done(prim):
    d = getattr(prim, "episode_done", None) or getattr(prim, "done", None)
    return bool(d["value"]) if isinstance(d, dict) else False


# ---------------------------------------------------------------- worker
def episode(task_cfg, seed, split, out, keep_all=False):
    warnings.filterwarnings("ignore")
    import gymnasium as gym
    import jem.envs.batteries_sort as bs
    import jem.envs.bins as bins
    from jem.envs.bin_primitives import make_bin_primitives
    from jem.envs.mikasa_primitives import IgnoreSuccessTerminationWrapper, make_primitives
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    bs.register_wrappers()
    bins.register_wrappers()
    env_id, family = task_cfg["env_id"], task_cfg["family"]
    steps = int(task_cfg.get("max_steps") or gym.spec(env_id).max_episode_steps)
    env = gym.make(env_id, num_envs=1, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="normalized_dense", split=split, max_episode_steps=steps)
    env = IgnoreSuccessTerminationWrapper(apply_mikasa_vla_wrappers(env, include_overlays=False))
    rng = np.random.default_rng(seed)
    beliefs = []
    rec = Recorder(env, "eem_log" if family == "batteries_sort" else "pol_log", beliefs)
    rec.reset(seed=seed)
    if family == "batteries_sort":
        rec.beliefs_ref = beliefs
        result = run_battery_episode(rec, make_primitives(rec, steps), rng)
    else:
        result = run_bin_episode(rec, make_bin_primitives(rec, steps), family == "sealed_bin", rng)
    u = rec.unwrapped
    rows = rec.rows
    kept = bool(result["expert_ok"]) and len(rows) > 0
    if kept or keep_all:
        d = Path(out) / task_cfg["name"]
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{seed:06d}.npz", episode_id=f"{task_cfg['name']}:{seed:06d}", language=u.LANGUAGE_INSTRUCTION,
                            rgb=np.stack([r["rgb"] for r in rows]).astype(np.uint8),
                            proprio=np.stack([r["proprio"] for r in rows]).astype(np.float32),
                            action=np.stack([r["action"] for r in rows]).astype(np.float32),
                            timestamp=(np.arange(len(rows)) * rec.dt).astype(np.float32))
        (d / f"{seed:06d}.log.json").write_text(json.dumps(
            {"seed": seed, "split": split, "task": task_cfg["name"], "env_id": env_id, "steps": len(rows),
             "success": result["success"], "expert_ok": bool(result["expert_ok"]), "decisions": result["decisions"],
             "diag": result.get("diag"),
             "modes": [r["mode"] for r in rows], "logs": [r["log"] for r in rows]}, default=_jsonable))
    env.close()
    return {"seed": seed, "steps": len(rows), "success": result["success"], "kept": kept, "diag": result.get("diag")}


def worker(args):
    task_cfg, seeds, quota, split, out = args
    done, results = 0, []
    for s in seeds:
        if done >= quota:
            break
        try:
            r = episode(task_cfg, s, split, out)
        except Exception as exc:  # motion-planner errors (e.g. mplib "Fail to parameterize path")
            r = {"seed": s, "steps": 0, "success": False, "kept": False, "diag": {"planner_failed": True, "error": repr(exc)[:200]}}
        results.append(r)
        done += int(r["kept"])
        print(json.dumps(r, default=_jsonable), flush=True)
    return results


def main():
    from pol.config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=None, help="kept episodes (default: task yaml)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--out", default=str(ROOT / "data/pol"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-attempt-factor", type=int, default=3)
    a = p.parse_args()
    cfg = load_config(ROOT / f"configs/pol/tasks/{a.task}.yaml")
    cfg["name"] = a.task
    total = a.episodes or int(cfg["episodes"])
    w = safe_workers(a.workers)
    if w < a.workers:
        print(f"[collect] limited to {w} worker(s) ({available_gb():.1f} GB RAM, {free_vram_gb():.1f} GB VRAM free)", flush=True)
    attempts = total * a.max_attempt_factor
    seeds = list(range(a.seed_start, a.seed_start + attempts))
    jobs = [(cfg, seeds[k::w], total // w + (k < total % w), a.split, a.out) for k in range(w)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(w) as pool:
        results = [r for rs in pool.map(worker, jobs) for r in rs]
    d = Path(a.out) / a.task
    d.mkdir(parents=True, exist_ok=True)
    kept = [r for r in results if r["kept"]]
    (d / f"summary-{a.split}-{a.seed_start}.json").write_text(json.dumps(
        {"task": a.task, "split": a.split, "attempted": len(results), "kept": len(kept),
         "expert_success_rate": float(np.mean([r["success"] for r in kept])) if kept else None,
         "steps_p95": float(np.percentile([r["steps"] for r in kept], 95)) if kept else None,
         "episodes": results}, indent=1, default=_jsonable))
    print(f"[collect] {a.task}: kept {len(kept)}/{len(results)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
