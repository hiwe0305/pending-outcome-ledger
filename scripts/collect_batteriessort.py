"""Collect BatteriesSort demonstrations with the belief-expert.

UNTESTED IN THIS REPOSITORY (needs mani_skill/sapien/mplib + pinned MIKASA). First run
`python scripts/check_batteries_sort.py` on the server, then e.g.

    python scripts/collect_batteriessort.py --task BatteriesSortHard-3-VLA-v0 \
        --episodes 250 --seed-start 0 --out data/batteriessort

Motion comes from the upstream MIKASA planner primitives (src/jem/envs/mikasa_primitives.py,
generated). Decisions come from jem.envs.expert using only policy-observable evidence.
Each episode is written as <out>/<task>/<episode>.npz (jem.data.Episode schema) plus
<episode>.labels.json (per-call labels, schema eem-env-log-v1) and <episode>.log.json
(evaluator-only event log for strata).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jem.envs.expert import BatteryBelief, ObservationModel, choose_verify, decide  # noqa: E402

CHUNK = 8


class Recorder:
    """Wraps the planner-facing env; records VLA observations, normalized ee-delta actions, logs."""

    def __init__(self, env, beliefs_ref):
        import gymnasium as gym  # noqa: F401
        self.env = env
        self.beliefs_ref = beliefs_ref
        self.mode = 0
        self.rows = []
        cfg = env.unwrapped.agent._controller_configs["pd_ee_delta_pose"]["arm"]
        self.pos_scale = float(np.max(np.abs([cfg.pos_lower, cfg.pos_upper])))
        self.rot_scale = float(np.max(np.abs([cfg.rot_lower, cfg.rot_upper])))
        self.dt = 1.0 / float(env.unwrapped.control_freq)
        self.last_obs = None

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
        prev = u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy().copy()
        pre_obs = self.last_obs
        obs, reward, terminated, truncated, info = self.env.step(action)
        nxt = u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy().copy()
        dq = qmult(nxt[3:7], qinverse(prev[3:7]))
        delta = np.r_[(nxt[:3] - prev[:3]) / self.pos_scale, np.array(quat2euler(dq)) / self.rot_scale]
        grip = float(np.asarray(action).reshape(-1)[-1])
        log = {k: np.asarray(v.detach().cpu() if hasattr(v, "detach") else v)[0].tolist() for k, v in info["eem_log"].items()}
        self.rows.append({
            "rgb": np.asarray(pre_obs["rgb"].detach().cpu() if hasattr(pre_obs["rgb"], "detach") else pre_obs["rgb"])[0],
            "proprio": np.asarray(pre_obs["proprio"].detach().cpu() if hasattr(pre_obs["proprio"], "detach") else pre_obs["proprio"])[0],
            "action": np.clip(np.r_[delta, grip], -1, 1).astype(np.float32),
            "mode": self.mode,
            "evidence": [b.evidence for b in self.beliefs_ref],
            "log": log,
        })
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


def lamp_reading(rows, start, persistent):
    """What the cameras showed after an insert: None if the shutter was down, else lamp on/off."""
    window = rows[start:]
    if not window or any(r["log"]["lamp_occluded"] for r in window[:2]):
        return None
    if persistent:
        return bool(window[-1]["log"]["lamp_on"])
    return any(bool(r["log"]["lamp_on"]) for r in window)


def run_episode(env, prim, model, deferred, persistent, max_retests, rng):
    u = env.unwrapped
    n = int(u.active_battery_count[0].item())
    beliefs = env.beliefs_ref
    beliefs.clear()
    beliefs.extend(BatteryBelief(model) for _ in range(n))
    order = rng.permutation(n)  # visiting order is not informative of quality
    home = prim.tray_slots
    bin_xy = u.bin_pos[0].detach().cpu().numpy()
    catch = u.catch_pos[0].detach().cpu().numpy()

    def insert_once():
        x, y = float(prim.socket_pos[0]), float(prim.socket_pos[1])
        prim.move_via_clearance(x, y, prim.z_socket_approach, prim.z_transport)
        prim.move_pose_xyz(x, y, prim.z_insert_socket)
        prim.open_gripper_safe(t=2)
        prim.hold_steps(2)
        prim.move_pose_xyz(x, y, prim.z_socket_approach)

    def test(i):
        start = len(env.rows)
        insert_once()
        reading = lamp_reading(env.rows, start, persistent)
        held = prim.try_extract_from_socket(i)
        if not held:  # spilled: the lamp stayed off (a non-event); fetch it from the catch tray
            prim.try_pick_from_pose(i, float(catch[0]), float(catch[1]), float(catch[2]) + 0.02)
        beliefs[i].observe(held, reading)

    def place(i, xy):
        if prim.try_return_to_home(i, float(xy[0]), float(xy[1])):
            prim.press_confirm_button()
            return True
        return False

    for i in order:
        env.mode = 0
        if not prim.try_pick_from_slot(i, float(home[i][0]), float(home[i][1])):
            continue
        test(i)
        retests = max_retests
        while True:
            choice = "home" if deferred and beliefs[i].tests[-1][0] and beliefs[i].tests[-1][1] is not None else decide(beliefs[i], retests)
            if choice != "retest":
                break
            env.mode = i + 1
            retests -= 1
            test(i)
        env.mode = 0
        place(i, home[i] if (deferred or choice == "home") else bin_xy)
    if deferred:
        budget = max_retests
        while (j := choose_verify(beliefs, budget)) is not None:
            env.mode, budget = j + 1, budget - 1
            if prim.try_pick_from_slot(j, float(home[j][0]), float(home[j][1])):
                test(j)
                place(j, home[j])
        env.mode = 0
        for i in range(n):
            if beliefs[i].p_working < 0.5 and prim.try_pick_from_slot(i, float(home[i][0]), float(home[i][1])):
                prim.move_via_clearance_carry(i, float(bin_xy[0]), float(bin_xy[1]), prim.z_slot_approach, prim.z_transport)
                prim.release_gripper_for_place_safe(t=2)
                prim.hold_steps(2)
    return u.evaluate()


def labels_from_rows(rows, roles):
    """Per-call labels (k=CHUNK) for jem.training: joint/location/evidence per role, mode, keyframes."""
    out = {}
    for t in range(0, len(rows), CHUNK):
        loc = np.full(roles, -1, np.int64)
        ev = np.full(roles, -1, np.int64)
        n = len(rows[t]["evidence"])
        loc[:n] = rows[t]["log"]["location"][:n]
        ev[:n] = rows[t]["evidence"]
        joint = np.where((loc >= 0) & (ev >= 0), 3 * loc + ev, -1)
        prev = rows[max(0, t - CHUNK):t]
        keyframes = [int(bool(r["log"]["lamp_on"]) or bool(r["log"]["insert_event"]) or bool(r["log"]["spill_event"])) for r in prev]
        out[t] = {"joint": joint.tolist(), "location": loc.tolist(), "evidence": ev.tolist(),
                  "mode": int(rows[t]["mode"]), "keyframes": keyframes}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "data/batteriessort"))
    p.add_argument("--max-retests", type=int, default=2)
    p.add_argument("--roles", type=int, default=6)
    args = p.parse_args()

    import gymnasium as gym
    import jem.envs.batteries_sort as bs
    from jem.envs.mikasa_primitives import IgnoreSuccessTerminationWrapper, make_primitives
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    bs.register_wrappers()
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_start)
    summary = []
    for e in range(args.episodes):
        seed = args.seed_start + e
        env = gym.make(args.task, num_envs=1, obs_mode="rgb", control_mode="pd_joint_pos",
                       render_mode="rgb_array", reward_mode="normalized_dense")
        steps = int(gym.spec(args.task).max_episode_steps)
        env = IgnoreSuccessTerminationWrapper(apply_mikasa_vla_wrappers(env, include_overlays=False))
        beliefs = []
        rec = Recorder(env, beliefs)
        rec.reset(seed=seed)
        u = rec.unwrapped
        model = ObservationModel(spill=u.SPILL_PROB, flip=u.FLIP_PROB, occlusion=u.OCCLUSION_PROB, p_working=u.P_WORKING)
        prim = make_primitives(rec, steps)
        info = run_episode(rec, prim, model, u.DEFERRED, u.LAMP_MODE == "persistent", args.max_retests, rng)
        rows = rec.rows
        eid = f"{args.task}:{seed:06d}"
        np.savez_compressed(
            out / f"{seed:06d}.npz", episode_id=eid, language=u.LANGUAGE_INSTRUCTION,
            rgb=np.stack([r["rgb"] for r in rows]).astype(np.uint8),
            proprio=np.stack([r["proprio"] for r in rows]).astype(np.float32),
            action=np.stack([r["action"] for r in rows]).astype(np.float32),
            timestamp=(np.arange(len(rows)) * rec.dt).astype(np.float32))
        (out / f"{seed:06d}.labels.json").write_text(json.dumps(labels_from_rows(rows, args.roles)))
        (out / f"{seed:06d}.log.json").write_text(json.dumps({"seed": seed, "logs": [r["log"] for r in rows],
                                                              "success": bool(info["success"][0])}))
        summary.append({"episode_id": eid, "steps": len(rows), "success": bool(info["success"][0])})
        env.close()
        print(summary[-1], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
