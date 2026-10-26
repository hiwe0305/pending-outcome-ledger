"""Sim acceptance check for the U-Suite envs (sim venv). Exit code 1 on failure.

    python scripts/pol/check_envs.py BatteriesSort-3-VLA-v0 [SealedBin-3-VLA-v0 ...]

Checks per env: registration, VLA observation schema (rgb [128,128,6], proprio [7]), the
evaluator-only log is not in the observation, held-out failure kinds are off in the train split and
on in the eval split, family-specific scene invariants, and 50 random steps run on the GPU renderer.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def make(task, split, seed=0):
    import gymnasium as gym
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    env = gym.make(task, num_envs=1, obs_mode="rgb", render_backend="gpu", split=split)
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    obs, info = env.reset(seed=seed)
    return env, obs


def check(task):
    failures = []
    env, obs = make(task, "train")
    if set(obs) != {"rgb", "proprio"}:
        failures.append(f"observation keys {sorted(obs)}")
    if tuple(obs["rgb"].shape[-3:]) != (128, 128, 6) or tuple(obs["proprio"].shape[-1:]) != (7,):
        failures.append("observation schema")
    u = env.unwrapped
    for kind in u.HELDOUT:
        if u.failures.p(kind) != 0.0:
            failures.append(f"held-out {kind} active in train split")
    if task.startswith("BatteriesSort-"):
        cover, catch = u.catch_cover.pose.p[0], u.catch_pos[0]
        if float((cover[:2] - catch[:2]).abs().max()) > 1e-4 or float(cover[2]) > 1.0:
            failures.append("catch tray not covered")
        if u.LAMP_MODE != "flash" or not u.DEFERRED:
            failures.append("BatteriesSort must use flash lamp and deferred sorting")
    for _ in range(50):
        obs, _, _, _, info = env.step(env.action_space.sample())
    if "eem_log" in obs or "pol_log" in obs:
        failures.append("evaluator log leaked into observation")
    env.close()
    env, _ = make(task, "eval")
    for kind in env.unwrapped.HELDOUT:
        if env.unwrapped.failures.p(kind) <= 0.0:
            failures.append(f"held-out {kind} inactive in eval split")
    env.close()
    return failures


def main(tasks):
    import jem.envs.batteries_sort as bs  # registers envs

    bs.register_wrappers()
    if any(not t.startswith("BatteriesSort") for t in tasks):
        import jem.envs.bins as bins

        bins.register_wrappers()
    report = {t: check(t) for t in tasks}
    print(json.dumps(report, indent=1))
    sys.exit(1 if any(report.values()) else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or ["BatteriesSort-3-VLA-v0"])
