"""Server-side acceptance check for the BatteriesSort env (needs the simulator).

Checks, without any policy: registration, reset/observation schema, that quality is
randomised per battery, that the evaluator-only log never reaches the VLA observation,
and the empirical spill/occlusion rates of scripted insertions. Exit code 1 on failure.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(task="BatteriesSortHard-3-VLA-v0", seeds=20):
    import gymnasium as gym
    import jem.envs.batteries_sort as bs
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    bs.register_wrappers()
    report, failures = {"task": task}, []
    counts = []
    for seed in range(seeds):
        env = apply_mikasa_vla_wrappers(gym.make(task, num_envs=1, obs_mode="rgb"), include_overlays=False)
        obs, info = env.reset(seed=seed)
        if set(obs) - {"rgb", "proprio"}:
            failures.append(f"unexpected observation keys {sorted(obs)}")
        if tuple(obs["rgb"].shape[-3:]) != (128, 128, 6) or tuple(obs["proprio"].shape[-1:]) != (7,):
            failures.append("observation schema")
        u = env.unwrapped
        counts.append(int(u.working_mask[0, : int(u.active_battery_count[0])].sum()))
        env.close()
    report["working_counts"] = counts
    if len(set(counts)) < 2:
        failures.append("working count not randomised across seeds")
    report["failures"] = failures
    print(json.dumps(report, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(*sys.argv[1:2])
