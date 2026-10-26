"""Save the two VLA camera views (top | wrist) of a U-Suite env after reset, for visual checks.

    python scripts/pol/render_scene.py SealedBin-3-VLA-v0 out.png [--seed 0] [--steps 0]
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    a = p.parse_args()
    import gymnasium as gym
    from PIL import Image
    import jem.envs.batteries_sort as bs
    import jem.envs.bins as bins
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    bs.register_wrappers()
    bins.register_wrappers()
    env = apply_mikasa_vla_wrappers(gym.make(a.task, num_envs=1, obs_mode="rgb"), include_overlays=False)
    obs, _ = env.reset(seed=a.seed)
    for _ in range(a.steps):
        obs, *_ = env.step(env.action_space.sample() * 0)
    rgb = obs["rgb"][0].cpu().numpy()
    img = np.concatenate([rgb[..., :3], rgb[..., 3:]], 1).astype(np.uint8)
    Image.fromarray(img).resize((512, 256), Image.NEAREST).save(a.out)


if __name__ == "__main__":
    main()
