"""Evaluate a trained adapter with the unchanged official MIKASA episode runner.

Runs 50 episodes/task on the fixed seed stream and writes records with the run
provenance that jem.evaluation.paired_report requires. No dummy/random head is
ever accepted as model performance. Needs the GPU server with the simulator.

    python scripts/evaluate_oft.py --source third_party_full/openvla-oft \
        --checkpoint checkpoints/openvla-7b/<rev> --run runs/train/EEM-s17 \
        --tasks BatteriesSortHard-3-VLA-v0 BatteriesSortHard-6-VLA-v0 --output runs/eval/EEM-s17
"""

import argparse, json, pathlib, subprocess, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jem.backbone import configure_lora, load_local_oft  # noqa: E402
from jem.checkpointing import load_adapter  # noqa: E402
from jem.data import SORT_TASKS, TASKS, TrainNormalizer  # noqa: E402
from jem.environment import official_episode  # noqa: E402
from jem.policy import JointPolicy  # noqa: E402


def make_env(task, sim_backend):
    from mikasa_robo_suite.vla.benchmarking import BenchmarkConfig, make_benchmark_env

    if task in TASKS:
        return make_benchmark_env(task, BenchmarkConfig(n_episodes=50, sim_backend=sim_backend))
    import gymnasium as gym
    import jem.envs.batteries_sort as bs
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    bs.register_wrappers()
    env = gym.make(task, num_envs=1, obs_mode="rgb", control_mode="pd_ee_delta_pose", sim_backend=sim_backend)
    return apply_mikasa_vla_wrappers(env, include_overlays=False)


def stratum(env):
    if not hasattr(env.unwrapped, "spills"):
        return "released_task"
    from jem.envs.batteries_sort import episode_strata

    return episode_strata(env)


def tests_per_battery(env):
    u = env.unwrapped
    if not hasattr(u, "tests"):
        return None
    active = u.active_mask[0]
    return float(u.tests[0][active].float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", nargs="+", default=list(SORT_TASKS[:2]))
    p.add_argument("--sim-backend", default="gpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    unknown = set(args.tasks) - set(TASKS) - set(SORT_TASKS)
    if unknown:
        raise SystemExit(f"Unknown tasks {sorted(unknown)}")
    manifest = json.loads((ROOT / "configs/manifest.json").read_text())
    if manifest["gates"]["G2_env"] != "verified":
        raise RuntimeError("G2: env/control/reset interface not verified on this machine")
    run = pathlib.Path(args.run)
    training = json.loads((run / "training.json").read_text())
    if training.get("kind") != "training_run" or training["provenance"]["init_checkpoint"] != manifest["checkpoint"]["revision"]:
        raise RuntimeError("Need a trained adapter with the pinned initialization")
    import mikasa_robo_suite
    from mikasa_robo_suite.vla.benchmarking import NUM_EPISODES_PER_TASK, START_SEED

    repo = pathlib.Path(mikasa_robo_suite.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["sources"]["mikasa"]["revision"]:
        raise RuntimeError("Simulator revision mismatch")
    torch.manual_seed(training["seed"])
    backbone = load_local_oft(args.source, args.checkpoint, manifest)
    configure_lora(backbone, training["config"]["training"]["lora_rank"])
    v = training["variant_config"]
    policy = JointPolicy(backbone, v["variant"], evidence=v["evidence"], gate=v["gate"], gate_kwargs=v.get("gate_kwargs"),
                         proprio_normalizer=TrainNormalizer.load(run / "proprio-normalizer.json"),
                         action_normalizer=TrainNormalizer.load(run / "action-normalizer.json")).to("cuda")
    load_adapter(policy, run / "adapter.pt")
    policy.eval()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for task in args.tasks:
        env = make_env(task, args.sim_backend)
        language = getattr(env.unwrapped, "LANGUAGE_INSTRUCTION", "")
        try:
            for i in range(NUM_EPISODES_PER_TASK):
                start = time.perf_counter()
                result, tracked = official_episode(env, policy, language, START_SEED + i)
                records.append({
                    "kind": "model_evaluation",
                    "variant": training["name"],
                    "training_seed": training["seed"],
                    "task": task,
                    "episode_seed": result.seed,
                    "success_once": bool(result.success_once),
                    "episode_length": result.n_steps,
                    "policy_calls": tracked.calls,
                    "elapsed_seconds": time.perf_counter() - start,
                    "provenance": {**training["provenance"], "benchmark_revision": revision},
                    "stratum": stratum(env),
                    "tests_per_battery": tests_per_battery(env),
                })
                (out / "records.json").write_text(json.dumps(records, indent=2))
        finally:
            env.close()
    print("Saved success_once records:", out / "records.json")


if __name__ == "__main__":
    main()
