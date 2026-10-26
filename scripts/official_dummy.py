"""Another machine: official simulator smoke. Dummy output is never model performance."""

import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jem.environment import official_episode
from jem.data import TASKS


class DummyPolicy:
    chunk_size = 8

    def reset(self):
        pass

    def inference(self, obs, executed=None, dt=None, frames=None):
        return np.zeros((1, 8, 7), np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), default=list(TASKS)[0])
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--output", default=str(ROOT / "runs/official-dummy.json"))
    args = parser.parse_args()
    from mikasa_robo_suite.vla.benchmarking import (
        BenchmarkConfig,
        make_benchmark_env,
        START_SEED,
    )

    config = BenchmarkConfig(n_episodes=1, sim_backend=args.sim_backend)
    env = make_benchmark_env(args.task, config)
    try:
        result, tracked = official_episode(
            env, DummyPolicy(), "check batteries", START_SEED
        )
    finally:
        env.close()
    report = {
        "kind": "dummy_interface_rollout",
        "benchmark_performance": False,
        "task": args.task,
        "steps": result.n_steps,
        "policy_calls": tracked.calls,
        "prefixes": [
            {
                "predicted": len(c["predicted"]),
                "dispatched": len(c["dispatched"]),
                "executed": len(c["executed"]),
            }
            for c in tracked.ledger.all_chunks
        ],
    }
    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
