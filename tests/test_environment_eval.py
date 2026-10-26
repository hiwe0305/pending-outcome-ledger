"""Execute exact pinned runner functions on a fake env: interface test, no simulator claim."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import collections, json
import numpy as np
import pytest
import torch
from jem.environment import (
    TrackingEnv,
    OfficialPolicyAdapter,
    controller_action,
    ActionLedger,
)
from jem.evaluation import SHARED_PROVENANCE, paired_report, START_SEED
from jem.data import SORT_TASKS
from jem.policy import VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def pinned_runner():
    source = ROOT / "third_party/mikasa/mikasa_robo_suite/vla/benchmarking.py"
    if not source.exists():
        pytest.skip("Pinned audit source absent: run fetch_sources.py")
    tree = ast.parse(source.read_text())
    names = {
        "run_episode",
        "_chunk_actions",
        "_batch_action_for_single_env",
        "_first_scalar",
        "EpisodeResult",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    # No simulator registrations/imports: only the unchanged audited queue functions.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    scope = {
        "torch": torch,
        "np": np,
        "deque": collections.deque,
        "dataclass": dataclass,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope["run_episode"]


class FakeEnv:
    def __init__(self, length=11):
        self.max_episode_steps = length
        self.action_space = SimpleNamespace(shape=(7,))
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.control_freq = 10
        self.actions = []

    def obs(self):
        return {
            "rgb": np.zeros((1, 128, 128, 6), np.uint8),
            "proprio": np.zeros((1, 7), np.float32),
            "oracle_info": {"secret": 1},
        }

    def reset(self, seed=None):
        self.t = 0
        self.actions = []
        self.seed = seed
        return self.obs(), {"language_instruction": "check all batteries"}

    def step(self, a):
        self.actions.append(a)
        self.t += 1
        return (
            self.obs(),
            0.0,
            self.t >= self.max_episode_steps,
            False,
            {"success": self.t == 2},
        )

    def close(self):
        pass


class RecordingPolicy:
    chunk_size = 8

    def __init__(self, variant):
        from jem.policy import JointPolicy
        from jem.fixtures import TinyBackbone

        self.model = JointPolicy(TinyBackbone(), variant)
        self.model.eval()

    def reset(self):
        self.seen = []
        self.model.reset()

    def inference(self, obs, executed, dt, frames=None):
        assert not hasattr(obs, "oracle_info")
        self.seen.append((executed.copy(), dt.copy()))
        return self.model.inference(obs, executed, dt, frames)


def test_pinned_runner_cadence_prefix_termination_and_reset():
    run = pinned_runner()
    for variant in VARIANTS:
        policy = RecordingPolicy(variant)
        tracked = TrackingEnv(FakeEnv(), policy, "check batteries")
        adapter = OfficialPolicyAdapter(policy)
        adapter.bind(tracked)
        result, _ = run(tracked, adapter, START_SEED)
        assert result.success_once and result.n_steps == 11 and tracked.calls == 2
        assert [len(x[0]) for x in policy.seen] == [0, 8] and np.allclose(
            policy.seen[1][1], 0.1
        )
        assert (
            len(tracked.ledger.predicted) == 8
            and len(tracked.ledger.dispatched) == len(tracked.ledger.executed) == 3
        )
        result, _ = run(tracked, adapter, START_SEED + 1)
        assert tracked.calls == 2 and len(policy.seen[0][0]) == 0


def test_dispatch_failure_is_not_execution_and_units():
    ledger = ActionLedger()
    chunk = np.ones((3, 7), np.float32) * 2
    ledger.prediction(chunk)
    actual = ledger.dispatch(chunk[0])
    assert actual.shape == (7,) and (actual == 1).all() and len(ledger.executed) == 0
    ledger.acknowledge(0.1)
    assert len(ledger.prefix()[0]) == 1
    with pytest.raises(ValueError):
        ledger.dispatch(np.zeros(7))
    with pytest.raises(ValueError):
        controller_action(np.zeros(6))
    with pytest.raises(ValueError):
        controller_action(np.full(7, np.nan))


def records():
    return [
        {
            "kind": "model_evaluation",
            "variant": m,
            "training_seed": s,
            "episode_seed": START_SEED + i,
            "task": task,
            "success_once": (i % 2 == 0),
            "provenance": {k: "same" for k in SHARED_PROVENANCE},
        }
        for m in ["KEM", "EEM"]
        for s in [17, 29]
        for i in range(50)
        for task in SORT_TASKS[:2]
    ]


def test_paired_cluster_reporting_and_completeness():
    rows = records()
    result = paired_report(rows, repeats=50)
    assert result["macro"] == {"KEM": 0.5, "EEM": 0.5} and result["difference_ci95"] == [
        0.0,
        0.0,
    ]
    assert result["conclusion"] == "practically_equivalent"
    mismatched = records()
    for r in mismatched:
        if r["variant"] == "EEM":
            r["provenance"] = {**r["provenance"], "label_hash": "other"}
    with pytest.raises(ValueError, match="Provenance"):
        paired_report(mismatched, repeats=2)
    with pytest.raises(ValueError):
        paired_report(rows[:-1], repeats=2)
    with pytest.raises(ValueError):
        paired_report(rows + [rows[0]], repeats=2)
    rows[0]["kind"] = "dummy_rollout"
    with pytest.raises(ValueError):
        paired_report(rows, repeats=2)
