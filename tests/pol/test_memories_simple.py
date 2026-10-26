import numpy as np
import pytest

from pol.events import Event
from pol.ledger import ENTRY_DIM
from pol.memories import StepObs, make_memory, memory_bytes

D = 1152


class ForbiddenLog(dict):
    """A log that fails the test if a non-oracle memory reads it."""

    def _deny(self, *a, **k):
        raise AssertionError("memory read evaluator-only log")

    __getitem__ = get = __contains__ = keys = items = values = __iter__ = _deny


def assets(seed=0):
    rng = np.random.default_rng(seed)
    return {"proj50": rng.normal(size=(2 * D, 50)) / np.sqrt(2 * D), "proj_mean": np.zeros(2 * D),
            "markov": lambda pooled, proprio: 0.25}


def stream(n=300, seed=0, log_factory=ForbiddenLog):
    rng = np.random.default_rng(seed)
    for t in range(n):
        event = Event(t, ["G+", "R+", "R0"][t % 3], rng.normal(size=3) * 0.1) if t % 40 == 20 else None
        log = log_factory({"event_outcome": t % 3}) if event else None
        yield StepObs(t=t, pooled=rng.normal(size=(2, D)).astype(np.float32), tokens=None,
                      proprio=rng.normal(size=7).astype(np.float32), action=rng.normal(size=7).astype(np.float32),
                      event=event, log=log)


@pytest.mark.parametrize("name", ["B0", "Frames", "Random", "Markov"])
def test_contract_and_no_log_access(name):
    mem = make_memory(name, {"name": name, "K": 8}, assets())
    mem.reset()
    for o in stream():
        mem.observe(o)
        feats, mask = mem.tokens()
        assert feats.shape == (8, ENTRY_DIM) and feats.dtype == np.float32
        assert mask.shape == (8,) and mask.dtype == bool
        assert not feats[~mask].any()


def test_b0_is_always_empty():
    mem = make_memory("B0", {"name": "B0", "K": 8}, assets())
    for o in stream(50):
        mem.observe(o)
    assert not mem.tokens()[1].any()


def test_frames_keeps_most_recent_call_frames():
    a = assets()
    mem = make_memory("Frames", {"name": "Frames", "K": 3, "exec_steps": 8}, a)
    obs = list(stream(80))
    for o in obs:
        mem.observe(o)
    feats, mask = mem.tokens()
    assert mask.all()
    expected = [obs[t].pooled.reshape(-1) @ a["proj50"] for t in (56, 64, 72)]
    assert np.allclose(feats, np.stack(expected), atol=1e-4)


def test_random_write_is_deterministic_per_seed():
    def run(seed):
        mem = make_memory("Random", {"name": "Random", "K": 8, "write_prob": 0.1, "seed": seed}, assets())
        mem.reset()
        for o in stream(200):
            mem.observe(o)
        return mem.tokens()[0]

    assert np.array_equal(run(1), run(1)) and not np.array_equal(run(1), run(2))


def test_oracle_reads_true_outcome_and_requires_log():
    mem = make_memory("Oracle", {"name": "Oracle", "K": 8}, assets())
    mem.reset()
    for o in stream(100, log_factory=dict):
        mem.observe(o)
    feats, mask = mem.tokens()
    assert mask.sum() == 2  # events at t = 20 and 60
    assert np.allclose(feats[0, 15:18], [0, 0, 1])  # outcome of t=20 is 20 % 3 == 2
    with pytest.raises(ValueError):
        mem.observe(StepObs(0, np.zeros((2, D)), None, np.zeros(7), np.zeros(7), Event(0, "G+", np.zeros(3)), None))


def test_memory_bytes_equal_across_variants():
    assert memory_bytes(8) == 8 * ENTRY_DIM * 4
    for name in ["B0", "Frames", "Random", "Markov", "Oracle"]:
        assert make_memory(name, {"name": name, "K": 8}, assets()).bytes == memory_bytes(8)
