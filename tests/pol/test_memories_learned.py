import numpy as np
import pytest
import torch

from pol import baseline_models as bm
from pol.events import Event
from pol.ledger import ENTRY_DIM
from pol.memories import StepObs, make_memory, memory_bytes
from pol.outcome_model import OutcomeModel

from test_memories_simple import ForbiddenLog

D = 1152


@pytest.fixture(scope="module")
def assets():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = OutcomeModel(M=4).eval()
    with torch.no_grad():
        n = 32
        model.set_banks({"v1": torch.randn(n, 4, 32, 256), "v2": torch.randn(n, 4, 32, 256),
                         "w2": torch.randn(n, 16), "has_v2": torch.ones(n, dtype=torch.bool),
                         "pooled": torch.randn(n, 2, 1152), "pooled2": torch.randn(n, 2, 1152)})
    return {"proj50": rng.normal(size=(2 * D, 50)) / np.sqrt(2 * D), "proj_mean": np.zeros(2 * D),
            "outcome_model": model, "sigma2": torch.ones(32, 256), "omega0": 0.0,
            "kem_scorer": bm.KeyframeScorer().eval(), "mu_net": bm.RecurrentMemory(K=8).eval(),
            "aura_net": bm.ActionPredictor().eval(), "statemem_net": bm.StateMemCell().eval(),
            "check_net": bm.FeaturePredictor().eval(), "check_mu": 0.0, "check_sd": 1.0, "conformal_q": 0.0,
            "still_threshold": 0.0, "verify_head": bm.VerifyHead().eval()}


def stream(n=300, seed=0, events=None, still=False):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(2, D)).astype(np.float32)
    events = events if events is not None else {t: Event(t, ["G+", "R+", "R0"][t % 3], np.zeros(3)) for t in range(20, n, 40)}
    for t in range(n):
        pooled = base if still else rng.normal(size=(2, D)).astype(np.float32)
        yield StepObs(t=t, pooled=pooled, tokens=rng.normal(size=(32, 256)).astype(np.float32),
                      proprio=rng.normal(size=7).astype(np.float32), action=rng.normal(size=7).astype(np.float32),
                      event=events.get(t), log=ForbiddenLog() if t in events else None)


VARIANTS = ["KEM", "MU", "AURA", "StateMem", "CheckVLA", "CheckFIFO", "AGM", "POL"]


@pytest.mark.parametrize("name", VARIANTS)
def test_contract_on_random_stream(name, assets):
    cfg = {"name": name, "K": 8, "fifo_occluded": name == "CheckFIFO"}
    mem = make_memory(name, cfg, assets)
    assert mem.bytes == memory_bytes(8)
    for o in stream(200):
        mem.observe(o)
        feats, mask = mem.tokens()
        assert feats.shape == (8, ENTRY_DIM) and feats.dtype == np.float32 and mask.dtype == bool
        assert np.isfinite(feats).all() and not feats[~mask].any()


def test_pol_writes_only_after_post_window(assets):
    mem = make_memory("POL", {"name": "POL", "K": 8}, assets)
    ev = {20: Event(20, "R+", np.zeros(3))}
    for o in stream(60, events=ev):
        mem.observe(o)
        written = len(mem.ledger.entries)
        assert written == (1 if o.t >= 20 + 16 - 1 else 0)


def test_pol_view_actions_are_the_actions_at_the_view_frames(assets):
    mem = make_memory("POL", {"name": "POL", "K": 8}, assets)
    ev, seen = Event(20, "R+", np.zeros(3)), {}
    for o in stream(40, events={20: ev}):
        seen[o.t] = o.action
        mem.hist.append(o)
    vact = mem._window(ev)[4]
    assert np.array_equal(vact, np.stack([seen[t] for t in range(20, 36)]))


def test_pol_resolves_at_reencounter_unless_disabled(assets):
    ev = {20: Event(20, "R+", np.zeros(3)), 60: Event(60, "G+", np.array([0.01, 0.0, 0.0]))}

    def first_entry(resolve):
        mem = make_memory("POL", {"name": "POL", "K": 8, "resolve": resolve}, assets)
        snap = None
        for o in stream(100, events=ev):
            mem.observe(o)
            if o.t == 50:
                snap = mem.ledger.entries[0].logp.copy()
        return snap, mem.ledger.entries[0].logp

    before, after = first_entry(resolve=False)
    assert np.allclose(before, after)
    before, after = first_entry(resolve=True)
    assert not np.allclose(before, after)


def test_pol_far_event_does_not_resolve(assets):
    ev = {20: Event(20, "R+", np.zeros(3)), 60: Event(60, "G+", np.array([0.5, 0.0, 0.0]))}
    mem = make_memory("POL", {"name": "POL", "K": 8}, assets)
    for o in stream(100, events=ev):
        mem.observe(o)
        if o.t == 50:
            snap = mem.ledger.entries[0].logp.copy()
    assert np.allclose(snap, mem.ledger.entries[0].logp)


def test_checkfifo_stores_still_chunks(assets):
    a = dict(assets, conformal_q=1e9, still_threshold=1.0)  # never flagged by risk, always still
    plain = make_memory("CheckVLA", {"name": "CheckVLA", "K": 8, "fifo_occluded": False}, a)
    fifo = make_memory("CheckFIFO", {"name": "CheckFIFO", "K": 8, "fifo_occluded": True}, a)
    for o in stream(100, still=True, events={}):
        plain.observe(o)
        fifo.observe(o)
    assert plain.tokens()[1][1:].sum() == 0 and fifo.tokens()[1][1:].sum() > 0


def test_baseline_stage_a_losses_are_scalars():
    b = {"x": torch.randn(4, 50), "label": torch.randint(0, 2, (4,)), "action": torch.randn(4, 7),
         "actions": torch.randn(4, 8, 7), "target": torch.randn(4, 50), "seq": torch.randn(4, 6, 50)}
    for m in [bm.KeyframeScorer(), bm.RecurrentMemory(K=8), bm.ActionPredictor(), bm.FeaturePredictor(),
              bm.StateMemCell(), bm.VerifyHead()]:
        loss = m.stage_a_loss(b)
        assert loss.ndim == 0 and torch.isfinite(loss)
