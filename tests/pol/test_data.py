import numpy as np
import pytest
import torch

from pol.data import (Normalizer, call_sample, event_tuples, hard_negative_batches, pad, prev_actions,
                      replay_memory, split_seeds, token_frames)
from pol.events import Event, FsmThresholds
from pol.memories import make_memory
from pol.outcome_model import OutcomeModel

TH = FsmThresholds(w_open=0.035, w_empty=0.005)


def episode(T=200, seed=0):
    rng = np.random.default_rng(seed)
    w = np.full(T, 0.04)
    w[30:70] = 0.02  # grasp at ~32, release at ~71
    w[120:160] = 0.02
    proprio = np.zeros((T, 7), np.float32)
    proprio[:, 6] = w
    proprio[:, :3] = 0.001 * np.arange(T)[:, None]
    return {"proprio": proprio, "action": rng.normal(size=(T, 7)).astype(np.float32), "logs": [{}] * T,
            "rgb": rng.integers(0, 255, (T, 128, 128, 6), dtype=np.uint8), "language": "do it", "seed": seed}


def features(T, seed=0):
    rng = np.random.default_rng(seed)
    pooled = rng.normal(size=(T, 2, 1152)).astype(np.float32)
    tokens = {t: rng.normal(size=(32, 256)).astype(np.float32) for t in range(T)}
    return pooled, tokens


@pytest.fixture(scope="module")
def pol_assets():
    torch.manual_seed(0)
    model = OutcomeModel(M=4).eval()
    with torch.no_grad():
        model.set_banks({"v1": torch.randn(16, 4, 32, 256), "v2": torch.randn(16, 4, 32, 256),
                         "w2": torch.randn(16, 16), "has_v2": torch.ones(16, dtype=torch.bool),
                         "pooled": torch.randn(16, 2, 1152), "pooled2": torch.randn(16, 2, 1152)})
    rng = np.random.default_rng(0)
    return {"outcome_model": model, "sigma2": torch.ones(32, 256), "omega0": 0.0,
            "proj50": rng.normal(size=(2304, 50)) / 48, "proj_mean": np.zeros(2304)}


@pytest.mark.parametrize("name", ["Frames", "POL"])
def test_memory_at_call_t_ignores_the_future(name, pol_assets):
    ep = episode()
    pooled, tokens = features(len(ep["proprio"]))
    mem = make_memory(name, {"name": name, "K": 8}, pol_assets)
    steps, feats, _ = replay_memory(mem, ep, pooled, tokens, TH)
    t_cut = 96
    ep2 = dict(ep, action=ep["action"].copy(), proprio=ep["proprio"].copy())
    ep2["action"][t_cut:] += 5.0  # includes action[t_cut], the prediction target at the call t_cut
    ep2["proprio"][t_cut + 1:, 6] = 0.0
    pooled2, tokens2 = pooled.copy(), dict(tokens)
    pooled2[t_cut + 1:] += 3.0
    for t in range(t_cut + 1, len(pooled)):
        tokens2[t] = tokens2[t] + 3.0
    steps2, feats2, _ = replay_memory(make_memory(name, {"name": name, "K": 8}, pol_assets), ep2, pooled2, tokens2, TH)
    keep = steps <= t_cut
    assert np.allclose(feats[keep], feats2[keep], atol=1e-5)
    assert not np.allclose(feats[~keep], feats2[~keep])


def test_prev_actions_shift():
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    p = prev_actions(a)
    assert not p[0].any() and np.array_equal(p[1:], a[:-1])


def test_event_tuples_have_both_views():
    ep = episode()
    pooled, tokens = features(len(ep["proprio"]))
    items = event_tuples(ep, pooled, tokens, TH, rho=1.0)
    assert [it["kind"] for it in items] == ["G+", "R+", "G+", "R+"]
    assert items[0]["has_v2"] and not items[-1]["has_v2"]
    assert items[0]["v1"].shape == (4, 32, 256) and items[0]["act"].shape == (16, 7)
    # view actions: the actions observed at each view's frames (StepObs.action = previous action)
    a = prev_actions(ep["action"])
    from pol.events import detect_events, reencounter_index

    events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], TH)
    t2 = events[reencounter_index(events, 0, 1.0)].t
    assert np.array_equal(items[0]["act1"], a[items[0]["t"]:items[0]["t"] + 16])
    assert np.array_equal(items[0]["act2"], a[t2:t2 + 16]) and not items[-1]["act2"].any()


def test_split_has_no_overlap():
    tr, va = split_seeds(range(100))
    assert len(va) == 10 and not set(tr) & set(va) and sorted(tr + va) == list(range(100))


def test_hard_negative_batches_partition_and_exclude_self():
    rng = np.random.default_rng(0)
    ctx, kinds = rng.normal(size=(300, 8)), rng.choice(["G+", "R+"], 300)
    batches = hard_negative_batches(ctx, kinds, batch_size=64, group=8)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(300))


def test_token_frames_cover_calls_and_windows():
    frames = token_frames(100, [Event(50, "R+", np.zeros(3))])
    assert set(range(0, 100, 8)) <= set(frames) and set(range(42, 66)) <= set(frames)


def test_action_pad_roundtrip_and_sample_layout():
    ep = episode()
    norm = Normalizer.fit([ep])
    a = ep["action"][:5]
    assert np.allclose(norm.unaction(pad(norm.action(a))), a, atol=1e-5)
    s = call_sample(ep, len(ep["action"]) - 4, np.zeros((8, 50)), np.zeros(8, bool), norm)
    assert s["actions"].shape == (16, 32) and s["state"].shape == (32,)
    assert s["action_is_pad"].sum() == 12 and s["image"]["base_0_rgb"].shape == (128, 128, 3)
