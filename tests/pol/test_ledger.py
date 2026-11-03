import numpy as np
import pytest

from pol.ledger import ENTRY_DIM, M_MAX, Entry, Ledger, positional_encoding

M = 3


def entry(t, x=(0.0, 0.0, 0.0), probs=(1 / 3, 1 / 3, 1 / 3), omega=1.0, kind=1):
    return Entry(x=np.array(x, float), kind=kind, logp=np.log(np.array(probs, float)), omega=omega, t=t,
                 summary=np.full(24, 0.5), assoc=np.ones(64) / 8.0)


def test_featurize_layout_and_mask():
    led = Ledger(K=4, M=M)
    led.write(entry(10, x=(0.1, 0.2, 0.3), probs=(0.7, 0.2, 0.1), omega=0.4, kind=2))
    feats, mask = led.featurize(t_now=110)
    assert feats.shape == (4, ENTRY_DIM) and feats.dtype == np.float32
    assert mask.tolist() == [True, False, False, False]
    row = feats[0]
    assert np.allclose(row[:12], positional_encoding(np.array([0.1, 0.2, 0.3])))
    assert row[12:15].tolist() == [0, 0, 1]
    assert np.allclose(row[15:15 + M_MAX], [0.7, 0.2, 0.1, 0, 0, 0, 0, 0])
    assert row[23] == pytest.approx(0.4)
    assert row[24] == pytest.approx(-(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1)), rel=1e-5)
    assert row[25] == pytest.approx(0.1)  # age in units of 1000 steps
    assert np.allclose(row[26:50], 0.5)
    assert not feats[1:].any()


def test_empty_ledger_is_fully_masked():
    feats, mask = Ledger(K=8, M=M).featurize(0)
    assert not mask.any() and not feats.any()


def test_decisive_resolution_and_status():
    led = Ledger(K=4, M=M, eta=0.3, omega0=0.5)
    led.write(entry(0, omega=0.1))
    led.write(entry(1, x=(1, 0, 0), omega=0.9))
    assert led.status(0) == "pending_unobservable"
    assert led.status(1) == "pending_ambiguous"
    w = led.association(np.zeros(3), np.ones(64) / 8.0)
    assert w[0] == pytest.approx(0.9) and w[1] == 0.0  # 1 - p_null in the only entry within rho
    led.resolve(w, np.array([[0.0, -20.0, -20.0], [0.0, 0.0, 0.0]]))
    assert led.status(0) == "resolved" and led.entries[0].entropy < 0.3
    assert led.status(1) == "pending_ambiguous"


def test_association_outside_radius_is_zero_and_changes_nothing():
    led = Ledger(K=4, M=M, rho=0.06)
    led.write(entry(0))
    before = led.entries[0].logp.copy()
    w = led.association(np.array([0.5, 0.0, 0.0]), np.ones(64) / 8.0)
    assert w.tolist() == [0.0]
    led.resolve(w, np.array([[0.0, -20.0, -20.0]]))
    assert np.allclose(led.entries[0].logp, before)


def test_association_uses_context_when_beta_positive():
    led = Ledger(K=4, M=M, rho=0.06, beta=5.0)
    a, b = np.zeros(64), np.zeros(64)
    a[0], b[1] = 1.0, 1.0
    led.write(Entry(np.zeros(3), 1, np.log(np.ones(M) / M), 1.0, 0, np.zeros(24), a))
    led.write(Entry(np.zeros(3), 1, np.log(np.ones(M) / M), 1.0, 1, np.zeros(24), b))
    w = led.association(np.zeros(3), a)
    assert w[0] == pytest.approx(0.9, abs=0.01) and w[1] < 0.01


def test_overflow_keeps_newest_and_size():
    led = Ledger(K=3, M=M)
    for t in range(5):
        led.write(entry(t, x=(t, 0, 0)))
    assert len(led.entries) == 3
    assert 4 in [e.t for e in led.entries]


def test_eviction_prefers_resolved_then_oldest():
    led = Ledger(K=3, M=M, eviction="entropy")
    led.write(entry(0))
    led.write(entry(1, probs=(0.98, 0.01, 0.01)))
    led.write(entry(2, probs=(0.98, 0.01, 0.01)))
    assert led.entries[led.evict_index()].t == 1  # lowest entropy, oldest among ties


def test_fifo_eviction_ablation():
    led = Ledger(K=2, M=M, eviction="fifo")
    led.write(entry(5, probs=(0.98, 0.01, 0.01)))
    led.write(entry(3))
    assert led.entries[led.evict_index()].t == 3


def test_association_sums_to_one_minus_p_null_and_zero_when_none():
    led = Ledger(K=4, M=M, rho=0.06, p_null=0.2)
    led.write(entry(0))
    led.write(entry(1, x=(0.02, 0, 0)))
    led.write(entry(2, x=(1, 0, 0)))
    w = led.association(np.zeros(3), np.ones(64) / 8.0)
    assert w.sum() == pytest.approx(0.8) and w[2] == 0.0 and w[0] > w[1] > 0
    assert not led.association(np.array([0.5, 0, 0]), np.ones(64) / 8.0).any()


def test_one_reencounter_splits_evidence_between_two_close_entries():
    led = Ledger(K=4, M=M, rho=0.06)
    led.write(entry(0))
    led.write(entry(1))
    w = led.association(np.zeros(3), np.ones(64) / 8.0)
    ll = np.array([0.0, 3.0, -3.0])
    led.resolve(w, np.stack([ll, ll]))
    shift = sum(np.log(e.probs[1] / e.probs[2]) for e in led.entries)
    full = Ledger(K=1, M=M)
    full.write(entry(0))
    full.resolve(np.array([1.0]), ll[None])
    assert shift <= np.log(full.entries[0].probs[1] / full.entries[0].probs[2]) + 1e-9
    assert all(np.isfinite(e.logp).all() and np.exp(e.logp).sum() == pytest.approx(1.0) for e in led.entries)


def test_mixture_update_matches_formula():
    led = Ledger(K=1, M=M)
    led.write(entry(0, probs=(0.5, 0.3, 0.2)))
    p, ll, w = np.array([0.5, 0.3, 0.2]), np.array([1.0, 0.0, -1.0]), 0.6
    post = p * np.exp(ll) / (p * np.exp(ll)).sum()
    led.resolve(np.array([w]), ll[None])
    assert np.allclose(led.entries[0].probs, (1 - w) * p + w * post)


def test_consumed_entry_is_evicted_first():
    led = Ledger(K=3, M=M)
    led.write(entry(0))
    led.write(entry(1, probs=(0.98, 0.01, 0.01)))
    led.write(entry(2))
    led.mark_consumed(2)
    assert led.entries[led.evict_index()].t == 2
    feats, _ = led.featurize(t_now=100)
    assert np.signbit(feats[2, 25]) and not np.signbit(feats[0, 25])


def test_resolved_unconsumed_entry_outlives_pending_low_value():
    led = Ledger(K=2, M=M)
    led.write(entry(0, probs=(0.9, 0.05, 0.05)))  # resolved, unconsumed, oldest
    led.write(entry(1, probs=(0.4, 0.3, 0.3)))  # ambiguous pending
    assert led.entries[led.evict_index()].t == 1
    led.mark_consumed(0)
    assert led.entries[led.evict_index()].t == 0


def test_unknown_eviction_raises():
    with pytest.raises(ValueError):
        Ledger(K=2, M=M, eviction="lru")
