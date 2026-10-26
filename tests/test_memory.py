import itertools
import pytest
import torch
from jem.memory import (
    J, L, Q, MEMORIES, NEEDS_EVIDENCE, DirectJointMemory, EEMMemory, KeyframeMemory,
    joint_support, transport,
)

EV = 5


def inputs(b=2, n=3, d=8, t=4, seed=0, evidence=True):
    g = torch.Generator().manual_seed(seed)
    prior = torch.randn(b, n, d, generator=g)
    obs = torch.randn(b, n, d, generator=g)
    actions = torch.rand(b, t, 7, generator=g) * 2 - 1
    mask = torch.ones(b, t, dtype=torch.bool)
    dt = torch.full((b, t), 0.1)
    active = torch.ones(b, dtype=torch.bool)
    ev = torch.randn(b, t, EV, generator=g) if evidence else None
    return prior, obs, actions, mask, dt, active, ev, (mask.clone() if evidence else None)


def build(variant, d=8):
    torch.manual_seed(0)
    return MEMORIES[variant](d, EV)


@pytest.mark.parametrize("variant", list(MEMORIES))
def test_belief_is_normalized_finite_and_differentiable(variant):
    m = build(variant)
    m.reset(2, 3)
    prior, obs, actions, mask, dt, active, ev, em = inputs()
    obs.requires_grad_(True)
    out = m.update(prior, obs, actions, mask, dt, active, evidence=ev, evidence_mask=em)
    assert torch.allclose(out.belief.sum(-1), torch.ones(2, 3), atol=1e-5)
    assert torch.isfinite(out.tokens).all() and out.tokens.shape[0] == 2
    (out.tokens.square().mean() + out.belief.square().mean()).backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all() and obs.grad.abs().sum() > 0


@pytest.mark.parametrize("cls", [DirectJointMemory, EEMMemory])
def test_kernels_are_stochastic_on_support(cls):
    m = cls(8, EV)
    x = torch.randn(4, 3, 16)
    k = m.log_kernel(x).exp()
    assert torch.allclose(k.sum(-1), torch.ones(4, 3, J), atol=1e-5)
    assert (k[..., ~joint_support()] == 0).all()


def test_eem_kernel_factorization_is_identifiable():
    """Chain rule: rho(q'|s) is the evidence marginal of the row and D is the conditional."""
    m = EEMMemory(8, EV)
    x = torch.randn(5, 16)
    rho, phys = m.factors(x)
    k = m.log_kernel(x).exp().reshape(5, J, L, Q)
    assert torch.allclose(k.sum(-2), rho.exp(), atol=1e-5)  # recover rho from K
    cond = k / k.sum(-2, keepdim=True).clamp_min(1e-30)
    tested_rows = torch.arange(J) % Q != 0
    # D is recoverable wherever its evidence value is reachable.
    reach = rho.exp() > 0
    got = cond.transpose(-1, -2)[reach]
    want = phys.exp()[reach]
    assert torch.allclose(got, want, atol=1e-5)
    assert (rho.exp()[:, tested_rows, 0] == 0).all()  # tested never returns to untested


def test_log_space_recovers_from_extreme_corrections():
    """Regression: an fp32 softmax correction used to zero the untested mass permanently."""
    m = DirectJointMemory(4)
    m.reset(1, 1)
    q = torch.arange(J) % Q
    push = torch.where(q != 0, 200.0, 0.0)
    log_b = m.correct(m.log_belief, push)
    log_b = m.correct(log_b, -push)
    assert torch.allclose(log_b.exp(), torch.full((1, 1, J), 1 / J), atol=1e-5)
    assert torch.isfinite(log_b).all()


@pytest.mark.parametrize("variant", list(MEMORIES))
def test_inactive_rows_and_unexecuted_steps_do_not_change_state(variant):
    m = build(variant)
    m.reset(2, 3)
    prior, obs, actions, mask, dt, active, ev, em = inputs()
    before = m.belief.clone()
    active = torch.tensor([True, False])
    m.update(prior, obs, actions, mask, dt, active, evidence=ev, evidence_mask=em)
    assert torch.equal(m.belief[1], before[1])


@pytest.mark.parametrize("variant", [v for v in MEMORIES if v not in NEEDS_EVIDENCE])
def test_variants_run_without_evidence_stream(variant):
    m = MEMORIES[variant](8)
    m.reset(2, 3)
    prior, obs, actions, mask, dt, active, _, _ = inputs(evidence=False)
    out = m.update(prior, obs, actions, mask, dt, active)
    assert torch.isfinite(out.belief).all()


def test_evidence_shape_and_prefix_are_validated():
    m = build("EEM")
    m.reset(2, 3)
    prior, obs, actions, mask, dt, active, ev, em = inputs()
    with pytest.raises(ValueError, match="align"):
        m.update(prior, obs, actions, mask, dt, active, evidence=ev[:, :2], evidence_mask=em)
    bad = mask.clone()
    bad[:, 0] = False
    with pytest.raises(ValueError, match="prefix"):
        m.update(prior, obs, actions, bad, dt, active, evidence=ev, evidence_mask=em & bad)


def test_kem_fifo_keeps_last_writes_and_teacher_forcing():
    m = KeyframeMemory(8, EV, capacity=2)
    m.reset(1, 3)
    prior, obs, actions, mask, dt, active, ev, em = inputs(b=1, t=4)
    teacher = torch.tensor([[True, False, True, True]])
    out = m.update(prior, obs, actions, mask, dt, active, evidence=ev, evidence_mask=em, keyframe_teacher=teacher)
    assert m.valid.all() and torch.equal(m.buffer[0], ev[0, [2, 3]])
    assert out.aux["keyframe_logits"].shape == (1, 4)


def test_kem_requires_evidence_stream():
    with pytest.raises(ValueError):
        KeyframeMemory(8, 0)


def test_transport_permutation_is_exact():
    state = torch.randn(1, 3, 4)
    perm = torch.eye(3)[[2, 0, 1]][None]
    assert torch.equal(transport(state, perm), state[:, [2, 0, 1]])
    with pytest.raises(ValueError):
        transport(state, torch.full((1, 3, 3), 0.5))


def test_role_permutation_equivariance_of_filters():
    for variant in ("B2", "EEM"):
        m = build(variant)
        prior, obs, actions, mask, dt, active, ev, em = inputs(b=1)
        m.reset(1, 3)
        a = m.update(prior, obs, actions, mask, dt, active, evidence=ev, evidence_mask=em).belief
        perm = [2, 0, 1]
        m.reset(1, 3)
        b = m.update(prior[:, perm], obs[:, perm], actions, mask, dt, active, evidence=ev, evidence_mask=em).belief
        assert torch.allclose(a[:, perm], b, atol=1e-5)
