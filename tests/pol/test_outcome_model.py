import numpy as np
import pytest
import torch
from sklearn.metrics import normalized_mutual_info_score

from pol.outcome_model import OutcomeModel, em_step

SMALL = dict(M=2, d_feat=16, d_tok=16, n_time=1, n_tok=4, d_ctx=16, d_act=8, d_assoc=8, tau=0.1)


def synthetic(n, seed=0, d=16, n_tok=4):
    """Two hidden outcomes. v2 always shows it; v1 shows it only for 'observable' events."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(0, 2, (n,), generator=g)
    observable = torch.rand(n, generator=g) < 0.5
    pattern = torch.randn(2, n_tok, d, generator=torch.Generator().manual_seed(123)) * 2.0
    common = torch.randn(n_tok, d, generator=torch.Generator().manual_seed(7))
    noise = lambda: 0.3 * torch.randn(n, 1, n_tok, d, generator=g)  # noqa: E731
    v1 = torch.where(observable[:, None, None, None], pattern[z][:, None], common[None, None]) + noise()
    v2 = pattern[z][:, None] + noise()
    pooled = torch.randn(n, 2, d, generator=g)
    pooled[:, 0, 0] = 3.0 * observable.float()  # the context shows whether the outcome region is visible
    batch = dict(pooled=pooled, act=torch.randn(n, 16, 7, generator=g), v1=v1, v2=v2,
                 w2=torch.randn(n, 16, generator=g), has_v2=torch.ones(n, dtype=torch.bool),
                 pooled2=pooled + 0.05 * torch.randn(n, 2, d, generator=g))
    return batch, z.numpy(), observable.numpy()


def test_shapes_and_posterior_normalised():
    torch.manual_seed(0)
    m = OutcomeModel(**SMALL)
    batch, _, _ = synthetic(32)
    out = em_step(m, batch)
    assert set(out) >= {"loss", "prior_nll", "nce1", "nce2", "recon", "usage_kl", "assoc", "q"}
    assert out["q"].shape == (32, 2) and torch.allclose(out["q"].sum(1), torch.ones(32))
    assert not out["q"].requires_grad and out["loss"].requires_grad
    ctx = m.context(batch["pooled"], batch["act"])
    assert m.predict_post(ctx).shape == (32, 2, 4, 16)
    assert m.omega(ctx, torch.ones(4, 16)).shape == (32,)
    assert m.summary(ctx).shape == (32, 24)
    a = m.assoc_embed(batch["pooled"])
    assert a.shape == (32, 8) and torch.allclose(a.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_usage_kl_zero_for_uniform_marginal():
    from pol.outcome_model import usage_kl
    assert usage_kl(torch.full((10, 4), 0.25)).item() == pytest.approx(0.0, abs=1e-6)
    assert usage_kl(torch.eye(4)[[0] * 10]).item() > 1.0


def test_em_recovers_outcomes_and_observability():
    torch.manual_seed(0)
    m = OutcomeModel(**SMALL)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for step in range(300):
        batch, _, _ = synthetic(128, seed=step)
        out = em_step(m, batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    batch, z, observable = synthetic(512, seed=10_000)
    with torch.no_grad():
        q = em_step(m, batch)["q"].numpy()
        sigma2 = batch["v1"].mean(1).var(0) + 1e-6
        omega = m.omega(m.context(batch["pooled"], batch["act"]), sigma2).numpy()
    assert normalized_mutual_info_score(z, q.argmax(1)) > 0.8
    # Hypothesis disagreement must be high where v1 can tell outcomes apart and low where it cannot.
    assert omega[observable].mean() > 3 * omega[~observable].mean()
    with torch.no_grad():
        m.set_banks(batch)
        lp1 = m.log_posterior_v1(batch["pooled"], batch["act"], batch["v1"]).numpy()
    ent = -(np.exp(lp1) * lp1).sum(1)
    assert ent[observable].mean() < ent[~observable].mean()
    assert np.isfinite(omega).all()


def test_banked_posteriors_are_normalised():
    torch.manual_seed(0)
    m = OutcomeModel(**SMALL)
    batch, _, _ = synthetic(64)
    with torch.no_grad():
        m.set_banks(batch)
        lp1 = m.log_posterior_v1(batch["pooled"], batch["act"], batch["v1"])
        ll2 = m.loglik_v2(batch["pooled"], batch["act"], batch["v2"], batch["w2"])
    assert torch.allclose(lp1.exp().sum(1), torch.ones(64), atol=1e-5)
    assert ll2.shape == (64, 2) and torch.isfinite(ll2).all()
