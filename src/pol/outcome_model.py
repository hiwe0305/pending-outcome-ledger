"""Latent outcome model of POL, trained by EM on frozen features (Stage A).

Each event j has a discrete outcome z in {1..M}. View scores are contrastive, so
exp s(v, z) is proportional to p(v | z, ctx) / p(v | ctx); a density ratio is all Bayes' rule
over z needs. The normaliser over candidate views is the in-batch set during training and a
fixed bank of training views at inference (set_banks).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def mlp(*dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


def usage_kl(q):
    """KL(mean_b q || uniform): penalises collapse onto few hypotheses."""
    marginal = q.mean(0).clamp_min(1e-8)
    return (marginal * (marginal.log() + math.log(q.shape[1]))).sum()


class ViewEncoder(nn.Module):
    def __init__(self, d_tok, n_time, n_tok, extra=0, layers=2, heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_time * n_tok + 1, d_tok))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_tok))
        self.extra = nn.Linear(extra, d_tok) if extra else None
        layer = nn.TransformerEncoderLayer(d_tok, heads, 2 * d_tok, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_tok, d_tok)

    def forward(self, v, extra=None):
        b = v.shape[0]
        cls = self.cls.expand(b, -1, -1)
        if self.extra is not None:
            cls = cls + self.extra(extra)[:, None]
        x = torch.cat([cls, v.flatten(1, 2)], 1) + self.pos
        return F.normalize(self.out(self.body(x)[:, 0]), dim=-1)


class OutcomeModel(nn.Module):
    def __init__(self, M=6, d_feat=1152, d_tok=256, n_time=4, n_tok=32, d_ctx=512, d_act=128, tau=0.1, d_assoc=64):
        super().__init__()
        self.M, self.tau, self.n_tok, self.d_tok = M, tau, n_tok, d_tok
        self.ctx_feat = nn.Sequential(nn.Linear(2 * d_feat, d_ctx), nn.LayerNorm(d_ctx), nn.GELU())
        self.ctx_act = mlp(16 * 7, d_act, d_act)
        d = d_ctx + d_act
        self.prior = mlp(d, 256, M)
        self.hyp = nn.Parameter(torch.randn(M, 128) * 0.5)
        self.hyp_head = nn.ModuleList([mlp(d + 128, 256, d_tok) for _ in range(2)])
        self.view = nn.ModuleList([ViewEncoder(d_tok, n_time, n_tok), ViewEncoder(d_tok, n_time, n_tok, extra=16)])
        self.post_head = mlp(d + 128, 512, n_tok * d_tok)
        self.summary_head = nn.Linear(d, 24)
        self.assoc_head = nn.Linear(2 * d_feat, d_assoc)
        self.register_buffer("bank1", torch.zeros(0, d_tok))
        self.register_buffer("bank2", torch.zeros(0, d_tok))

    # ------------------------------------------------------------------ pieces
    def context(self, pooled, act):
        return torch.cat([self.ctx_feat(pooled.flatten(1)), self.ctx_act(act.flatten(1))], -1)

    def prior_logits(self, ctx):
        return self.prior(ctx)

    def _with_hyp(self, ctx):
        b = ctx.shape[0]
        return torch.cat([ctx[:, None].expand(b, self.M, -1), self.hyp[None].expand(b, -1, -1)], -1)

    def hyp_embed(self, ctx, which):
        return F.normalize(self.hyp_head[which - 1](self._with_hyp(ctx)), dim=-1)

    def view_embed(self, v, which, w2=None):
        return self.view[which - 1](v, w2 if which == 2 else None)

    def scores(self, f, u):
        """[B, B', M]: score of candidate view c under hypothesis z of event b."""
        return torch.einsum("bmd,cd->bcm", u, f) / self.tau

    def predict_post(self, ctx):
        return self.post_head(self._with_hyp(ctx)).view(ctx.shape[0], self.M, self.n_tok, self.d_tok)

    def omega(self, ctx, sigma2):
        """Expected information about z from view 1: hypothesis disagreement of predicted features."""
        p = self.prior_logits(ctx).softmax(-1)
        g = self.predict_post(ctx)
        gbar = (p[..., None, None] * g).sum(1, keepdim=True)
        dev = ((g - gbar) ** 2 / sigma2).mean((-1, -2))
        return (p * dev).sum(-1)

    def summary(self, ctx):
        return torch.tanh(self.summary_head(ctx))

    def assoc_embed(self, pooled):
        return F.normalize(self.assoc_head(pooled.flatten(1)), dim=-1)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def set_banks(self, batch, max_size=4096):
        self.bank1 = self.view_embed(batch["v1"][:max_size], 1)
        keep = batch["has_v2"][:max_size]
        self.bank2 = self.view_embed(batch["v2"][:max_size][keep], 2, batch["w2"][:max_size][keep])

    def _banked_loglik(self, f, u, bank):
        own = torch.einsum("bmd,bd->bm", u, f) / self.tau
        ref = torch.einsum("bmd,cd->bcm", u, bank) / self.tau
        return own - (torch.logsumexp(ref, 1) - math.log(bank.shape[0]))

    def log_posterior_v1(self, pooled, act, v1):
        ctx = self.context(pooled, act)
        ll = self._banked_loglik(self.view_embed(v1, 1), self.hyp_embed(ctx, 1), self.bank1)
        return torch.log_softmax(self.prior_logits(ctx) + ll, -1)

    def loglik_v2(self, pooled_j, act_j, v2, w2):
        """Log-likelihood ratio of a re-encounter view under each hypothesis of the original event j."""
        ctx = self.context(pooled_j, act_j)
        return self._banked_loglik(self.view_embed(v2, 2, w2), self.hyp_embed(ctx, 2), self.bank2)


def _nce(S):
    """Per-hypothesis InfoNCE log-ratio of each event's own view against the in-batch candidates."""
    own = torch.diagonal(S, dim1=0, dim2=1).T  # [B, M]
    return own - (torch.logsumexp(S, 1) - math.log(S.shape[1]))


def em_step(model, batch, lam_g=1.0, lam_u=0.1, lam_a=0.1):
    ctx = model.context(batch["pooled"], batch["act"])
    prior = torch.log_softmax(model.prior_logits(ctx), -1)
    ll1 = _nce(model.scores(model.view_embed(batch["v1"], 1), model.hyp_embed(ctx, 1)))
    ll2 = _nce(model.scores(model.view_embed(batch["v2"], 2, batch["w2"]), model.hyp_embed(ctx, 2)))
    has2 = batch["has_v2"].float()[:, None]
    joint = prior + ll1 + has2 * ll2
    q = joint.softmax(-1).detach()  # E-step
    n2 = has2.sum().clamp_min(1.0)
    prior_nll = -(q * prior).sum(-1).mean()
    nce1 = -(q * ll1).sum(-1).mean()
    nce2 = -(q * ll2 * has2).sum() / n2
    target = batch["v1"].mean(1)
    recon = (q[..., None, None] * (model.predict_post(ctx) - target[:, None]) ** 2).mean((-1, -2)).sum(-1).mean()
    ukl = usage_kl(joint.softmax(-1))
    a1, a2 = model.assoc_embed(batch["pooled"]), model.assoc_embed(batch["pooled2"])
    logits = a1 @ a2.T / model.tau
    labels = torch.arange(len(a1), device=a1.device)
    assoc = (F.cross_entropy(logits, labels, reduction="none") * has2[:, 0]).sum() / n2
    loss = prior_nll + nce1 + nce2 + lam_g * recon + lam_u * ukl + lam_a * assoc
    return {"loss": loss, "prior_nll": prior_nll.detach(), "nce1": nce1.detach(), "nce2": nce2.detach(),
            "recon": recon.detach(), "usage_kl": ukl.detach(), "assoc": assoc.detach(), "q": q}
