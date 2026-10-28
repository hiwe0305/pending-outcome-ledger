"""Latent outcome model of POL, trained by EM on frozen features (Stage A).

Each event j has a discrete outcome z in {1..M}. View scores are contrastive, so
exp s(v, z) is proportional to p(v | z, ctx) / p(v | ctx); a density ratio is all Bayes' rule
over z needs. The normaliser over candidate views is the in-batch set during training and a
fixed bank of training views at inference (set_banks).

The hypothesis embedding u_z sees only z and the event kind, never the context: views and context come
from the same scene, so a context-dependent u matches each event's own view through the context alone
and leaves z unused (instance discrimination). The scene part of a view is explained away instead by a
regression from the view's own pre-window and the actions executed during the view (scene_mean, plain MSE
on frozen features, one prediction per time block): the encoder sees v - scene_mean(pooled, actions), so
what is left to cluster is what neither the scene nor the motion predicts. Without the actions, where the
arm and the object go after the event stays in the residual, and within a task:kind group that is the
largest variation shared by v1 and v2 (round 1: z found it instead of the outcome). The context also
enters the prior and the predicted post features.

InfoNCE alone does not pick a partition: with u_z shared across events and q uniform, its optimum is a
flat score (Jensen), so EM stays at the symmetric point. Paired views do: v1 and v2 of an event show the
same outcome, so the M-step also maximises I(z1; z2) between the per-view posteriors (IIC, Ji et al.
2019), on events with a re-encounter. That term rewards only what both views show, and a silent outcome
is absent from v2 by design (the BatteriesSort lamp is off at the re-encounter), so lam_i1 adds IIC
between two random augmentations of v1 itself (time-block and token dropout, feature noise).

All of it is conditional on the event's task:kind group (`batch["group"]`, the task is known from the
instruction): InfoNCE candidates and inference bank views of the same group only, IIC within each group.
Unconditional, z spends its M values on telling the tasks apart, the largest variation in the views.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .events import KINDS

VIEW_STEPS = 16  # actions per view window (= events post), one per frame


def mlp(*dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


def usage_kl(q):
    """KL(mean_b q || uniform): penalises collapse onto few hypotheses."""
    marginal = q.mean(0).clamp_min(1e-8)
    return (marginal * (marginal.log() + math.log(q.shape[1]))).sum()


def paired_mi(q1, q2):
    """IIC mutual information I(z1; z2) of two posteriors [B, M] of the same events: high when each is
    sharp, they agree, and the shared marginal is uniform."""
    P = (q1.T @ q2) / q1.shape[0]
    P = ((P + P.T) / 2).clamp_min(1e-8)
    return (P * (P.log() - P.sum(1, keepdim=True).log() - P.sum(0, keepdim=True).log())).sum()


def grouped_mi(q1, q2, group=None):
    """I(z1; z2 | group): paired_mi within each group (same task and event kind), weighted by group size.
    Pooled over tasks, the uniform-marginal part of IIC is maximised by spending z on the task; within a
    group it has to come from what differs between that group's events. None = one group."""
    if group is None:
        return paired_mi(q1, q2)
    total = q1.new_zeros(())
    for g in group.unique():
        m = group == g
        if m.sum() > 1:
            total = total + m.float().mean() * paired_mi(q1[m], q2[m])
    return total


def augment_view(x, p_block=0.25, p_tok=0.25, noise=0.1):
    """Random view of a residual view [B, T, N, d]: whole time blocks dropped (at least one kept), tokens
    dropped, Gaussian noise at `noise` x the batch std. Dropped = zero = as predicted by the scene."""
    B, T, N, _ = x.shape
    keep_t = torch.rand(B, T, device=x.device) >= p_block
    keep_t[torch.arange(B, device=x.device), torch.randint(T, (B,), device=x.device)] = True
    keep = keep_t[:, :, None] & (torch.rand(B, T, N, device=x.device) >= p_tok)
    return x * keep[..., None] + noise * x.detach().std() * torch.randn_like(x)


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
    def __init__(self, M=6, d_feat=1152, d_tok=256, n_time=4, n_tok=32, d_ctx=512, d_act=128, tau=0.1, d_assoc=64,
                 scene_act=True):
        super().__init__()
        self.M, self.tau, self.n_tok, self.d_tok, self.n_time, self.scene_act = M, tau, n_tok, d_tok, n_time, scene_act
        self.ctx_feat = nn.Sequential(nn.Linear(2 * d_feat, d_ctx), nn.LayerNorm(d_ctx), nn.GELU())
        self.ctx_act = mlp(16 * 7, d_act, d_act)
        d = d_ctx + d_act
        self.prior = mlp(d, 256, M)
        self.hyp = nn.Parameter(torch.randn(M, 128) * 0.5)
        self.hyp_head = nn.ModuleList([mlp(128 + len(KINDS), 256, d_tok) for _ in range(2)])
        self.view = nn.ModuleList([ViewEncoder(d_tok, n_time, n_tok), ViewEncoder(d_tok, n_time, n_tok, extra=16)])
        d_scene = 2 * d_feat + (VIEW_STEPS * 7 if scene_act else 0)
        self.scene = nn.ModuleList([mlp(d_scene, 256, n_time * n_tok * d_tok) for _ in range(2)])
        self.post_head = mlp(d + 128, 512, n_tok * d_tok)
        self.summary_head = nn.Linear(d, 24)
        self.assoc_head = nn.Linear(2 * d_feat, d_assoc)
        self.register_buffer("bank1", torch.zeros(0, d_tok))
        self.register_buffer("bank2", torch.zeros(0, d_tok))
        self.register_buffer("bank1_group", torch.zeros(0, dtype=torch.long))  # task:kind of each bank view
        self.register_buffer("bank2_group", torch.zeros(0, dtype=torch.long))

    # ------------------------------------------------------------------ pieces
    def context(self, pooled, act):
        return torch.cat([self.ctx_feat(pooled.flatten(1)), self.ctx_act(act.flatten(1))], -1)

    def prior_logits(self, ctx):
        return self.prior(ctx)

    def _with_hyp(self, ctx):
        b = ctx.shape[0]
        return torch.cat([ctx[:, None].expand(b, self.M, -1), self.hyp[None].expand(b, -1, -1)], -1)

    def hyp_embed(self, kind, which):
        """[B, M, d_tok] for event kinds [B] (indices into events.KINDS)."""
        K = len(KINDS)
        x = torch.cat([self.hyp[None].expand(K, -1, -1),
                       torch.eye(K, device=self.hyp.device)[:, None].expand(-1, self.M, -1)], -1)
        return F.normalize(self.hyp_head[which - 1](x), dim=-1)[kind]

    def scene_mean(self, pooled, which, vact=None):
        """[B, n_time, n_tok, d_tok]: the view predicted from its own pre-window and the actions executed during
        it (vact [B, VIEW_STEPS, 7], None = zeros), one prediction per time block; never the outcome."""
        x = pooled.flatten(1)
        if self.scene_act:
            if vact is None:
                vact = x.new_zeros(len(x), VIEW_STEPS, 7)
            x = torch.cat([x, vact.flatten(1)], -1)
        return self.scene[which - 1](x).view(-1, self.n_time, self.n_tok, self.d_tok)

    def view_embed(self, v, pooled, which, w2=None, aug=False, vact=None):
        resid = v - self.scene_mean(pooled, which, vact).detach()
        return self.view[which - 1](augment_view(resid) if aug else resid, w2 if which == 2 else None)

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
        """Reference views for the inference normaliser; pass a random sample of all tasks (an ordered list
        cut at max_size can leave a task out)."""
        g = batch.get("group", torch.zeros(len(batch["v1"]), dtype=torch.long))[:max_size]
        a1, a2 = (batch[k][:max_size] if k in batch else None for k in ("act1", "act2"))
        self.bank1 = self.view_embed(batch["v1"][:max_size], batch["pooled"][:max_size], 1, vact=a1)
        self.bank1_group = g.clone()
        keep = batch["has_v2"][:max_size]
        self.bank2 = self.view_embed(batch["v2"][:max_size][keep], batch["pooled2"][:max_size][keep], 2,
                                     batch["w2"][:max_size][keep], vact=None if a2 is None else a2[keep])
        self.bank2_group = g[keep].clone()

    def _banked_loglik(self, f, u, bank, group=None, bank_group=None):
        own = torch.einsum("bmd,bd->bm", u, f) / self.tau
        ref = torch.einsum("bmd,cd->bcm", u, bank) / self.tau
        return own - _log_mean_exp(ref, _same_group(group, bank_group, len(f), len(bank)))

    def log_posterior_v1(self, pooled, act, v1, kind, group=None, act1=None):
        """group: task:kind ids (data.event_group) to normalise against bank views of the same group; None
        (no task at hand) = the whole bank. act1: actions executed during v1 (for its scene residual)."""
        ctx = self.context(pooled, act)
        ll = self._banked_loglik(self.view_embed(v1, pooled, 1, vact=act1), self.hyp_embed(kind, 1), self.bank1,
                                 group, self.bank1_group)
        return torch.log_softmax(self.prior_logits(ctx) + ll, -1)

    def loglik_v2(self, v2, w2, pooled2, kind_j, group=None, act2=None):
        """Log-likelihood ratio of a re-encounter view (pre-window pooled2, actions act2 during it) under each
        hypothesis of the original event j (of kind_j, group = its task:kind)."""
        return self._banked_loglik(self.view_embed(v2, pooled2, 2, w2, vact=act2), self.hyp_embed(kind_j, 2), self.bank2,
                                   group, self.bank2_group)


def _same_group(group, cand_group, n, n_cand):
    """[n, n_cand] mask of candidates in the event's own group; rows with none fall back to all candidates."""
    if group is None or cand_group is None or len(cand_group) != n_cand:
        return None
    mask = group[:, None] == cand_group[None, :]
    return mask | ~mask.any(1, keepdim=True)


def _log_mean_exp(S, mask=None):
    """log mean_c exp S[b, c, m] over candidates c (only those in mask[b, c])."""
    if mask is None:
        return torch.logsumexp(S, 1) - math.log(S.shape[1])
    S = S.masked_fill(~mask[..., None], float("-inf"))
    return torch.logsumexp(S, 1) - mask.sum(1, keepdim=True).float().log()


def _nce(S, group=None):
    """Per-hypothesis InfoNCE log-ratio of each event's own view against the in-batch candidates of its own
    task:kind group: against all candidates, a hypothesis gains most by telling the tasks apart."""
    own = torch.diagonal(S, dim1=0, dim2=1).T  # [B, M]
    return own - _log_mean_exp(S, _same_group(group, group, S.shape[0], S.shape[1]))


def em_step(model, batch, lam_g=1.0, lam_u=0.1, lam_a=0.1, lam_i=1.0, lam_i1=0.0):
    ctx = model.context(batch["pooled"], batch["act"])
    prior = torch.log_softmax(model.prior_logits(ctx), -1)
    a1, a2 = batch.get("act1"), batch.get("act2")  # actions during each view, for its scene residual
    f1 = model.view_embed(batch["v1"], batch["pooled"], 1, vact=a1)
    f2 = model.view_embed(batch["v2"], batch["pooled2"], 2, batch["w2"], vact=a2)
    group = batch.get("group")  # task:kind; None = one group
    ll1 = _nce(model.scores(f1, model.hyp_embed(batch["kind"], 1)), group)
    ll2 = _nce(model.scores(f2, model.hyp_embed(batch["kind"], 2)), group)
    has2 = batch["has_v2"].float()[:, None]
    joint = prior + ll1 + has2 * ll2
    q = joint.softmax(-1).detach()  # posterior, as used at inference
    # E-step targets come from the views only: the prior sees the context, which can carry a balanced
    # nuisance (e.g. whether the outcome region is visible); fed back into the targets it locks in a split
    # on that nuisance. The prior is distilled from the view posterior instead.
    qv = (ll1 + has2 * ll2).softmax(-1).detach()
    n2 = has2.sum().clamp_min(1.0)
    prior_nll = -(qv * prior).sum(-1).mean()
    nce1 = -(qv * ll1).sum(-1).mean()
    nce2 = -(qv * ll2 * has2).sum() / n2
    target = batch["v1"].mean(1)
    scene = ((model.scene_mean(batch["pooled"], 1, a1) - batch["v1"]) ** 2).mean() + (
        ((model.scene_mean(batch["pooled2"], 2, a2) - batch["v2"]) ** 2).mean((1, 2, 3)) * has2[:, 0]).sum() / n2
    recon = (q[..., None, None] * (model.predict_post(ctx) - target[:, None]) ** 2).mean((-1, -2)).sum(-1).mean()
    ukl = usage_kl(joint.softmax(-1))
    pair = batch["has_v2"]
    mi = (grouped_mi(ll1[pair].softmax(-1), ll2[pair].softmax(-1), None if group is None else group[pair])
          if pair.sum() > 1 else ll1.new_zeros(()))
    mi1 = ll1.new_zeros(())
    if lam_i1:
        u1 = model.hyp_embed(batch["kind"], 1)
        qa, qb = (_nce(model.scores(model.view_embed(batch["v1"], batch["pooled"], 1, aug=True, vact=a1), u1), group).softmax(-1)
                  for _ in range(2))
        mi1 = grouped_mi(qa, qb, group)
    a1, a2 = model.assoc_embed(batch["pooled"]), model.assoc_embed(batch["pooled2"])
    logits = a1 @ a2.T / model.tau
    labels = torch.arange(len(a1), device=a1.device)
    assoc = (F.cross_entropy(logits, labels, reduction="none") * has2[:, 0]).sum() / n2
    loss = prior_nll + nce1 + nce2 + lam_g * (recon + scene) + lam_u * ukl - lam_i * mi - lam_i1 * mi1 + lam_a * assoc
    return {"loss": loss, "prior_nll": prior_nll.detach(), "nce1": nce1.detach(), "nce2": nce2.detach(),
            "recon": recon.detach(), "scene": scene.detach(), "usage_kl": ukl.detach(), "view_mi": mi.detach(),
            "v1_mi": mi1.detach(), "assoc": assoc.detach(), "q": q}
