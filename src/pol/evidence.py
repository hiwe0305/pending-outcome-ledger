"""POL-V evidence heads (plan 2.2-2.3): few-label outcome classifiers per view, a prior over outcomes from the
context, and omega = the expected information gain of view 1, all on frozen features.

q_k(z | v_k, c) is trained discriminatively (CE, label smoothing) on <= n_labels events per task, an ensemble of E
members averaged in probability space. The ledger adds evidence as the posterior/prior ratio
l_k(z) = log q_k(z|v_k,c) - log pi_k(z), pi_k = smoothed class frequencies of head k's labelled set. The prior
p(z|c) is distilled from q_1 on all events; omega-hat regresses the realised KL(q_1 || p) from the context and the
first 2 post-event frames only (never later frames: an occluder lowered later must not leak in).
"""

import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .labels import OUTCOMES
from .outcome_model import ViewEncoder, mlp

CLASSES = {f: tuple(o for o in OUTCOMES[f] if o != "stuck") for f in OUTCOMES}  # stuck is held out
N_EARLY = 2  # post-event frames omega may see


def item_id(it):
    return f"{it['task']}:{it['seed']}:{it['t']}"


def sample_labelled(items, n_per_task, seed, exclude=()):
    """Stratified by outcome class within each task: >= 1 per class present, the rest proportional to the class
    counts (largest remainders). Items without a usable outcome (None, stuck) are never picked."""
    rng = np.random.default_rng(seed)
    skip, out = set(exclude), []
    for task in sorted({it["task"] for it in items}):
        by = {}
        for it in items:
            if it["task"] == task and it["outcome"] is not None and it["outcome"] < len(CLASSES[it["family"]]) \
                    and item_id(it) not in skip:
                by.setdefault(it["outcome"], []).append(it)
        n_tot = sum(map(len, by.values()))
        n = min(n_per_task, n_tot)
        if not n:
            continue
        cls = sorted(by)
        exact = {c: n * len(by[c]) / n_tot for c in cls}
        quota = {c: min(len(by[c]), max(1, int(exact[c]))) for c in cls}
        for c in sorted(cls, key=lambda c: int(exact[c]) - exact[c]):  # largest remainders first
            if sum(quota.values()) >= n:
                break
            quota[c] += quota[c] < len(by[c])
        while sum(quota.values()) < n:  # small classes saturated: fill from the rest
            c = next(c for c in cls if quota[c] < len(by[c]))
            quota[c] += 1
        while sum(quota.values()) > n:  # more classes than labels
            quota[max(cls, key=quota.get)] -= 1
        for c in cls:
            out += [by[c][i] for i in rng.permutation(len(by[c]))[:quota[c]]]
    return out


class _Head(nn.Module):
    """One ensemble member: view encoder (no scene residual) + context -> per-task logits, plus a linear readout
    of the raw tokens. With 32 labels the encoder alone often latches onto a nuisance 3x larger than the outcome
    (synthetic test, 1 seed in 3); the linear path does not."""

    def __init__(self, n_classes, d_feat, d_tok, n_time, n_tok, d_ctx, extra):
        super().__init__()
        self.enc = ViewEncoder(d_tok, n_time, n_tok, extra=extra)
        self.ctx = mlp(2 * d_feat + (16 * 7 if not extra else 0), d_ctx, d_ctx)
        self.out = nn.ModuleDict({t: nn.Linear(d_tok + d_ctx, z) for t, z in n_classes.items()})
        self.lin = nn.ModuleDict({t: nn.Linear(n_time * n_tok * d_tok, z) for t, z in n_classes.items()})

    def forward(self, task, v, c, extra=None):
        return self.out[task](torch.cat([16.0 * self.enc(v, extra), self.ctx(c)], -1)) + self.lin[task](v.flatten(1))


class EvidenceHeads(nn.Module):
    def __init__(self, n_classes, E=5, d_tok=256, d_feat=1152, n_time=4, n_tok=32, d_ctx=128):
        super().__init__()
        self.n_classes = dict(n_classes)
        hk = (self.n_classes, d_feat, d_tok, n_time, n_tok, d_ctx)
        self.h1 = nn.ModuleList([_Head(*hk, 0) for _ in range(E)])
        self.h2 = nn.ModuleList([_Head(*hk, 16) for _ in range(E)])
        self.prior_ctx = mlp(2 * d_feat + 16 * 7, 256, d_ctx)
        self.prior_out = nn.ModuleDict({t: nn.Linear(d_ctx, z) for t, z in self.n_classes.items()})
        self.omega_net = nn.ModuleDict({t: mlp(2 * d_feat + 16 * 7 + N_EARLY * 2 * d_feat, 256, 1) for t in self.n_classes})
        for t, z in self.n_classes.items():
            for k in (1, 2):
                self.register_buffer(f"pi{k}_{t}", torch.full((z,), 1.0 / z))
        self.register_buffer("temp", torch.ones(2))  # per-head temperature (calibrate)

    @staticmethod
    def _c(pooled, act):
        return torch.cat([pooled.flatten(1), act.flatten(1)], -1)

    def member_logits(self, k, task, v, c, w2=None):
        """[E, B, Z] uncalibrated logits of head k."""
        heads = self.h1 if k == 1 else self.h2
        return torch.stack([h(task, v, c, w2) for h in heads])

    def probs(self, k, task, v, c, w2=None):
        return (self.member_logits(k, task, v, c, w2) / self.temp[k - 1]).softmax(-1).mean(0)

    def pi(self, k, task):
        return getattr(self, f"pi{k}_{task}")

    def q_v1(self, task, pooled, act, v1):
        return self.probs(1, task, v1, self._c(pooled, act))

    def loglik_v1(self, task, pooled, act, v1):
        return self.q_v1(task, pooled, act, v1).clamp_min(1e-8).log() - self.pi(1, task).log()

    def loglik_v2(self, task, v2, w2, pooled2):
        return self.probs(2, task, v2, pooled2.flatten(1), w2).clamp_min(1e-8).log() - self.pi(2, task).log()

    def prior_logits(self, task, pooled, act):
        return self.prior_out[task](self.prior_ctx(self._c(pooled, act)))

    def omega(self, task, pooled, act, early_frames):
        """Expected KL(q(z|v1,c) || p(z|c)) from the context and early_frames [B, 2, 2, d_feat] (frames t, t+1)."""
        x = torch.cat([self._c(pooled, act), early_frames[:, :N_EARLY].flatten(1)], -1)
        return F.softplus(self.omega_net[task](x))[:, 0]

    def realised_kl(self, task, b):
        q = self.q_v1(task, b["pooled"], b["act"], b["v1"]).clamp_min(1e-8)
        logp = torch.log_softmax(self.prior_logits(task, b["pooled"], b["act"]), -1)
        return (q * (q.log() - logp)).sum(-1)

    @torch.no_grad()
    def calibrate(self, val_batch):
        """Per-head temperature minimising NLL on labelled val events: a dict (one task) or a list of dicts with
        keys task, label and the collate_events views."""
        grid = torch.logspace(-1.5, 1.5, 61)
        for k in (1, 2):
            nll = torch.zeros(len(grid))
            for b in val_batch if isinstance(val_batch, list) else [val_batch]:
                m = torch.ones_like(b["label"], dtype=torch.bool) if k == 1 else b["has_v2"]
                if not m.any():
                    continue
                L = (self.member_logits(1, b["task"], b["v1"], self._c(b["pooled"], b["act"])) if k == 1 else
                     self.member_logits(2, b["task"], b["v2"], b["pooled2"].flatten(1), b["w2"]))[:, m]
                for i, T in enumerate(grid):
                    p = (L / T).softmax(-1).mean(0).clamp_min(1e-8)
                    nll[i] += -p.log().gather(1, b["label"][m][:, None]).sum()
            if nll.any():
                self.temp[k - 1] = grid[nll.argmin()]


def set_class_prior(model, task, labels, has_v2):
    """pi_k = class frequencies of head k's labelled set, add-one smoothed so no class is 0."""
    z = model.n_classes[task]
    for k, lab in ((1, labels), (2, labels[has_v2])):
        model.pi(k, task).copy_((torch.bincount(lab, minlength=z).float() + 1) / (len(lab) + z))


def fit_evidence(model, labelled, unlabelled, steps=300, lr=3e-4, label_smoothing=0.05, batch_size=64, seed=0,
                 val=None, log=None):
    """labelled / unlabelled: {task: batch} (collate_events + 'label' [B]; 'early' for unlabelled). Three phases
    of `steps`: CE heads (each member on all labels), prior distilled from q_1, omega-hat on the realised KL. q_1
    and the KL targets are computed once per phase (fixed, detached). val: labelled val batches for calibrate,
    fitted after the heads so the prior and omega see calibrated q."""
    g = torch.Generator().manual_seed(seed)
    for t, b in labelled.items():
        set_class_prior(model, t, b["label"], b["has_v2"])

    def run(name, mods, loss_fn, held=False):
        """held: early stopping on the other half of the unlabelled events (the context MLPs memorise q / KL of
        the events they fit, and a confident prior from an uninformative context inflates every KL)."""
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        best, state = math.inf, None
        for step in range(steps):
            loss = sum(loss_fn(t, 0) for t in labelled)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if log is not None:
                log(step, {f"{name}/loss": float(loss.detach())})
            if held and (step % 10 == 9 or step == steps - 1):
                with torch.no_grad():
                    h = float(sum(loss_fn(t, 1) for t in labelled))
                if h < best:
                    best, state = h, [copy.deepcopy(m.state_dict()) for m in mods]
        for m, st in zip(mods, state or []):
            m.load_state_dict(st)

    def heads_loss(t, _):
        b = labelled[t]
        c, y, m = model._c(b["pooled"], b["act"]), b["label"], b["has_v2"]
        loss = sum(F.cross_entropy(l, y, label_smoothing=label_smoothing) for l in model.member_logits(1, t, b["v1"], c))
        if m.any():
            L2 = model.member_logits(2, t, b["v2"][m], b["pooled2"][m].flatten(1), b["w2"][m])
            loss = loss + sum(F.cross_entropy(l, y[m], label_smoothing=label_smoothing) for l in L2)
        return loss

    # cross-fit: the prior learns on one half, omega's KL targets come from the other (a prior that has seen an
    # event can memorise its q from the context and drive its KL to 0)
    halves = {t: torch.randperm(len(unlabelled[t]["pooled"]), generator=g).chunk(2) for t in labelled}

    def pick(t, h, train):
        idx = halves[t][h] if len(halves[t]) > 1 else halves[t][0]
        return idx[torch.randint(len(idx), (batch_size,), generator=g)] if train else idx

    run("heads", [model.h1, model.h2], heads_loss)
    if val:
        model.calibrate(val)
    model.eval()
    U = unlabelled
    with torch.no_grad():
        q = {t: torch.cat([model.q_v1(t, U[t]["pooled"][i:i + 256], U[t]["act"][i:i + 256], U[t]["v1"][i:i + 256])
                           for i in range(0, len(U[t]["pooled"]), 256)]) for t in labelled}

    def prior_loss(t, held):  # fit on half 0, stop on half 1
        i = pick(t, held, not held)
        return -(q[t][i] * torch.log_softmax(model.prior_logits(t, U[t]["pooled"][i], U[t]["act"][i]), -1)).sum(-1).mean()

    run("prior", [model.prior_ctx, model.prior_out], prior_loss, held=True)
    with torch.no_grad():
        kl = {t: (q[t] * (q[t].clamp_min(1e-8).log() - torch.log_softmax(model.prior_logits(t, U[t]["pooled"], U[t]["act"]), -1))).sum(-1)
              for t in labelled}

    def omega_loss(t, held):  # fit on half 1 (KL targets from a prior that never saw it), stop on half 0
        i = pick(t, 1 - held, not held)
        return F.mse_loss(model.omega(t, U[t]["pooled"][i], U[t]["act"][i], U[t]["early"][i]), kl[t][i])

    run("omega", [model.omega_net], omega_loss, held=True)
    return model.eval()


def ece(p, y, bins=10):
    conf, pred = p.max(1)
    edges, err = torch.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.float().mean() * abs(float((pred[m] == y[m]).float().mean() - conf[m].mean()))
    return float(err)
