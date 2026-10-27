"""The POL producer: gripper events -> ledger entries with posteriors, resolved at re-encounter.

An entry is written once the post-event window is complete (so it only uses frames up to now).
Before writing, the new event's view is scored as a re-encounter view for every pending entry
within rho, and those entries are updated (late explaining-away).
"""

from collections import deque

import numpy as np
import torch

from ..events import KINDS
from ..ledger import Entry, Ledger
from .base import MemoryProducer, register

PRE, POST, N_TIME = 8, 16, 4


def _blocks(frames, n_time, shape):
    """Average token frames into n_time temporal blocks; missing frames contribute nothing."""
    out = np.zeros((n_time, *shape), np.float32)
    if not frames:
        return out
    for i, chunk in enumerate(np.array_split(np.arange(len(frames)), n_time)):
        idx = chunk if len(chunk) else [len(frames) - 1]
        out[i] = np.mean([frames[k] for k in idx], 0)
    return out


@register("POL")
class PendingOutcomeLedger(MemoryProducer):
    def reset(self):
        model = self.assets["outcome_model"]
        c = self.cfg
        omega0 = float(self.assets.get("omega0", 0.0)) if c.get("use_omega", True) else 0.0
        self.ledger = Ledger(self.K, model.M, eta=float(c.get("eta", 0.3)), omega0=omega0,
                             rho=float(c.get("rho", 0.06)), beta=float(c.get("beta", 0.0)),
                             eviction=c.get("eviction", "entropy"))
        self.hist = deque(maxlen=PRE + POST + 8)
        self.open = []
        self.t = 0

    def observe(self, o):
        self.t = o.t
        self.hist.append(o)
        if o.event is not None:
            self.open.append(o.event)
        for ev in [e for e in self.open if o.t >= e.t + POST - 1]:
            self._close(ev)
            self.open.remove(ev)

    def tokens(self):
        return self.ledger.featurize(self.t)

    # ------------------------------------------------------------------
    def _window(self, ev):
        pre = [s for s in self.hist if ev.t - PRE <= s.t < ev.t]
        post = [s for s in self.hist if ev.t <= s.t < ev.t + POST]
        pooled = np.mean([s.pooled for s in pre], 0) if pre else post[0].pooled
        act = np.zeros((16, 7), np.float32)
        span = [s for s in self.hist if ev.t - 8 <= s.t < ev.t + 8]
        for s in span:
            act[s.t - (ev.t - 8)] = s.action
        if self.cfg.get("call_frames_only", False):
            post_frames = [s.tokens for s in post if s.tokens is not None and self.is_call(s.t)]
        else:
            post_frames = [s.tokens for s in post if s.tokens is not None]
        model = self.assets["outcome_model"]
        v = _blocks(post_frames, N_TIME, (model.n_tok, model.d_tok))
        w = np.zeros(16, np.float32)
        for i, s in enumerate(post[:16]):
            w[i] = s.proprio[6]
        return pooled.astype(np.float32), act, v, w

    @torch.no_grad()
    def _close(self, ev):
        model = self.assets["outcome_model"]
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)[None]  # noqa: E731
        pooled, act, v, w = self._window(ev)
        assoc = model.assoc_embed(tt(pooled))[0].numpy()
        if self.cfg.get("resolve", True) and self.ledger.entries:
            weights = self.ledger.association(ev.x_ee, assoc)
            if weights.any():
                kj = torch.as_tensor([e.kind for e in self.ledger.entries])
                n = len(self.ledger.entries)
                ll = model.loglik_v2(tt(v).expand(n, -1, -1, -1), tt(w).expand(n, -1), tt(pooled).expand(n, -1, -1), kj).numpy()
                self.ledger.resolve(weights, ll)
        ctx = model.context(tt(pooled), tt(act))
        kind = KINDS.index(ev.kind)
        logp = model.log_posterior_v1(tt(pooled), tt(act), tt(v), torch.as_tensor([kind]))[0].numpy()
        omega = float(model.omega(ctx, self.assets["sigma2"])[0]) if self.cfg.get("use_omega", True) else 1.0
        self.ledger.write(Entry(x=ev.x_ee, kind=kind, logp=logp, omega=omega, t=ev.t,
                                summary=model.summary(ctx)[0].numpy(), assoc=assoc,
                                extra={"pooled": pooled, "act": act}))
