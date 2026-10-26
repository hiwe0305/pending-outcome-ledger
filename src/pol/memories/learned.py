"""Baseline memories with a learned component: KEM, MU, AURA, StateMem, CheckVLA(+FIFO), AGM.

Each module comes from assets (trained in Stage A, see pol.baseline_models) and runs under no_grad.
All write the same 50-number rows into K slots, so memory bytes match POL.
"""

from collections import deque

import numpy as np
import torch

from .base import MemoryProducer, SlotBuffer, project50, register


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]


@register("KEM")
class KeyframeMemory(MemoryProducer):
    def reset(self):
        self.buf = SlotBuffer(self.K)

    @torch.no_grad()
    def observe(self, o):
        x = project50(o.pooled, self.assets)
        if torch.sigmoid(self.assets["kem_scorer"](_t(x)))[0] > float(self.assets.get("kem_threshold", 0.5)):
            self.buf.push(x)

    def tokens(self):
        return self.buf.tokens()


@register("MU")
class RecurrentTokens(MemoryProducer):
    def reset(self):
        self.h = None

    @torch.no_grad()
    def observe(self, o):
        if self.is_call(o.t):
            net = self.assets["mu_net"]
            h = torch.zeros(1, net.hidden) if self.h is None else self.h
            self.h = net.step(_t(project50(o.pooled, self.assets)), h)

    @torch.no_grad()
    def tokens(self):
        if self.h is None:
            return self._empty()
        return self.assets["mu_net"].read(self.h)[0].numpy().astype(np.float32), np.ones(self.K, bool)


@register("AURA")
class ActionGatedMemory(MemoryProducer):
    def reset(self):
        self.buf = SlotBuffer(self.K)
        self.last = None

    @torch.no_grad()
    def observe(self, o):
        x = project50(o.pooled, self.assets)
        a = self.assets["aura_net"](_t(x))[0].numpy()
        if self.last is None or np.linalg.norm(a - self.last) > float(self.assets.get("aura_threshold", 0.5)):
            self.buf.push(x)
            self.last = a

    def tokens(self):
        return self.buf.tokens()


@register("StateMem")
class StateMemMemory(MemoryProducer):
    def reset(self):
        self.m = None

    @torch.no_grad()
    def observe(self, o):
        m = torch.zeros(1, 50) if self.m is None else self.m
        self.m = self.assets["statemem_net"].step(m, _t(project50(o.pooled, self.assets)))

    def tokens(self):
        feats, mask = self._empty()
        if self.m is not None:
            feats[0] = self.m[0].numpy()
            mask[0] = True
        return feats, mask


@register("CheckVLA")
class CheckVLAMemory(MemoryProducer):
    """Point predictor of features `horizon` steps ahead; a standardized discrepancy above the
    conformal quantile flags the chunk and stores its frame. Slot 0 carries the current risk."""

    def reset(self):
        self.buf = SlotBuffer(self.K - 1)
        self.hist = deque(maxlen=int(self.cfg.get("horizon", 8)) + 1)
        self.risk = None

    @torch.no_grad()
    def observe(self, o):
        h = int(self.cfg.get("horizon", 8))
        x = project50(o.pooled, self.assets)
        self.hist.append((x, np.asarray(o.action, np.float32)))
        if not self.is_call(o.t) or len(self.hist) <= h:
            return
        x0 = self.hist[0][0]
        acts = np.stack([a for _, a in list(self.hist)[:h]])
        pred = self.assets["check_net"](_t(x0), _t(acts))[0].numpy()
        r = (np.linalg.norm(pred - x) - self.assets["check_mu"]) / self.assets["check_sd"]
        flag = r > self.assets["conformal_q"]
        if flag:
            self.buf.push(x)
        if self.cfg.get("fifo_occluded", False) and np.linalg.norm(x - x0) < self.assets["still_threshold"]:
            self.buf.push(x)  # chunk with no visible change: kept as possibly unobserved
        self.risk = np.array([r, float(flag)], np.float32)

    def tokens(self):
        bank, bmask = self.buf.tokens()
        feats, mask = self._empty()
        feats[1:], mask[1:] = bank, bmask
        if self.risk is not None:
            feats[0, :2] = self.risk
            mask[0] = True
        return feats, mask


@register("AGM")
class AchievementPointer(MemoryProducer):
    """Subgoal pointer advanced only by verified achievement: grasps by coherent lift, placements by
    a label-trained verify head on the post-release frame (AGM-style)."""

    WAIT = 8

    def reset(self):
        n = int(self.cfg.get("n_objects", 3))
        self.subgoals = self.cfg.get("subgoals") or ["grasp", "place"] * n
        self.pointer, self.verified, self.last = 0, 0, 0.0
        self.pending = []  # (event, z at event)

    @torch.no_grad()
    def observe(self, o):
        if o.event is not None and o.event.kind in ("G+", "R+"):
            self.pending.append((o.event, float(o.proprio[2])))
        for ev, z0 in [p for p in self.pending if o.t >= p[0].t + self.WAIT]:
            self.pending.remove((ev, z0))
            if ev.kind == "G+":
                ok = (o.proprio[2] - z0) > float(self.cfg.get("lift_threshold", 0.01))
                want = "grasp"
            else:
                ok = torch.sigmoid(self.assets["verify_head"](_t(project50(o.pooled, self.assets))))[0].item() > 0.5
                want = "place"
            self.last = float(ok)
            current = self.subgoals[min(self.pointer, len(self.subgoals) - 1)]
            if ok and current == want:
                self.pointer, self.verified = self.pointer + 1, self.verified + 1
            elif not ok and want == "place":
                self.pointer = max(0, self.pointer - 1)  # back to grasping the same object

    def tokens(self):
        feats, mask = self._empty()
        n = len(self.subgoals)
        current = self.subgoals[min(self.pointer, n - 1)]
        feats[0, :5] = [self.pointer / n, self.verified / n, self.last, current == "grasp", current == "place"]
        mask[0] = True
        return feats, mask
