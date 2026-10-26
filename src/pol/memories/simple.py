"""Memory variants without learned writing: B0, Frames, Random, Markov, Oracle."""

import numpy as np

from ..events import KINDS
from ..ledger import M_MAX, Entry, Ledger
from .base import MemoryProducer, SlotBuffer, project50, register


@register("B0")
class NoMemory(MemoryProducer):
    def reset(self):
        pass

    def observe(self, o):
        pass

    def tokens(self):
        return self._empty()


@register("Frames")
class FrameFIFO(MemoryProducer):
    """The K most recent policy-call frames."""

    def reset(self):
        self.buf = SlotBuffer(self.K)

    def observe(self, o):
        if self.is_call(o.t):
            self.buf.push(project50(o.pooled, self.assets))

    def tokens(self):
        return self.buf.tokens()


@register("Random")
class RandomWrite(MemoryProducer):
    """Frames written with probability write_prob per step, FIFO of K."""

    def reset(self):
        self.buf = SlotBuffer(self.K)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 0)))

    def observe(self, o):
        if self.rng.random() < float(self.cfg.get("write_prob", 0.1)):
            self.buf.push(project50(o.pooled, self.assets))

    def tokens(self):
        return self.buf.tokens()


@register("Markov")
class MarkovChecker(MemoryProducer):
    """Memoryless: one token with P(failure) from a label-trained classifier on the current frame + proprio."""

    def reset(self):
        self.row = None

    def observe(self, o):
        p = float(self.assets["markov"](o.pooled, o.proprio))
        self.row = np.concatenate([[p, 1.0 - p], o.proprio]).astype(np.float32)

    def tokens(self):
        feats, mask = self._empty()
        if self.row is not None:
            feats[0, : len(self.row)] = self.row
            mask[0] = True
        return feats, mask


@register("Oracle")
class OracleOutcome(MemoryProducer):
    """Upper bound: the true outcome of every event, in the POL token layout."""

    def reset(self):
        self.ledger = Ledger(self.K, M_MAX)
        self.t = 0

    def observe(self, o):
        self.t = o.t
        if o.event is None:
            return
        if o.log is None:
            raise ValueError("Oracle needs the evaluator log at event steps")
        logp = np.full(M_MAX, -30.0)
        logp[int(o.log["event_outcome"])] = 0.0
        self.ledger.write(Entry(x=o.event.x_ee, kind=KINDS.index(o.event.kind), logp=logp, omega=1.0, t=o.t,
                                summary=np.zeros(24), assoc=np.zeros(64)))

    def tokens(self):
        return self.ledger.featurize(self.t)
