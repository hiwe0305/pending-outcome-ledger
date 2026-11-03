"""Pending-Outcome Ledger: fixed-capacity memory of action outcomes kept as posteriors.

Each entry is one gripper event with a log posterior over M learned outcome hypotheses and an
observability score. A later event near the same place adds the log-likelihood of its view
(late explaining-away). The ledger is read by pi0 as K tokens of ENTRY_DIM numbers:
  PE(x) 12 | kind one-hot 3 | probabilities padded to M_MAX 8 | omega 1 | entropy 1 | age 1 | summary 24
The age feature carries the consumed flag in its sign (negative = a decision already used the entry).
"""

from dataclasses import dataclass

import numpy as np

ENTRY_DIM, M_MAX, SUMMARY_DIM, ASSOC_DIM = 50, 8, 24, 64
AGE_UNIT = 1000.0
_FREQS = 2 * np.pi * np.array([1.0, 4.0])


def positional_encoding(x):
    phase = np.asarray(x, float)[:, None] * _FREQS[None]
    return np.stack([np.sin(phase), np.cos(phase)], -1).ravel()


def _normalize(logp):
    return logp - np.logaddexp.reduce(logp)


@dataclass
class Entry:
    x: np.ndarray
    kind: int
    logp: np.ndarray
    omega: float
    t: int
    summary: np.ndarray
    assoc: np.ndarray
    extra: dict | None = None  # producer-side inputs needed to score a later view (not featurized)
    consumed: bool = False  # a decision has used this entry; it is then the first to be evicted

    @property
    def probs(self):
        return np.exp(_normalize(self.logp))

    @property
    def entropy(self):
        p = self.probs
        return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


class Ledger:
    def __init__(self, K, M, eta=0.3, omega0=0.0, rho=0.06, beta=0.0, eviction="value",
                 p_null=0.1):
        if not 1 <= M <= M_MAX:
            raise ValueError(f"M must be in 1..{M_MAX}")
        self.K, self.M, self.eta, self.omega0, self.rho, self.beta = K, M, eta, omega0, rho, beta
        if eviction not in ("value", "entropy", "fifo"):
            raise ValueError(f"unknown eviction {eviction!r}")
        self.eviction, self.p_null = eviction, p_null
        self.reset()

    def reset(self):
        self.entries = []

    def write(self, e):
        e.logp = _normalize(np.asarray(e.logp, float))
        if len(self.entries) < self.K:
            self.entries.append(e)
            return len(self.entries) - 1
        slot = self.evict_index()  # evict before writing, so the new entry is never the victim
        self.entries[slot] = e
        return slot

    def association(self, x, assoc):
        """Distribution over entries for one re-encounter, summing to 1 - p_null (the rest is the null option).

        Entries within rho are weighted by the spatial kernel times a context term when beta > 0 and
        normalised; with no entry within rho every weight is zero.
        """
        w = np.zeros(len(self.entries))
        for i, e in enumerate(self.entries):
            d2 = float(np.sum((np.asarray(x, float) - e.x) ** 2))
            if d2 <= self.rho ** 2:
                cos = float(np.dot(assoc, e.assoc) / (np.linalg.norm(assoc) * np.linalg.norm(e.assoc) + 1e-8))
                w[i] = np.exp(-d2 / (2 * self.rho ** 2) + self.beta * (cos - 1.0))
        total = w.sum()
        return w * ((1.0 - self.p_null) / total) if total > 0 else w

    def resolve(self, weights, loglik):
        """Mixture update: p_j <- (1 - w_j) p_j + w_j normalise(p_j exp(ll_j)), so one view is one unit of evidence."""
        loglik = np.broadcast_to(np.asarray(loglik, float), (len(self.entries), self.M))
        for e, w, ll in zip(self.entries, weights, loglik):
            if w > 0:
                logp = _normalize(e.logp)
                e.logp = _normalize(np.logaddexp(np.log1p(-w) + logp if w < 1 else -np.inf,
                                                 np.log(w) + _normalize(logp + ll)))

    def mark_consumed(self, i):
        self.entries[i].consumed = True

    def status(self, i):
        e = self.entries[i]
        if e.entropy < self.eta:
            return "resolved"
        return "pending_unobservable" if e.omega < self.omega0 else "pending_ambiguous"

    def evict_index(self):
        if self.eviction == "fifo":
            return min(range(len(self.entries)), key=lambda i: self.entries[i].t)
        if self.eviction == "entropy":
            return self._evict_by_entropy()
        return self._evict_by_value()

    def keep_value(self, i):
        """0 once consumed; otherwise max_z p(z): a resolved verdict not yet used outranks an ambiguous one."""
        e = self.entries[i]
        return 0.0 if e.consumed else float(e.probs.max())

    def _evict_by_value(self):
        """Drop the entry least worth keeping (consumed first); among equal values, the oldest."""
        values = [self.keep_value(i) for i in range(len(self.entries))]
        lowest = min(values)
        ties = [i for i, v in enumerate(values) if v - lowest < 1e-6]
        return min(ties, key=lambda i: self.entries[i].t)

    def _evict_by_entropy(self):
        """Drop the most resolved entry (lowest entropy); among equal entropies, the oldest.

        Pending entries are never dropped in favour of age alone: at small K the ledger keeps the
        open questions, which is what the capacity sweep is meant to stress.
        """
        entropies = [e.entropy for e in self.entries]
        lowest = min(entropies)
        ties = [i for i, h in enumerate(entropies) if h - lowest < 1e-6]
        return min(ties, key=lambda i: self.entries[i].t)

    def featurize(self, t_now):
        feats = np.zeros((self.K, ENTRY_DIM), np.float32)
        mask = np.zeros(self.K, bool)
        for i, e in enumerate(self.entries):
            probs = np.zeros(M_MAX)
            probs[: self.M] = e.probs
            kind = np.eye(3)[e.kind]
            age = np.copysign((t_now - e.t) / AGE_UNIT, -1.0 if e.consumed else 1.0)
            feats[i] = np.concatenate([positional_encoding(e.x), kind, probs, [e.omega, e.entropy, age], e.summary])
            mask[i] = True
        return feats, mask
