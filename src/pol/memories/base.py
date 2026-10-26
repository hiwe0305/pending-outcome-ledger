"""Common contract of every memory variant: observe one step at a time, emit K tokens of ENTRY_DIM.

A producer sees exactly what the policy could see at that step (features, proprio, executed action,
the gripper event detected from proprio). StepObs.log is evaluator-only and may be read solely by
the Oracle variant; label-trained heads (Markov, AGM-style) read logs at their own training time,
never inside observe().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..events import Event
from ..ledger import ENTRY_DIM

REGISTRY = {}


@dataclass
class StepObs:
    t: int
    pooled: np.ndarray  # [2, 1152] mean-pooled SigLIP per camera
    tokens: np.ndarray | None  # [32, 256] PCA patch tokens, None where not stored
    proprio: np.ndarray  # [7] xyz, rpy, gripper width
    action: np.ndarray  # [7] action executed at t
    event: Event | None
    log: dict | None = None  # evaluator-only


def memory_bytes(K):
    return K * ENTRY_DIM * 4


def register(name):
    def wrap(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


def make_memory(name, cfg, assets):
    if name not in REGISTRY:
        raise KeyError(f"unknown memory variant {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, assets)


def project50(pooled, assets):
    """Fixed linear projection (PCA fitted on train) of both cameras' pooled features to 50 numbers."""
    return ((np.asarray(pooled, np.float64).reshape(-1) - assets["proj_mean"]) @ assets["proj50"]).astype(np.float32)


class MemoryProducer(ABC):
    name = "base"

    def __init__(self, cfg, assets):
        self.cfg, self.assets = cfg, assets
        self.K = int(cfg["K"])
        self.exec_steps = int(cfg.get("exec_steps", 8))
        self.reset()

    @property
    def bytes(self):
        return memory_bytes(self.K)

    def is_call(self, t):
        return t % self.exec_steps == 0

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def observe(self, o: StepObs): ...

    @abstractmethod
    def tokens(self) -> tuple[np.ndarray, np.ndarray]: ...

    def _empty(self):
        return np.zeros((self.K, ENTRY_DIM), np.float32), np.zeros(self.K, bool)


class SlotBuffer:
    """FIFO of at most K vectors of ENTRY_DIM numbers."""

    def __init__(self, K):
        self.K = K
        self.rows = []

    def push(self, row):
        self.rows = (self.rows + [np.asarray(row, np.float32)])[-self.K:]

    def tokens(self):
        feats = np.zeros((self.K, ENTRY_DIM), np.float32)
        mask = np.zeros(self.K, bool)
        for i, r in enumerate(self.rows):
            feats[i, : len(r)] = r
            mask[i] = True
        return feats, mask
