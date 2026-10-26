"""Gripper-load event detection (after AGM), event windows and spatial re-encounter.

Events come from the gripper width only (proprio[6]), so they are available at train and test
time without simulator labels:
  G+  stable loaded grasp (width settles between empty and open)
  R+  release from a loaded state
  R0  release from an empty (closed-on-nothing) state
"""

from dataclasses import dataclass

import numpy as np

KINDS = ("G+", "R+", "R0")
OPEN, LOADED, EMPTY = "open", "loaded", "empty"


@dataclass(frozen=True)
class Event:
    t: int
    kind: str
    x_ee: np.ndarray


@dataclass
class FsmThresholds:
    w_open: float
    w_empty: float
    min_loaded_steps: int = 3
    min_open_steps: int = 2


class GripperFSM:
    """Hysteretic state machine: a state change needs several consecutive steps in the new band."""

    def __init__(self, th):
        self.th = th
        self.reset()

    def reset(self):
        self.state = OPEN
        self.run_band, self.run_len = None, 0

    def _band(self, w):
        if w > self.th.w_open:
            return OPEN
        return EMPTY if w < self.th.w_empty else LOADED

    def step(self, t, width, x_ee):
        band = self._band(width)
        self.run_len = self.run_len + 1 if band == self.run_band else 1
        self.run_band = band
        needed = {OPEN: self.th.min_open_steps, LOADED: self.th.min_loaded_steps, EMPTY: 1}[band]
        if band == self.state or self.run_len != needed:
            return None
        prev, self.state = self.state, band
        kind = {(OPEN, LOADED): "G+", (EMPTY, LOADED): "G+", (LOADED, OPEN): "R+", (EMPTY, OPEN): "R0"}.get((prev, band))
        return Event(t, kind, np.asarray(x_ee, float).copy()) if kind else None


def detect_events(widths, x_ee, th):
    fsm = GripperFSM(th)
    events = (fsm.step(t, float(w), x_ee[t]) for t, w in enumerate(widths))
    return [e for e in events if e is not None]


def event_windows(t, T, pre=8, post=16):
    return slice(max(0, t - pre), t), slice(t, min(T, t + post))


def reencounter_index(events, j, rho=0.06):
    for k in range(j + 1, len(events)):
        if np.linalg.norm(events[k].x_ee - events[j].x_ee) <= rho:
            return k
    return None


def calibrate_thresholds(widths, iters=50):
    """1D 3-means over all widths (empty, loaded, open); thresholds at the midpoints."""
    w = np.concatenate([np.asarray(x, float).ravel() for x in widths])
    centers = np.quantile(w, [0.05, 0.5, 0.95])
    for _ in range(iters):
        assign = np.argmin(np.abs(w[:, None] - centers[None]), axis=1)
        centers = np.array([w[assign == k].mean() if (assign == k).any() else centers[k] for k in range(3)])
    c = np.sort(centers)
    return FsmThresholds(w_open=float((c[1] + c[2]) / 2), w_empty=float((c[0] + c[1]) / 2))
