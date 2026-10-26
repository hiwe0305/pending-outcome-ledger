"""Strict policy-only observations, episode splits and train-fitted scalers."""

from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np

# Released MIKASA tasks: used for the data audit and as the negative control.
TASKS = {
    f"BatteriesChecker{level}-{n}-VLA-v0": f"batteries_checker_{level.lower()}_{n}_vla_v0"
    for level in ("Easy", "Hard")
    for n in (3, 6)
}
# BatteriesSort extension (src/jem/envs/batteries_sort.py): primary EEM tasks.
SORT_TASKS = tuple(
    f"BatteriesSort{kind}-{n}-VLA-v0" for kind in ("Hard", "Flash", "Compound") for n in (3, 6)
)


@dataclass(frozen=True)
class PolicyObservation:
    rgb: np.ndarray  # [B,128,128,6], top then wrist
    proprio: np.ndarray  # [B,7], absolute xyz/rpy/metres
    language: tuple[str, ...]

    def __post_init__(self):
        if self.rgb.dtype != np.uint8 or self.rgb.shape[1:] != (128, 128, 6):
            raise ValueError("RGB must be [B,128,128,6] uint8")
        if (
            self.proprio.shape != (len(self.rgb), 7)
            or not np.isfinite(self.proprio).all()
        ):
            raise ValueError("Invalid proprio")
        if len(self.language) != len(self.rgb) or any(
            not isinstance(s, str) or not s.strip() for s in self.language
        ):
            raise ValueError("Missing language")


@dataclass(frozen=True)
class EvaluatorRecord:
    """Never passed to policy or backbone."""

    success_once: bool
    seed: int
    episode_length: int


def policy_observation(obs, language):
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # Explicit allowlist; unknown keys including oracle_info/task_cue are discarded.
    return PolicyObservation(
        array(obs["rgb"]), array(obs["proprio"]).astype(np.float32), tuple(language)
    )


def pack_cameras(top, wrist):
    if (
        top.shape != wrist.shape
        or top.shape[-3:] != (128, 128, 3)
        or top.dtype != np.uint8
        or wrist.dtype != np.uint8
    ):
        raise ValueError("Camera schema/order mismatch")
    return np.concatenate((top, wrist), axis=-1)


@dataclass
class Episode:
    episode_id: str
    rgb: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    language: str

    def __post_init__(self):
        n = len(self.actions)
        if n < 1 or self.rgb.shape != (n, 128, 128, 6) or self.rgb.dtype != np.uint8:
            raise ValueError("Invalid episode RGB")
        if self.proprio.shape != (n, 7) or self.actions.shape != (n, 7):
            raise ValueError("7D schema required")
        if not np.isfinite(self.actions).all() or not np.isfinite(self.proprio).all():
            raise ValueError("Nonfinite episode")
        if np.max(np.abs(self.actions)) > 1.00001:
            raise ValueError("Actions outside normalized controller units")
        if (
            self.timestamps.shape != (n,)
            or not np.isfinite(self.timestamps).all()
            or np.any(np.diff(self.timestamps) <= 0)
        ):
            raise ValueError("Noncausal timestamps")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(
                str(z["episode_id"]),
                z["rgb"],
                z["proprio"],
                z["action"],
                z["timestamp"],
                str(z["language"]),
            )

    def observation(self, t):
        if not 0 <= t < len(self.actions):
            raise IndexError(t)
        return PolicyObservation(
            self.rgb[t : t + 1], self.proprio[t : t + 1], (self.language,)
        )


def episode_split(ids, seed=17):
    """Independent deterministic rank split per task, no frame-level leakage."""
    groups = {}
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs")
    for eid in ids:
        groups.setdefault(eid.rsplit(":", 1)[0], []).append(eid)
    out = {"train": [], "val": [], "test": []}
    for group in sorted(groups):
        ordered = sorted(
            groups[group],
            key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest(),
        )
        a = int(0.8 * len(ordered))
        b = int(0.9 * len(ordered))
        for key, part in [
            ("train", ordered[:a]),
            ("val", ordered[a:b]),
            ("test", ordered[b:]),
        ]:
            out[key].extend(part)
    return out


@dataclass
class TrainNormalizer:
    low: np.ndarray
    high: np.ndarray
    train_ids: tuple[str, ...]

    def __post_init__(self):
        if self.low.shape != (7,) or self.high.shape != (7,):
            raise ValueError("Normalizer must have seven dimensions")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Invalid normalizer bounds")
        if not self.train_ids or len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("Invalid normalizer train IDs")

    @classmethod
    def fit(cls, episodes, train_ids, field):
        train_ids = tuple(sorted(train_ids))
        if not train_ids or not set(train_ids) <= set(episodes):
            raise ValueError("Missing train episodes")
        x = np.concatenate([getattr(episodes[eid], field) for eid in train_ids])
        low, high = np.quantile(x, [0.01, 0.99], axis=0).astype(np.float32)
        if field == "actions":
            low[6], high[6] = -1, 1  # continuous physical gripper convention
        return cls(low, high, train_ids)

    def transform(self, x):
        span = self.high - self.low
        safe = np.where(span > 1e-6, span, 1)
        return np.where(
            span > 1e-6, np.clip(2 * (x - self.low) / safe - 1, -1, 1), 0
        ).astype(np.float32)

    def inverse(self, x):
        return (
            (np.clip(x, -1, 1) + 1) * 0.5 * (self.high - self.low) + self.low
        ).astype(np.float32)

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "low": self.low.tolist(),
                    "high": self.high.tolist(),
                    "train_ids": self.train_ids,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(
            np.array(d["low"], np.float32),
            np.array(d["high"], np.float32),
            tuple(d["train_ids"]),
        )


@dataclass(frozen=True)
class Call:
    """One policy call: call-time observation, executed prefix, its microstep frames, targets."""

    t: int
    observation: PolicyObservation
    executed: np.ndarray  # [T,7] actions executed since the previous call
    dt: np.ndarray  # [T]
    frames: np.ndarray  # [T,128,128,6] frame observed after each executed action (last = call frame)
    target: np.ndarray  # [k,7] future action chunk
    mask: np.ndarray  # [k] valid targets


def calls(episode, k=8):
    """Yield calls with executed history and future targets kept strictly separate."""
    if k < 1:
        raise ValueError("positive chunk required")
    for t in range(0, len(episode.actions), k):
        start = max(0, t - k)
        previous = episode.actions[start:t].copy()
        dt = (
            np.diff(np.r_[episode.timestamps[start:t], episode.timestamps[t]]).astype(np.float32)
            if t
            else np.empty(0, np.float32)
        )
        frames = episode.rgb[start + 1 : t + 1] if t else episode.rgb[:0]
        target = np.zeros((k, 7), np.float32)
        mask = np.zeros(k, bool)
        n = min(k, len(episode.actions) - t)
        target[:n] = episode.actions[t : t + n]
        mask[:n] = True
        yield Call(t, episode.observation(t), previous, dt, frames, target, mask)


def pilot_normalizers(episodes, train_ids, config_dir):
    """A bounded RGB pilot may reuse the exact train-only numeric audit scalers."""
    if set(train_ids) <= set(episodes):
        return TrainNormalizer.fit(episodes, train_ids, "proprio"), TrainNormalizer.fit(episodes, train_ids, "actions")
    prop = TrainNormalizer.load(Path(config_dir) / "proprio-normalizer.json")
    action = TrainNormalizer.load(Path(config_dir) / "actions-normalizer.json")
    if set(prop.train_ids) != set(train_ids) or set(action.train_ids) != set(train_ids):
        raise ValueError("Prefitted normalizer train IDs differ from saved split")
    return prop, action
