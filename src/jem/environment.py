"""Official runner integration with evaluator-only information and action ledger."""

from dataclasses import dataclass, field
import numpy as np
from .data import policy_observation


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def controller_action(action):
    x = to_numpy(action).astype(np.float32)
    if x.shape not in {(7,), (1, 7)} or not np.isfinite(x).all():
        raise ValueError("MIKASA action must be finite 7D")
    return np.clip(x.reshape(1, 7), -1, 1)


@dataclass
class ActionLedger:
    predicted: np.ndarray | None = None
    dispatched: list = field(default_factory=list)
    executed: list = field(default_factory=list)
    elapsed: list = field(default_factory=list)
    frames: list = field(default_factory=list)  # RGB observed after each executed action
    all_chunks: list = field(default_factory=list)

    def prediction(self, chunk):
        x = np.asarray(chunk, np.float32)
        if x.ndim != 2 or x.shape[1] != 7 or len(x) < 1 or not np.isfinite(x).all():
            raise ValueError("Invalid action chunk")
        if self.predicted is not None and len(self.executed) != len(self.predicted):
            raise RuntimeError("Refill before prior queue exhausted")
        self.predicted = x.copy()
        self.dispatched = []
        self.executed = []
        self.elapsed = []
        self.frames = []

    def dispatch(self, action):
        if self.predicted is None or len(self.dispatched) >= len(self.predicted):
            raise RuntimeError("No queued action")
        if len(self.dispatched) != len(self.executed):
            raise RuntimeError("Previous dispatch not completed")
        expected = controller_action(self.predicted[len(self.dispatched)])[0]
        actual = controller_action(action)[0]
        if not np.allclose(expected, actual, atol=1e-5):
            raise ValueError("Dispatch order/value differs from prediction")
        self.dispatched.append(actual.copy())
        return actual

    def acknowledge(self, dt, frame=None):
        if (
            len(self.dispatched) != len(self.executed) + 1
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise RuntimeError("Invalid execution acknowledgement")
        self.executed.append(self.dispatched[-1].copy())
        self.elapsed.append(float(dt))
        if frame is not None:
            self.frames.append(np.asarray(frame, np.uint8))

    def prefix(self):
        return np.array(self.executed, np.float32).reshape(-1, 7), np.array(
            self.elapsed, np.float32
        )

    def prefix_frames(self):
        """[T,128,128,6] frames aligned with the executed prefix, or None if not recorded."""
        if not self.frames or len(self.frames) != len(self.executed):
            return None
        return np.stack(self.frames)

    def finish(self):
        if self.predicted is not None:
            self.all_chunks.append(
                {
                    "predicted": self.predicted.copy(),
                    "dispatched": np.array(self.dispatched).reshape(-1, 7),
                    "executed": np.array(self.executed).reshape(-1, 7),
                }
            )
        # Prefix remains available for audit until reset; unexecuted suffix is never history.


class TrackingEnv:
    """Wrap an already canonical env, leaving official run_episode queue intact."""

    def __init__(self, env, policy, language, control_dt=None):
        self.env = env
        self.policy = policy
        self.language = language
        self.control_dt = control_dt
        self.ledger = ActionLedger()
        self.calls = 0

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.policy.reset()
        self.ledger = ActionLedger()
        self.calls = 0
        if "language_instruction" in info:
            self.language = str(info["language_instruction"])
        return obs, info

    def step(self, action):
        a = self.ledger.dispatch(action)
        original = action
        if hasattr(action, "new_tensor"):
            original = action.new_tensor(a[None])
        else:
            original = a[None]
        result = self.env.step(
            original
        )  # If this raises, it was dispatched but not acknowledged.
        dt = self.control_dt
        if dt is None:
            frequency = getattr(self.env.unwrapped, "control_freq", None)
            if frequency is None:
                raise RuntimeError(
                    "Control cadence unavailable; do not invent timestamps"
                )
            dt = 1 / float(frequency)
        frame = None
        obs = result[0]
        if isinstance(obs, dict) and "rgb" in obs:
            frame = to_numpy(obs["rgb"]).reshape(-1, 128, 128, 6)[0]
        self.ledger.acknowledge(dt, frame)
        if bool(to_numpy(result[2]).reshape(-1)[0]) or bool(
            to_numpy(result[3]).reshape(-1)[0]
        ):
            self.ledger.finish()
        return result

    def close(self):
        return self.env.close()


class OfficialPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.chunk_size = policy.chunk_size
        self.env = None

    def bind(self, tracking_env):
        self.env = tracking_env

    def forward(self, obs):
        if self.env is None:
            raise RuntimeError("Tracking env not bound")
        observation = policy_observation(obs, (self.env.language,))
        prefix, dt = self.env.ledger.prefix()
        frames = self.env.ledger.prefix_frames()
        self.env.calls += 1
        chunk = self.policy.inference(observation, prefix, dt, frames)[0]
        if self.env.ledger.predicted is not None:
            self.env.ledger.finish()
        self.env.ledger.prediction(chunk)
        return chunk


def official_episode(env, policy, language, seed, control_dt=None):
    from mikasa_robo_suite.vla.benchmarking import run_episode

    tracked = TrackingEnv(env, policy, language, control_dt)
    adapter = OfficialPolicyAdapter(policy)
    adapter.bind(tracked)
    result, _ = run_episode(tracked, adapter, seed)
    return result, tracked
