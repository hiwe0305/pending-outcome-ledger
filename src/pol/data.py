"""Datasets for Stage A (event tuples on frozen features) and Stage B (pi0 policy calls), and the
causal replay that turns an episode into the memory tokens every policy call sees.

Causality: at step t a memory observes the frame and proprio of step t and the action executed at
t-1 (never action[t], which is what the policy is about to predict). Memory tokens for the call at t
are read after observing step t.
"""

import json

import numpy as np

from .events import KINDS, GripperFSM, detect_events, event_windows, reencounter_index
from .memories.base import StepObs
from .memories.pol import N_TIME, POST, PRE, _blocks

CALL = 8
STATE_DIM, ACTION_DIM, PAD_DIM = 7, 7, 32


def split_seeds(seeds, val_frac=0.1):
    s = sorted(seeds)
    n_val = max(1, int(round(len(s) * val_frac))) if len(s) > 1 else 0
    return s[: len(s) - n_val], s[len(s) - n_val:]


def token_frames(T, events, pre=PRE, post=POST, call=CALL):
    frames = set(range(0, T, call))
    for e in events:
        a, b = event_windows(e.t, T, pre, post)
        frames.update(range(a.start, b.stop))
    return sorted(frames)


def prev_actions(actions):
    out = np.zeros_like(actions)
    out[1:] = actions[:-1]
    return out


# ---------------------------------------------------------------- Stage A
def _window_tokens(tokens, start, stop):
    return _blocks([tokens[t] for t in range(start, stop) if t in tokens], N_TIME, (32, 256))


def event_tuples(ep, pooled, tokens, th, rho):
    """One dict per gripper event, with view 2 at its re-encounter when there is one."""
    T = len(ep["proprio"])
    events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], th)
    act_prev = prev_actions(ep["action"])
    out = []
    for j, e in enumerate(events):
        if e.t + POST > T:
            continue
        pre, post = event_windows(e.t, T)
        p_pre = pooled[pre].mean(0) if pre.stop > pre.start else pooled[e.t]
        act = np.zeros((16, 7), np.float32)
        lo = max(0, e.t - 8)
        act[lo - (e.t - 8):] = act_prev[lo:e.t + 8]
        # act1/act2, pos1/pos2: actions and end-effector positions observed at the frames of each view (the
        # scene residual's motion and place inputs)
        item = {"t": e.t, "kind": e.kind, "pooled": p_pre.astype(np.float32), "act": act,
                "v1": _window_tokens(tokens, e.t, e.t + POST), "act1": act_prev[e.t:e.t + POST].astype(np.float32),
                "pos1": ep["proprio"][e.t:e.t + POST, :3].astype(np.float32),
                "has_v2": False, "v2": np.zeros((N_TIME, 32, 256), np.float32), "w2": np.zeros(16, np.float32),
                "act2": np.zeros((POST, 7), np.float32), "pos2": np.zeros((POST, 3), np.float32),
                "pooled2": p_pre.astype(np.float32)}
        r = reencounter_index(events, j, rho)
        if r is not None and events[r].t + POST <= T:
            tr = events[r].t
            pre_r, _ = event_windows(tr, T)
            ta = min(T - 1, tr + 8)  # just after the re-encounter interaction (gripper closed / released)
            item.update(markov_pooled=np.asarray(pooled[ta], np.float32), markov_proprio=ep["proprio"][ta].astype(np.float32))
            item.update(has_v2=True, v2=_window_tokens(tokens, tr, tr + POST), act2=act_prev[tr:tr + POST].astype(np.float32),
                        pos2=ep["proprio"][tr:tr + POST, :3].astype(np.float32),
                        w2=ep["proprio"][tr:tr + 16, 6].astype(np.float32),
                        pooled2=(pooled[pre_r].mean(0) if pre_r.stop > pre_r.start else pooled[tr]).astype(np.float32))
        out.append(item)
    return out


def hard_negative_batches(ctx, kinds, batch_size, group=32, seed=0):
    """Batches made of groups: an anchor and its group-1 nearest same-kind neighbours in context.
    Every event is in exactly one batch per epoch; neighbours never include the anchor itself."""
    rng = np.random.default_rng(seed)
    ctx = np.asarray(ctx, np.float32)
    ctx = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8)
    kinds = np.asarray(kinds)
    unused = set(range(len(ctx)))
    groups = []
    for a in rng.permutation(len(ctx)):
        if a not in unused:
            continue
        unused.discard(a)
        cand = np.array([i for i in unused if kinds[i] == kinds[a]], int)
        if len(cand):
            near = cand[np.argsort(-(ctx[cand] @ ctx[a]))[: group - 1]]
            unused.difference_update(near.tolist())
        else:
            near = np.array([], int)
        groups.append([int(a)] + near.tolist())
    order = [i for g in groups for i in g]
    return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]


def collate_events(items):
    import torch

    keys = ["pooled", "act", "v1", "act1", "pos1", "v2", "act2", "pos2", "w2", "pooled2"]
    batch = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in keys}
    batch["has_v2"] = torch.as_tensor([it["has_v2"] for it in items])
    batch["kind"] = torch.as_tensor([KINDS.index(it["kind"]) for it in items])
    batch["group"] = torch.as_tensor([event_group(it.get("task", ""), it["kind"]) for it in items])
    return batch


def event_group(task, kind):
    """Stable id of a task:kind group, for the outcome model's clustering within a group. The task is known to
    the policy from its instruction, so this is not a label."""
    import zlib

    return zlib.crc32(f"{task}:{kind}".encode()) & 0x7FFFFFFF


# ------------------------------------------------------------ memory replay
def replay_memory(producer, ep, pooled, tokens, th, call=CALL, with_log=False):
    """Run a MemoryProducer causally over an episode. Returns call steps [n], tokens [n,K,50], mask [n,K]."""
    producer.reset()
    fsm = GripperFSM(th)
    act_prev = prev_actions(ep["action"])
    outcome_at = {}
    if with_log:  # evaluator-only: the Oracle sees the true outcome of each event (index 7 = no outcome)
        from .labels import event_outcomes
        from .ledger import M_MAX

        events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], th)
        outcome_at = {e.t: (M_MAX - 1 if o is None else o)
                      for e, o in zip(events, event_outcomes(ep["family"], events, ep["logs"]))}
    steps, feats, masks = [], [], []
    for t in range(len(ep["proprio"])):
        prop = ep["proprio"][t]
        event = fsm.step(t, float(prop[6]), prop[:3])
        log = {**ep["logs"][t], "event_outcome": outcome_at.get(t, M_MAX - 1)} if with_log else None
        producer.observe(StepObs(t=t, pooled=np.asarray(pooled[t], np.float32), tokens=tokens.get(t),
                                 proprio=prop, action=act_prev[t], event=event, log=log))
        if t % call == 0:
            f, m = producer.tokens()
            steps.append(t)
            feats.append(f)
            masks.append(m)
    return np.array(steps), np.stack(feats), np.stack(masks)


# ---------------------------------------------------------------- Stage B
class Normalizer:
    def __init__(self, state_mean, state_std, action_mean, action_std):
        self.sm, self.ss = np.asarray(state_mean, np.float32), np.asarray(state_std, np.float32)
        self.am, self.as_ = np.asarray(action_mean, np.float32), np.asarray(action_std, np.float32)

    @classmethod
    def fit(cls, episodes):
        s = np.concatenate([e["proprio"] for e in episodes])
        a = np.concatenate([e["action"] for e in episodes])
        return cls(s.mean(0), s.std(0) + 1e-6, a.mean(0), a.std(0) + 1e-6)

    def state(self, x):
        return (np.asarray(x, np.float32) - self.sm) / self.ss

    def action(self, x):
        return (np.asarray(x, np.float32) - self.am) / self.as_

    def unaction(self, x):
        return np.asarray(x, np.float32)[..., :ACTION_DIM] * self.as_ + self.am

    def save(self, path):
        json.dump({k: v.tolist() for k, v in zip(["state_mean", "state_std", "action_mean", "action_std"],
                                                 [self.sm, self.ss, self.am, self.as_])}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["state_mean"], d["state_std"], d["action_mean"], d["action_std"])


def pad(x, dim=PAD_DIM):
    x = np.asarray(x, np.float32)
    out = np.zeros(x.shape[:-1] + (dim,), np.float32)
    out[..., : x.shape[-1]] = x
    return out


def call_sample(ep, t, memory, memory_mask, norm, horizon=16):
    """One Stage B sample in openpi's observation layout (images uint8 HWC, state/actions padded to 32)."""
    T = len(ep["action"])
    idx = np.minimum(np.arange(t, t + horizon), T - 1)
    rgb = ep["rgb"][t]
    return {"image": {"base_0_rgb": rgb[..., :3], "left_wrist_0_rgb": rgb[..., 3:],
                      "right_wrist_0_rgb": np.zeros_like(rgb[..., :3])},
            "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False},
            "state": pad(norm.state(ep["proprio"][t])), "actions": pad(norm.action(ep["action"][idx])),
            "action_is_pad": np.arange(t, t + horizon) >= T, "memory": memory.astype(np.float32),
            "memory_mask": memory_mask.astype(bool), "prompt": ep["language"]}


def stage_a_items(data_root, store, tasks, split, task_cfgs):
    """All Stage A event tuples of the given tasks and split ('train' / 'val' by seed, see split_seeds)."""
    from pathlib import Path

    from .episodes import load_episodes
    from .events import FsmThresholds

    items = []
    for task in tasks:
        eps = [e for e in load_episodes(Path(data_root) / task, with_rgb=False) if e["split"] in (None, "train")]
        train, val = split_seeds([e["seed"] for e in eps])
        keep = set(train if split == "train" else val)
        th = FsmThresholds(**json.loads((store.root / task / "fsm.json").read_text()))
        for ep in eps:
            if ep["seed"] not in keep:
                continue
            from .labels import FAILED, event_observable, event_outcomes, gate_target

            events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], th)
            outcome = dict(zip([e.t for e in events], event_outcomes(ep["family"], events, ep["logs"])))
            observable = dict(zip([e.t for e in events], event_observable(ep["family"], events, ep["logs"])))
            for it in event_tuples(ep, store.pooled(task, ep["seed"]), store.tokens(task, ep["seed"]), th,
                                   float(task_cfgs[task]["rho"])):
                o = outcome.get(it["t"])
                # evaluator-only labels: used for metrics, gates and the label-trained baselines, never by POL
                it.update(task=task, seed=ep["seed"], family=ep["family"], revealed=bool(task_cfgs[task]["revealed"]),
                          outcome=o, failed=None if o is None else o in FAILED[ep["family"]],
                          target=gate_target(ep["family"], o),
                          observable=observable.get(it["t"]))
                items.append(it)
    return items
