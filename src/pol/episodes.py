"""Load collected episodes and their gripper events (shared by the revealed test, probes and Stage A data)."""

import json
from pathlib import Path

import numpy as np

from .events import calibrate_thresholds, detect_events, reencounter_index
from .labels import event_outcomes

FAMILY = {"BatteriesSort": "batteries_sort", "SealedBin": "sealed_bin", "GatherInject": "gather_inject"}


def family_of(task):
    return FAMILY[task.split("-")[0]]


def load_episodes(task_dir, with_rgb=True):
    """[{seed, task, family, rgb?, proprio, action, logs, split}] for every <seed>.npz with a log."""
    out = []
    for npz in sorted(Path(task_dir).glob("*.npz")):
        log = json.loads(npz.with_suffix(".log.json").read_text())
        z = np.load(npz)
        ep = {"seed": log["seed"], "task": log["task"], "family": family_of(log["task"]), "split": log.get("split"),
              "proprio": z["proprio"], "action": z["action"], "logs": log["logs"], "language": str(z["language"])}
        if with_rgb:
            ep["rgb"] = z["rgb"]
        out.append(ep)
    return out


def fsm_thresholds(episodes):
    return calibrate_thresholds([ep["proprio"][:, 6] for ep in episodes])


def episode_events(ep, th, rho=0.06):
    """Events, outcome index (or None) and re-encounter index (or None) per event."""
    events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], th)
    outcomes = event_outcomes(ep["family"], events, ep["logs"])
    reenc = [reencounter_index(events, j, rho) for j in range(len(events))]
    return events, outcomes, reenc
