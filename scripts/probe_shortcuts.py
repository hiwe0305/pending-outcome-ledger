"""Shortcut probes on collected BatteriesSort episodes (proposal section 4).

A linear probe on downsampled call-time frames must NOT predict information that the
task design hides. Each probe reports held-out AUC; the gate fails above --max-auc.
  quality:  the battery in the gripper's hidden quality, at frames before its first reading
  seating:  seated vs spilled, at the first frame after release (before reaching in)
Usage: python scripts/probe_shortcuts.py data/batteriessort/BatteriesSortHard-3-VLA-v0
"""
import argparse, json
from pathlib import Path
import numpy as np


def features(rgb):
    """[N,128,128,6] uint8 -> [N,F] mean-pooled 8x8 colour features."""
    x = rgb.astype(np.float32).reshape(len(rgb), 8, 16, 8, 16, 6).mean((2, 4)) / 255
    return x.reshape(len(rgb), -1)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    if pos.all() or not pos.any():
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def logistic_probe(x, y, seed=0, steps=300, lr=0.5, l2=1e-3):
    """Fit on a random 70% split, return held-out AUC."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(0.7 * len(x))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    z = (x - mu) / sd
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z[tr] @ w + b)))
        g = p - y[tr]
        w -= lr * (z[tr].T @ g / len(tr) + l2 * w)
        b -= lr * g.mean()
    return auc(z[te] @ w + b, y[te])


def collect(folder):
    xs = {"quality": [], "seating": []}
    ys = {"quality": [], "seating": []}
    for npz in sorted(Path(folder).glob("*.npz")):
        logs = json.loads(npz.with_suffix("").with_suffix(".log.json").read_text())["logs"]
        rgb = np.load(npz)["rgb"]
        seen = set()
        for t, log in enumerate(logs):
            for i, loc in enumerate(log["location"]):
                if loc == 1 and i not in seen and t < len(rgb):  # held, not yet read
                    xs["quality"].append(rgb[t]); ys["quality"].append(int(log["working"][i])); seen.add(i)
            if (log["insert_event"] or log["spill_event"]) and t + 1 < len(rgb):
                xs["seating"].append(rgb[t + 1]); ys["seating"].append(int(bool(log["insert_event"])))
    return {k: (features(np.stack(v)), np.array(ys[k])) for k, v in xs.items() if len(v) >= 20}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--max-auc", type=float, default=0.6)
    args = p.parse_args()
    data = collect(args.folder)
    report = {k: logistic_probe(x, y) for k, (x, y) in data.items()}
    report["pass"] = all(v <= args.max_auc for v in report.values() if v == v)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
