"""Gate P0: can the decision target be read from v1 only, v2 only, or needs both (v1 + v2)?

    python scripts/pol/probe_views.py --data data/pol --features data/features --tasks SealedBin-5 ... \
        --labels-per-task 32 --out runs/<round>/probe_views.json

Per task, on events with a re-encounter and a decision target: cross-validated AUROC of a logistic probe on
mean-pooled v1, mean-pooled v2 and their concatenation, plus the 3-class (outcome) macro one-vs-rest AUROC.
Gate: v12 - max(v1, v2) >= 0.05 (late evidence combination is needed), else the task needs no late update.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GATE_GAIN = 0.05


def _cv_proba(X, y, seed, folds=4):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    k = max(2, min(folds, int(np.bincount(np.searchsorted(classes, y)).min())))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(X[tr], y[tr])
        P[np.ix_(te, np.searchsorted(classes, m.classes_))] = m.predict_proba(X[te])
    return classes, P


def _auroc(y, X, seed, multiclass=False):
    from sklearn.metrics import roc_auc_score

    classes, P = _cv_proba(X, y, seed)
    if len(classes) < 2:
        return None
    if not multiclass:
        return float(roc_auc_score(y == classes[-1], P[:, -1]))
    a = [roc_auc_score(y == c, P[:, i]) for i, c in enumerate(classes)]
    return float(np.mean(a))


def probe_aurocs(X1, X2, y, n_labels, seed=0):
    """CV AUROC of logistic probes on v1, v2, [v1, v2] for labels y (binary decision target, or integer outcome
    classes -> macro one-vs-rest). At most n_labels examples are used (random subset)."""
    X1, X2, y = np.asarray(X1, np.float64), np.asarray(X2, np.float64), np.asarray(y)
    sel = np.random.default_rng(seed).permutation(len(y))[:n_labels]
    X1, X2, y = X1[sel], X2[sel], y[sel]
    multi = len(np.unique(y)) > 2
    return {k: _auroc(y, X, seed, multi) for k, X in (("v1", X1), ("v2", X2), ("v12", np.hstack([X1, X2])))}


def task_probe(items, n_labels, seed=0):
    sel = [it for it in items if it["has_v2"] and it["target"] is not None]
    X1 = np.stack([it["v1"].mean((0, 1)) for it in sel]) if sel else np.zeros((0, 1))
    X2 = np.stack([it["v2"].mean((0, 1)) for it in sel]) if sel else np.zeros((0, 1))
    out = {"n": len(sel), "positives": int(sum(it["target"] for it in sel))}
    if len(sel) < 8 or out["positives"] in (0, len(sel)):
        return out
    out["target"] = probe_aurocs(X1, X2, np.array([it["target"] for it in sel], int), n_labels, seed)
    o = np.array([it["outcome"] for it in sel], int)
    if len(np.unique(o)) > 1:
        out["outcome_macro"] = probe_aurocs(X1, X2, o, n_labels, seed)
    t = out["target"]
    if None not in t.values():
        out["gain"] = t["v12"] - max(t["v1"], t["v2"])
        out["gate_pass"] = bool(out["gain"] >= GATE_GAIN)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", default=str(ROOT / "data/features"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--labels-per-task", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from pol.config import load_config
    from pol.data import stage_a_items
    from pol.features import FeatureStore

    store = FeatureStore(a.features)
    cfgs = {t: load_config(ROOT / f"configs/pol/tasks/{t}.yaml") for t in a.tasks}
    res = {}
    for t in a.tasks:  # train + val events together: the probe is cross-validated
        items = stage_a_items(a.data, store, [t], "train", cfgs) + stage_a_items(a.data, store, [t], "val", cfgs)
        res[t] = task_probe(items, a.labels_per_task, a.seed)
        print(t, json.dumps(res[t]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
