"""Gate P0: can the decision target be read from v1 only, v2 only, or needs both (v1 + v2)?

    python scripts/pol/probe_views.py --data data/pol --features data/features --tasks SealedBin-5 ... \
        --labels-per-task 32 --out runs/<round>/probe_views.json

Per task, events with a re-encounter and a decision target: logistic probes on mean-pooled v1, v2 and their
concatenation, trained on n labelled TRAIN events (stratified; `--draws` random draws), AUROC on all VAL events
(mean, sd over draws), plus the macro one-vs-rest AUROC of the outcome classes. Degenerate draws are skipped
and counted. Provenance (seed, labels, draws, sampled event ids) is in the JSON.
Gate: v12 - max(v1, v2) >= 0.05 (late evidence combination is needed), else the task needs no late update.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

GATE_GAIN = 0.05


def _draw(y, n, rng):
    """n training indices, stratified: at least one per class present, the rest uniform."""
    first = [rng.choice(np.flatnonzero(y == c)) for c in np.unique(y)]
    rest = np.setdiff1d(np.arange(len(y)), first)
    extra = rng.permutation(rest)[:max(0, n - len(first))]
    return np.array(first + list(extra), int)


def _score(Xtr, ytr, Xva, yva):
    """AUROC on val of a logistic probe fitted on train: binary -> positive class, else macro one-vs-rest over
    the classes with both sides in val. None if the train set or val set lacks a class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(ytr)
    if len(classes) < 2 or len(classes) < len(np.unique(yva)):
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(Xtr, ytr)
    P = m.predict_proba(Xva)
    ok = [i for i, c in enumerate(m.classes_) if 0 < (yva == c).sum() < len(yva)]
    if not ok or (len(classes) == 2 and len(ok) < 2):
        return None
    if len(classes) == 2:
        return float(roc_auc_score(yva == m.classes_[1], P[:, 1]))
    return float(np.mean([roc_auc_score(yva == m.classes_[i], P[:, i]) for i in ok]))


def probe_aurocs(tr, va, y_tr, y_va, n_labels, draws=20, seed=0):
    """tr / va = (X1, X2) of the train / val events. Per draw: n_labels train events (stratified), probes on v1,
    v2, [v1, v2], AUROC on all val events. Returns {view: {mean, sd, n}} plus skipped draws and the sampled
    train indices per draw. Degenerate draws (a class missing in train or val) are skipped, never raised."""
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
    feats = {"v1": (tr[0], va[0]), "v2": (tr[1], va[1]), "v12": (np.hstack(tr), np.hstack(va))}
    rng = np.random.default_rng(seed)
    got = {k: [] for k in feats}
    skipped, idx = 0, []
    for _ in range(draws):
        i = _draw(y_tr, n_labels, rng) if len(y_tr) else np.zeros(0, int)
        idx.append(i.tolist())
        r = {k: _score(np.asarray(a)[i], y_tr[i], np.asarray(b), y_va) for k, (a, b) in feats.items()} if len(i) else {}
        if not r or None in r.values():
            skipped += 1
            continue
        for k, v in r.items():
            got[k].append(v)
    out = {k: {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v)) if v else None, "n": len(v)}
           for k, v in got.items()}
    out.update(skipped=skipped, draws_idx=idx)
    return out


def _gain(r):
    m = [r[k]["mean"] for k in ("v1", "v2", "v12")]
    return None if None in m else m[2] - max(m[0], m[1])


def _xy(items):
    sel = [it for it in items if it["has_v2"] and it["target"] is not None]
    X = (np.stack([it["v1"].mean((0, 1)) for it in sel]), np.stack([it["v2"].mean((0, 1)) for it in sel])) if sel \
        else (np.zeros((0, 1)), np.zeros((0, 1)))
    return sel, X


def task_probe(train, val, n_labels, draws=20, seed=0):
    """Train / val items are split by episode seed (data.split_seeds), so no val episode trains a probe."""
    st, Xt = _xy(train)
    sv, Xv = _xy(val)
    ev = lambda sel: [[it["task"], it["seed"], it["t"]] for it in sel]
    out = {"n_train": len(st), "n_val": len(sv), "positives_val": int(sum(it["target"] for it in sv))}
    if not st or not sv:
        return out
    out["target"] = probe_aurocs(Xt, Xv, [int(it["target"]) for it in st], [int(it["target"]) for it in sv],
                                 n_labels, draws, seed)
    out["outcome_macro"] = probe_aurocs(Xt, Xv, [it["outcome"] for it in st], [it["outcome"] for it in sv],
                                        n_labels, draws, seed)
    for k in ("target", "outcome_macro"):
        out[k]["gain"] = _gain(out[k])
    g = out["target"]["gain"]
    out["gate_pass"] = None if g is None else bool(g >= GATE_GAIN)
    out["train_events"] = ev(st)  # draws_idx index into this list
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", default=str(ROOT / "data/features"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--labels-per-task", type=int, default=32)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from pol.config import load_config
    from pol.data import stage_a_items
    from pol.features import FeatureStore

    store = FeatureStore(a.features)
    cfgs = {t: load_config(ROOT / f"configs/pol/tasks/{t}.yaml") for t in a.tasks}
    res = {"seed": a.seed, "labels_per_task": a.labels_per_task, "draws": a.draws, "tasks": {}}
    for t in a.tasks:
        tr = stage_a_items(a.data, store, [t], "train", cfgs)
        va = stage_a_items(a.data, store, [t], "val", cfgs)
        res["tasks"][t] = task_probe(tr, va, a.labels_per_task, a.draws, a.seed)
        print(t, json.dumps({k: v for k, v in res["tasks"][t].items() if k != "train_events"}, default=str)[:600])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
