"""Gate P1 (roadmap 2026-11-03, Task 2): does the lowered shutter hide the lamp, and does BatteriesSort need v1 and v2?

    python scripts/pol/probe_p1.py --data data/pol --features data/features_round2 --out runs/round2/probe_p1.json

All probes: scaler + logistic regression (C=0.1) on frozen features, train split -> val split (by episode seed).
  leak     train on VISIBLE train events (working vs broken, v1), score the SHUTTER-DOWN events of both splits
           (none of them trained the probe) and, as the reference, the visible val events. Full labels: a leak test
           wants the strongest probe. Gate: AUROC on shutter-down <= 0.60.
  views    events with a re-encounter, n labelled train events (`--labels`, 20 draws) and all labels:
             v1 broken vs spill    gate <= 0.65 (the flash alone cannot tell an empty socket from a broken battery)
             v2 working vs broken  gate <= 0.65 (the re-encounter alone cannot read the lamp)
             decision target from v1, v2, v1+v2   gate v1+v2 >= 0.85 (all labels), and the gain over max(v1, v2)
           decision AUROC is also reported per stratum (visible / shutter-down val events).
Two feature sets: "mean" (v1/v2 tokens averaged over time blocks and tokens, as probe_views) and "blocks" (one mean
per time block; the flash sits in blocks 0-1). 95 % CIs: bootstrap over val episodes (1000 resamples).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/pol"))

from probe_views import _draw  # noqa: E402

TASK = "BatteriesSort-3"
GATES = {"leak_max": 0.60, "v1_broken_spill_max": 0.65, "v2_working_broken_max": 0.65, "decision_v12_min": 0.85}


def feats(it, view, kind):
    v = it[view]
    return v.mean((0, 1)) if kind == "mean" else v.mean(1).reshape(-1)


def fit_score(Xtr, ytr, Xva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def auroc_ci(y, s, groups, n_boot=1000, seed=0):
    """AUROC with a 95 % bootstrap CI over groups (episodes); None if a class is missing."""
    from sklearn.metrics import roc_auc_score

    y, s, groups = np.asarray(y, bool), np.asarray(s), np.asarray(groups)
    if y.all() or not y.any():
        return {"auroc": None, "n": int(len(y)), "pos": int(y.sum())}
    rng, ug, boots = np.random.default_rng(seed), np.unique(groups), []
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    for _ in range(n_boot):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": float(roc_auc_score(y, s)), "ci": [float(lo), float(hi)], "n": int(len(y)), "pos": int(y.sum())}


def probe(tr, va, label, views, kind, n_labels=None, draws=20, seed=0):
    """AUROC on val of a probe on the concatenated views; all labels (with CI) or n_labels (mean, sd over draws)."""
    Xtr = np.stack([np.concatenate([feats(it, v, kind) for v in views]) for it in tr])
    Xva = np.stack([np.concatenate([feats(it, v, kind) for v in views]) for it in va])
    ytr, yva = np.array([label(it) for it in tr]), np.array([label(it) for it in va])
    if n_labels is None:
        return auroc_ci(yva, fit_score(Xtr, ytr, Xva), [it["seed"] for it in va])
    from sklearn.metrics import roc_auc_score

    rng, got = np.random.default_rng(seed), []
    for _ in range(draws):
        i = _draw(ytr, n_labels, rng)
        if len(np.unique(ytr[i])) == 2 and 0 < yva.sum() < len(yva):
            got.append(roc_auc_score(yva, fit_score(Xtr[i], ytr[i], Xva)))
    return {"mean": float(np.mean(got)) if got else None, "sd": float(np.std(got)) if got else None, "n": len(got)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", required=True)
    p.add_argument("--labels", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from pol.config import load_config
    from pol.data import stage_a_items
    from pol.features import FeatureStore

    store, cfg = FeatureStore(a.features), {TASK: load_config(ROOT / f"configs/pol/tasks/{TASK}.yaml")}
    tr = [it for it in stage_a_items(a.data, store, [TASK], "train", cfg) if it["outcome"] is not None]
    va = [it for it in stage_a_items(a.data, store, [TASK], "val", cfg) if it["outcome"] is not None]
    count = lambda xs: {f"{o}/{'vis' if ob else 'occ'}": sum(1 for it in xs if it["outcome"] == o and it["observable"] == ob)
                        for o in (0, 1, 2) for ob in (True, False)}
    res = {"data": a.data, "features": a.features, "labels": a.labels, "gates": GATES,
           "events": {"train": count(tr), "val": count(va)}, "leak": {}, "views": {}}

    wb = lambda it: it["outcome"] in (0, 1)
    broken = lambda it: it["outcome"] == 1
    vis_tr = [it for it in tr if wb(it) and it["observable"]]
    occ_all = [it for it in tr + va if wb(it) and it["observable"] is False]
    vis_va = [it for it in va if wb(it) and it["observable"]]
    for kind in ("mean", "blocks"):
        res["leak"][kind] = {"shutter_down": probe(vis_tr, occ_all, broken, ["v1"], kind),
                             "visible_val": probe(vis_tr, vis_va, broken, ["v1"], kind)}

    r_tr, r_va = [it for it in tr if it["has_v2"]], [it for it in va if it["has_v2"]]
    sub = lambda xs, f: [it for it in xs if f(it)]
    bs = lambda it: it["outcome"] in (1, 2)
    target = lambda it: bool(it["target"])
    for kind in ("mean", "blocks"):
        out = res["views"][kind] = {}
        for n in (a.labels, None):
            key = "all" if n is None else f"n{n}"
            out[key] = {
                "v1_broken_vs_spill": probe(sub(r_tr, bs), sub(r_va, bs), broken, ["v1"], kind, n),
                "v2_working_vs_broken": probe(sub(r_tr, wb), sub(r_va, wb), broken, ["v2"], kind, n),
                **{f"decision_{name}": probe(r_tr, r_va, target, views, kind, n)
                   for name, views in (("v1", ["v1"]), ("v2", ["v2"]), ("v12", ["v1", "v2"]))},
            }
        for ob, name in ((True, "visible"), (False, "shutter_down")):
            va_s = sub(r_va, lambda it: it["observable"] is ob)
            out[f"all_{name}"] = {f"decision_{k}": probe(r_tr, va_s, target, views, kind)
                                  for k, views in (("v1", ["v1"]), ("v12", ["v1", "v2"]))}

    m = lambda r: r.get("auroc", r.get("mean"))
    best = lambda key: max((res["views"][k]["all"][key]["auroc"] or 0) for k in ("mean", "blocks"))
    res["verdict"] = {
        "leak": max(m(res["leak"][k]["shutter_down"]) for k in ("mean", "blocks")) <= GATES["leak_max"],
        "v1_broken_vs_spill": best("v1_broken_vs_spill") <= GATES["v1_broken_spill_max"],
        "v2_working_vs_broken": best("v2_working_vs_broken") <= GATES["v2_working_broken_max"],
        "decision_v12": best("decision_v12") >= GATES["decision_v12_min"],
    }
    res["verdict"]["pass"] = all(res["verdict"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
