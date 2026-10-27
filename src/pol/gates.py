"""Go / no-go gates after Stage A (spec section 10.1), on validation events, before any pi0 training.

    python -m pol.gates --pol runs/stage_a/POL --markov runs/stage_a/Markov --data data/pol \
        --features data/features --tasks ... --out runs/gates.json

Unit: a gripper event with a labelled outcome and a re-encounter. Scores of "this event failed":
  POL     posterior after the event's own view and its re-encounter view, mapped to P(failed) with
          P(failed | z) estimated on train events (evaluator-only calibration, not used by the policy)
  Markov  the memoryless label-trained classifier on the frame + proprio just after the re-encounter
  POL with fake actions (taken from another event of the same kind)
  -omega  for occluded vs visible outcome events (tasks with a static occluder)
Exit code 0 = all gates pass, 2 = N1 or G1 fails (stop: Plan B), 3 = only G2 fails (drop omega from the claim).
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

THRESHOLDS = {"n1_gain": 0.10, "n1_markov_max": 0.75, "n1_revealed_tol": 0.03, "g1_min": 0.80,
              "g1_fake_max": 0.60, "g2_min": 0.80}
ROOT = Path(__file__).resolve().parents[2]


def decide(s, th=THRESHOLDS):
    """s = {"unrevealed": {"pol", "markov", "pol_fake"}, "revealed": {"pol", "markov"}, "omega_auroc"}; values may be None."""
    u, r = s.get("unrevealed", {}), s.get("revealed", {})
    reasons = []

    def have(*v):
        return all(x is not None for x in v)

    n1 = have(u.get("pol"), u.get("markov")) and u["pol"] - u["markov"] >= th["n1_gain"] and u["markov"] <= th["n1_markov_max"]
    if have(r.get("pol"), r.get("markov")):
        n1 = n1 and abs(r["pol"] - r["markov"]) <= th["n1_revealed_tol"]
    else:
        reasons.append("N1: no revealed-set scores (control not checked)")
    g1 = have(u.get("pol"), u.get("pol_fake")) and u["pol"] >= th["g1_min"] and u["pol_fake"] <= th["g1_fake_max"]
    g2 = have(s.get("omega_auroc")) and s["omega_auroc"] >= th["g2_min"]
    for name, ok, keys in [("N1", n1, [u.get("pol"), u.get("markov")]), ("G1", g1, [u.get("pol"), u.get("pol_fake")]),
                           ("G2", g2, [s.get("omega_auroc")])]:
        if not ok and not have(*keys):
            reasons.append(f"{name}: insufficient data")
    code = 0 if (n1 and g1 and g2) else (2 if not (n1 and g1) else 3)
    return {"N1": bool(n1), "G1": bool(g1), "G2": bool(g2), "exit_code": code, "notes": reasons, "thresholds": th}


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, bool)
    if len(labels) < 4 or labels.all() or (~labels).all():
        return None
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def posterior(model, items, fake=False, seed=0):
    from .data import collate_events

    b = collate_events(items)
    act = b["act"]
    if fake:  # actions of another event of the same kind
        rng = np.random.default_rng(seed)
        kinds = np.array([it["kind"] for it in items])
        perm = np.arange(len(items))
        for k in set(kinds):
            idx = np.where(kinds == k)[0]
            perm[idx] = rng.permutation(idx)
        act = act[torch.as_tensor(perm)]
    lp = model.log_posterior_v1(b["pooled"], act, b["v1"])
    ll2 = model.loglik_v2(b["pooled"], act, b["v2"], b["w2"])
    has2 = b["has_v2"].float()[:, None]
    return torch.softmax(lp + has2 * ll2, -1).numpy()


def compute_scores(pol_dir, markov_dir, data, features, tasks):
    from .assets import load_assets
    from .config import load_config
    from .data import stage_a_items
    from .features import FeatureStore
    from .train_stage_a import markov_xy

    store = FeatureStore(features)
    cfgs = {t: load_config(ROOT / f"configs/pol/tasks/{t}.yaml") for t in tasks}
    tr = stage_a_items(data, store, tasks, "train", cfgs)
    va = stage_a_items(data, store, tasks, "val", cfgs)
    pol = load_assets(pol_dir, features)
    model = pol["outcome_model"]
    lab_tr = [it for it in tr if it["failed"] is not None]
    q_tr = posterior(model, lab_tr) if lab_tr else np.zeros((0, model.M))
    y_tr = np.array([it["failed"] for it in lab_tr], float)
    p_fail_z = (q_tr * y_tr[:, None]).sum(0) / (q_tr.sum(0) + 1e-6) if len(lab_tr) else np.full(model.M, 0.5)
    markov = load_assets(markov_dir, features).get("markov_head") if markov_dir else None
    proj = __import__("pol.features", fromlist=["PCA"]).PCA.load(Path(features) / "proj50.npz")
    out = {"n_val_events": len(va), "p_failed_given_z": p_fail_z.round(3).tolist()}
    for group, want in [("unrevealed", False), ("revealed", True)]:
        sel = [it for it in va if it["revealed"] == want and it["failed"] is not None and it["has_v2"]]
        y = [it["failed"] for it in sel]
        g = {"n": len(sel), "failures": int(sum(y))}
        if sel:
            g["pol"] = _auroc(posterior(model, sel) @ p_fail_z, y)
            g["pol_fake"] = _auroc(posterior(model, sel, fake=True) @ p_fail_z, y)
            if markov is not None:
                X, ym = markov_xy(sel, proj)
                with torch.no_grad():
                    g["markov"] = _auroc(torch.sigmoid(markov(torch.as_tensor(X))).numpy(), ym)
        out[group] = g
    occl = [it for it in va if it["observable"] is not None and it["family"] in ("batteries_sort", "sealed_bin")]
    if occl:
        from .data import collate_events

        b = collate_events(occl)
        with torch.no_grad():
            om = model.omega(model.context(b["pooled"], b["act"]), pol["sigma2"]).numpy()
        out["omega_auroc"] = _auroc(-om, [not it["observable"] for it in occl])
        out["n_occlusion_events"] = len(occl)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pol", required=True)
    p.add_argument("--markov", default=None)
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", default=str(ROOT / "data/features"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "runs/gates.json"))
    a = p.parse_args(argv)
    scores = compute_scores(a.pol, a.markov, a.data, a.features, a.tasks)
    verdict = decide(scores)
    report = {"scores": scores, "verdict": verdict}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return verdict["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
