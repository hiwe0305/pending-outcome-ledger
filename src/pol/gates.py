"""Go / no-go gates after Stage A (spec section 10.1), on validation events, before any pi0 training.

    python -m pol.gates --pol runs/stage_a/POL --markov runs/stage_a/Markov --data data/pol \
        --features data/features --tasks ... --out runs/gates.json

Unit: a gripper event with a re-encounter whose outcome is a gate target (`labels.GATE_TARGET`: the outcome
hidden at decision time — broken battery, cube behind the bin; any failure on the revealed control).
Scores of "this event is positive":
  POL           posterior after the event's own view and its re-encounter view, mapped to P(target) with
                P(target | z) estimated per task on train events (evaluator-only calibration, not used by the policy)
  Markov        the memoryless label-trained classifier on the frame + proprio just after the re-encounter
  POL shuffled  the event's views (v1, v2) swapped with another event of the same task and kind, context and
                actions kept: what is left without the event's own evidence (G1 control). Fake actions are
                reported too but not gated: the injected outcomes do not depend on the action.
  -omega        for occluded vs visible outcome events (tasks with a static occluder)
Group scores (unrevealed, revealed) are the mean of the per-task AUROCs (`per_task`); pooling the events over
tasks also rewards the base rate of each task, so the pooled AUROCs are reported under "pooled" only.
Exit code 0 = all gates pass, 2 = N1 or G1 fails (stop: Plan B), 3 = only G2 fails (drop omega from the claim).
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

THRESHOLDS = {"n1_gain": 0.10, "n1_markov_max": 0.75, "n1_revealed_tol": 0.03, "g1_min": 0.80,
              "g1_shuffled_max": 0.60, "g2_min": 0.80}
VIEWS = ("v1", "act1", "pos1", "v2", "act2", "pos2", "w2", "pooled2", "has_v2")
ROOT = Path(__file__).resolve().parents[2]


def decide(s, th=THRESHOLDS):
    """s = {"unrevealed": {"pol", "markov", "pol_shuffled"}, "revealed": {"pol", "markov"}, "omega_auroc"}; values may be None."""
    u, r = s.get("unrevealed", {}), s.get("revealed", {})
    reasons = []

    def have(*v):
        return all(x is not None for x in v)

    n1 = have(u.get("pol"), u.get("markov")) and u["pol"] - u["markov"] >= th["n1_gain"] and u["markov"] <= th["n1_markov_max"]
    if have(r.get("pol"), r.get("markov")):
        n1 = n1 and abs(r["pol"] - r["markov"]) <= th["n1_revealed_tol"]
    else:
        reasons.append("N1: no revealed-set scores (control not checked)")
    g1 = have(u.get("pol"), u.get("pol_shuffled")) and u["pol"] >= th["g1_min"] and u["pol_shuffled"] <= th["g1_shuffled_max"]
    g2 = have(s.get("omega_auroc")) and s["omega_auroc"] >= th["g2_min"]
    for name, ok, keys in [("N1", n1, [u.get("pol"), u.get("markov")]), ("G1", g1, [u.get("pol"), u.get("pol_shuffled")]),
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


def shuffle_evidence(items):
    """Each event gets the views of the next event of the same task and kind (a rotation, so never its own
    unless it is alone in its group); context and actions stay."""
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault((it["task"], it["kind"]), []).append(i)
    src = list(range(len(items)))
    for idx in groups.values():
        for a, b in zip(idx, idx[1:] + idx[:1]):
            src[a] = b
    return [{**it, **{k: items[j][k] for k in VIEWS}} for it, j in zip(items, src)]


def calibrate(q, items, tasks):
    """P(target | z) per task from train posteriors q [N, M]: the target means a different thing in each task
    (broken battery, cube behind the bin, ...), so one mapping pooled over tasks would mix them."""
    M = q.shape[1]
    out = {}
    for t in tasks:
        m = np.array([it["task"] == t for it in items], bool)
        y = np.array([it["target"] for it in items], float)[m] if m.any() else np.zeros(0)
        out[t] = (q[m] * y[:, None]).sum(0) / (q[m].sum(0) + 1e-6) if m.any() else np.full(M, 0.5)
    return out


def macro(per_task):
    """Group score = mean of the per-task AUROCs (tasks without one are skipped); n and failures are summed.
    Pooling events over tasks instead rewards the task's base rate: with shuffled evidence it still
    reached 0.71 on round 1."""
    g = {"n": sum(s["n"] for s in per_task), "failures": sum(s["failures"] for s in per_task)}
    for k in ("pol", "pol_shuffled", "pol_fake_actions", "markov"):
        v = [s[k] for s in per_task if s.get(k) is not None]
        g[k] = float(np.mean(v)) if v else None
    return g


def readout(q, items, p_target_z):
    """Score of each event = its posterior against its own task's P(target | z)."""
    return np.array([qi @ p_target_z[it["task"]] for qi, it in zip(q, items)])


@torch.no_grad()
def posterior(model, items, fake=False, seed=0):
    from .data import collate_events

    b = collate_events(items)
    act, act1 = b["act"], b["act1"]
    if fake:  # actions of another event of the same kind
        rng = np.random.default_rng(seed)
        kinds = np.array([it["kind"] for it in items])
        perm = np.arange(len(items))
        for k in set(kinds):
            idx = np.where(kinds == k)[0]
            perm[idx] = rng.permutation(idx)
        act, act1 = act[torch.as_tensor(perm)], act1[torch.as_tensor(perm)]
    lp = model.log_posterior_v1(b["pooled"], act, b["v1"], b["kind"], b["group"], act1, b["pos1"])
    ll2 = model.loglik_v2(b["v2"], b["w2"], b["pooled2"], b["kind"], b["group"], b["act2"], b["pos2"])
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
    lab_tr = [it for it in tr if it["target"] is not None and it["has_v2"]]  # the population the gates score
    p_target_z = calibrate(posterior(model, lab_tr) if lab_tr else np.zeros((0, model.M)), lab_tr, tasks)
    markov = load_assets(markov_dir, features).get("markov_head") if markov_dir else None
    proj = __import__("pol.features", fromlist=["PCA"]).PCA.load(Path(features) / "proj50.npz")

    def scores(sel):
        y = [it["target"] for it in sel]
        g = {"n": len(sel), "failures": int(sum(y))}
        if sel:
            g["pol"] = _auroc(readout(posterior(model, sel), sel, p_target_z), y)
            g["pol_shuffled"] = _auroc(readout(posterior(model, shuffle_evidence(sel)), sel, p_target_z), y)
            g["pol_fake_actions"] = _auroc(readout(posterior(model, sel, fake=True), sel, p_target_z), y)
            if markov is not None:
                X, ym = markov_xy(sel, proj)
                with torch.no_grad():
                    g["markov"] = _auroc(torch.sigmoid(markov(torch.as_tensor(X))).numpy(), ym)
        return g

    gated = [it for it in va if it["target"] is not None and it["has_v2"]]
    out = {"n_val_events": len(va), "p_target_given_z": {t: p.round(3).tolist() for t, p in p_target_z.items()}}
    out["per_task"] = {t: scores([it for it in gated if it["task"] == t]) for t in tasks}
    for group, want in [("unrevealed", False), ("revealed", True)]:
        in_group = [t for t in tasks if bool(cfgs[t]["revealed"]) == want]
        out[group] = {**macro([out["per_task"][t] for t in in_group]),
                      "pooled": scores([it for it in gated if it["revealed"] == want])}
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
