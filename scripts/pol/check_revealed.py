"""Revealed / unrevealed test (paper, Definition 1) and single-frame shortcut probes.

    python scripts/pol/check_revealed.py data/pol/SealedBin-5 --features pixels --out runs/revealed/SealedBin-5.json

For every event with a labelled outcome and a re-encounter, a memoryless classifier sees only the
re-encounter (frames [t_r, t_r+8) and the gripper width over [t_r, t_r+16), plus the end-effector
position): logistic regression, 5-fold cross-validated AUROC, one outcome class against the rest.
A class is *revealed* at AUROC >= 0.9. The probe does the same with the single frame just before the
event, where the outcome must not be predictable (AUC <= 0.6).

--features pixels (16x16 downsampled RGB) is a smoke test only: a weak featurizer biases the
verdict towards "unrevealed". The paper's verdict uses --features siglip (frozen pi0 SigLIP,
pol.features.FeatureStore from scripts/pol/precompute_features.py).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

THRESHOLD_REVEALED, THRESHOLD_PROBE = 0.9, 0.6


def pixel_features(rgb):
    x = rgb.reshape(len(rgb), 16, 8, 16, 8, 6).mean((2, 4))
    return x.reshape(len(rgb), -1).astype(np.float32) / 255.0


def cv_auroc(X, y, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if min(y.sum(), (~y).sum()) < folds:
        return None
    scores = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, scores))


def build(episodes, featurize, which, rho):
    from pol.episodes import episode_events, fsm_thresholds

    th = fsm_thresholds(episodes)
    X, Y = [], []
    for ep in episodes:
        events, outcomes, reenc = episode_events(ep, th, rho)
        feats = None
        for j, (e, o) in enumerate(zip(events, outcomes)):
            if o is None:
                continue
            if which == "reencounter":
                if reenc[j] is None:
                    continue
                t = events[reenc[j]].t
                frames = slice(t, min(len(ep["rgb"]), t + 8))
            else:
                t = max(0, e.t - 1)
                frames = slice(t, t + 1)
            feats = featurize(ep, frames)
            w = np.zeros(16, np.float32)
            seg = ep["proprio"][t:t + 16, 6]
            w[: len(seg)] = seg
            X.append(np.concatenate([feats, w, ep["proprio"][t, :3]]))
            Y.append(o)
    return np.array(X), np.array(Y)


def main():
    from pol.episodes import load_episodes
    from pol.labels import OUTCOMES

    p = argparse.ArgumentParser()
    p.add_argument("task_dir")
    p.add_argument("--features", choices=["pixels", "siglip"], default="pixels")
    p.add_argument("--feature-root", default=str(ROOT / "data/features"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    episodes = load_episodes(a.task_dir)
    if not episodes:
        raise SystemExit(f"no episodes in {a.task_dir}")
    family = episodes[0]["family"]
    from pol.config import load_config

    rho = float(load_config(ROOT / f"configs/pol/tasks/{episodes[0]['task']}.yaml")["rho"])
    if a.features == "pixels":
        featurize = lambda ep, s: pixel_features(ep["rgb"][s]).mean(0)  # noqa: E731
    else:
        from pol.features import FeatureStore

        store = FeatureStore(a.feature_root)
        featurize = lambda ep, s: store.pooled(ep["task"], ep["seed"])[s].astype(np.float32).mean(0).ravel()  # noqa: E731
    report = {"task": episodes[0]["task"], "features": a.features, "episodes": len(episodes), "rho": rho, "classes": {}}
    for which, thr, key in [("reencounter", THRESHOLD_REVEALED, "revealed_test"), ("before", THRESHOLD_PROBE, "probe")]:
        X, Y = build(episodes, featurize, which, rho)
        report[f"n_{which}"] = int(len(Y))
        for c, name in enumerate(OUTCOMES[family]):
            auc = cv_auroc(X, Y == c) if len(Y) else None
            entry = report["classes"].setdefault(name, {})
            entry[f"count_{which}"] = int((Y == c).sum())
            entry[key] = auc
            if key == "revealed_test":
                entry["verdict"] = None if auc is None else ("revealed" if auc >= thr else "unrevealed")
            else:
                entry["probe_ok"] = None if auc is None else bool(auc <= thr)
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
