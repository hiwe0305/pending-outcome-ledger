"""Precompute frozen SigLIP features for every collected episode (policy venv, GPU).

    python scripts/pol/precompute_features.py --data data/pol --tasks BatteriesSort-3 SealedBin-5 ... \
        --out data/features [--fit-pca] [--pca-frames 50000]

--fit-pca (run once, on train episodes of all given tasks) fits the token PCA (1152 -> 256) and the
pooled projection (2304 -> 50, used by the baseline memories), then every episode is encoded:
pooled features for all frames, PCA tokens at policy-call frames and event windows (pol.data.token_frames).
The gripper FSM thresholds of each task are calibrated on its episodes and saved next to the features.
Episodes are processed one at a time, so RAM stays at about one episode plus the model.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def episode_files(data, task, split):
    out = []
    for log in sorted((Path(data) / task).glob("*.log.json")):
        meta = json.loads(log.read_text())
        if split is None or meta.get("split") == split:
            out.append(log.with_name(log.name.replace(".log.json", ".npz")))
    return out


def load(npz):
    from pol.episodes import family_of

    log = json.loads(npz.with_name(npz.name.replace(".npz", ".log.json")).read_text())
    z = np.load(npz)
    return {"seed": log["seed"], "task": log["task"], "family": family_of(log["task"]), "split": log.get("split"),
            "rgb": z["rgb"], "proprio": z["proprio"], "action": z["action"], "logs": log["logs"],
            "language": str(z["language"])}


def fit_projections(extractor, files, n_frames, seed=0):
    from pol.features import PCA

    rng = np.random.default_rng(seed)
    per_ep = max(1, n_frames // max(1, len(files)))
    grids, pooled = [], []
    for f in files:
        rgb = np.load(f)["rgb"]
        idx = np.sort(rng.choice(len(rgb), min(per_ep, len(rgb)), replace=False))
        p, g = extractor(rgb[idx])
        grids.append(g.reshape(-1, g.shape[-1]))
        pooled.append(p.reshape(len(p), -1))
    return PCA.fit(np.concatenate(grids), 256, seed), PCA.fit(np.concatenate(pooled), 50, seed)


def main():
    from pol.config import load_config
    from pol.data import token_frames
    from pol.events import detect_events
    from pol.episodes import fsm_thresholds
    from pol.features import PCA, FeatureStore, SiglipExtractor

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "data/features"))
    p.add_argument("--weights", default=str(ROOT / "checkpoints/pi0_siglip.pt"))
    p.add_argument("--fit-pca", action="store_true")
    p.add_argument("--pca-frames", type=int, default=load_config(ROOT / "configs/pol/stage_a.yaml")["features"]["pca_fit_frames"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    store = FeatureStore(a.out)
    store.root.mkdir(parents=True, exist_ok=True)
    extractor = SiglipExtractor(a.weights, device=a.device, batch=a.batch)
    if a.fit_pca or not (store.root / "pca.npz").exists():
        train = [f for t in a.tasks for f in episode_files(a.data, t, "train")]
        pca, proj = fit_projections(extractor, train, a.pca_frames)
        pca.save(store.root / "pca.npz")
        proj.save(store.root / "proj50.npz")
        print(f"[features] PCA fitted on {len(train)} train episodes", flush=True)
    pca = PCA.load(store.root / "pca.npz")
    for task in a.tasks:
        files = episode_files(a.data, task, None)
        th = fsm_thresholds([{"proprio": np.load(f)["proprio"]} for f in files])
        (store.root / task).mkdir(parents=True, exist_ok=True)
        (store.root / task / "fsm.json").write_text(json.dumps(th.__dict__))
        for f in files:
            ep = load(f)
            if store.has(task, ep["seed"]):
                continue
            pooled, grid = extractor(ep["rgb"])
            events = detect_events(ep["proprio"][:, 6], ep["proprio"][:, :3], th)
            idx = token_frames(len(ep["rgb"]), events)
            tok = pca.transform(grid[idx].reshape(-1, grid.shape[-1])).reshape(len(idx), grid.shape[1], -1)
            store.write(task, ep["seed"], pooled, idx, tok)
        print(f"[features] {task}: {len(files)} episodes", flush=True)


if __name__ == "__main__":
    main()
