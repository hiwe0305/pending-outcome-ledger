"""End-to-end smoke test of the parts of the POL pipeline that exist so far (policy venv).

    python scripts/pol/smoke.py --data data/pol --features data/features --tasks SealedBin-3 GatherInject-3

Uses already collected episodes and precomputed features: builds the Stage A event set, runs a few
EM steps of the outcome model on the GPU, then replays POL and every baseline memory causally over one
episode and prints token statistics. It checks wiring, shapes and causality, not quality: with a
handful of episodes the numbers mean nothing.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch

    from pol import baseline_models as bm
    from pol.config import load_config
    from pol.data import collate_events, replay_memory, stage_a_items
    from pol.episodes import load_episodes
    from pol.events import FsmThresholds
    from pol.features import PCA, FeatureStore
    from pol.memories import REGISTRY, make_memory
    from pol.outcome_model import OutcomeModel, em_step

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", default=str(ROOT / "data/features"))
    p.add_argument("--tasks", nargs="+", default=["SealedBin-3", "GatherInject-3"])
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = FeatureStore(a.features)
    cfgs = {t: load_config(ROOT / f"configs/pol/tasks/{t}.yaml") for t in a.tasks}
    report = {"device": dev}

    items = stage_a_items(a.data, store, a.tasks, "train", cfgs)
    report["stage_a_events"] = len(items)
    report["with_reencounter"] = int(sum(i["has_v2"] for i in items))
    torch.manual_seed(0)
    model = OutcomeModel(M=6).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0, losses = time.time(), []
    for step in range(a.steps):
        idx = np.random.default_rng(step).choice(len(items), min(64, len(items)), replace=False)
        batch = {k: v.to(dev) for k, v in collate_events([items[i] for i in idx]).items()}
        out = em_step(model, batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"]))
    report["em_loss_first_last"] = [round(losses[0], 3), round(losses[-1], 3)]
    report["em_seconds"] = round(time.time() - t0, 1)
    report["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None

    model = model.cpu().eval()
    with torch.no_grad():
        model.set_banks(collate_events(items[:256]))
    v1 = np.stack([i["v1"] for i in items]).mean(1)
    proj = PCA.load(store.root / "proj50.npz")
    assets = {"outcome_model": model, "sigma2": torch.as_tensor(v1.var(0) + 1e-6, dtype=torch.float32), "omega0": 0.0,
              "proj50": proj.components.T.astype(np.float64), "proj_mean": proj.mean.astype(np.float64),
              "markov": lambda pooled, proprio: 0.5, "kem_scorer": bm.KeyframeScorer().eval(),
              "mu_net": bm.RecurrentMemory(K=8).eval(), "aura_net": bm.ActionPredictor().eval(),
              "statemem_net": bm.StateMemCell().eval(), "check_net": bm.FeaturePredictor().eval(),
              "check_mu": 0.0, "check_sd": 1.0, "conformal_q": 1.0, "still_threshold": 0.5,
              "verify_head": bm.VerifyHead().eval()}
    task = a.tasks[0]
    ep = load_episodes(Path(a.data) / task, with_rgb=False)[0]
    th = FsmThresholds(**json.loads((store.root / task / "fsm.json").read_text()))
    pooled, tokens = store.pooled(task, ep["seed"]), store.tokens(task, ep["seed"])
    report["replay_episode"] = f"{task}:{ep['seed']} ({len(ep['proprio'])} steps)"
    report["memories"] = {}
    for name in sorted(REGISTRY):
        mem = make_memory(name, {"name": name, "K": 8, "n_objects": cfgs[task]["n_objects"]}, assets)
        t0 = time.time()
        steps, feats, masks = replay_memory(mem, ep, pooled, tokens, th, with_log=(name == "Oracle"))
        report["memories"][name] = {"calls": int(len(steps)), "slots_used_last": int(masks[-1].sum()),
                                    "finite": bool(np.isfinite(feats).all()),
                                    "ms_per_step": round(1000 * (time.time() - t0) / len(ep["proprio"]), 2)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
