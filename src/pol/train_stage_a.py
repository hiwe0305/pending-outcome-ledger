"""Stage A: train the memory-side modules of one variant on frozen features (policy venv, GPU).

    python -m pol.train_stage_a --variant POL --data data/pol --features data/features \
        --tasks BatteriesSort-3 SealedBin-5 ... --out runs/stage_a/POL [--steps N] [--seed 0]

POL: the latent outcome model by EM (pol.outcome_model.em_step) on gripper-event tuples, batches built
from nearest-context groups (hard negatives). Baselines: their small modules (pol.baseline_models) with
their own losses; KEM, MU, AURA, StateMem and CheckVLA are self-supervised, AGM-style and Markov are
label-trained as in the methods they stand for. Variants without a learned part (B0, Frames, Random,
Oracle) only get the shared pooled projection. Writes <out>/assets.pt (see pol.assets) and metrics.json.
Logs to W&B (project `pol`, group stage_a/<variant>) every `log_every` steps.
"""

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

from . import baseline_models as bm
from .assets import save_assets
from .config import load_config, provenance
from .data import collate_events, hard_negative_batches, split_seeds, stage_a_items
from .episodes import load_episodes
from .features import PCA, FeatureStore
from .outcome_model import OutcomeModel, em_step
from .wandb_utils import init_run

ROOT = Path(__file__).resolve().parents[2]
LEARNED = {"POL", "KEM", "MU", "AURA", "StateMem", "CheckVLA", "CheckFIFO", "AGM", "Markov"}
MODEL_KEYS = ("M", "tau", "d_feat", "d_tok", "n_time", "n_tok", "d_ctx", "d_act", "d_assoc")


def lr_at(step, total, warmup, base, final_frac=0.0):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, bool)
    return None if labels.all() or (~labels).all() else float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------- POL
def train_pol(a, cfg, items_tr, items_va, run, device):
    tc = cfg["train"]
    kwargs = {k: cfg["model"][k] for k in MODEL_KEYS}
    model = OutcomeModel(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ctx = np.stack([it["pooled"].reshape(-1) for it in items_tr])
    kinds = [it["kind"] for it in items_tr]
    bs = min(tc["batch_size"], len(items_tr))
    batches, epoch, step = [], 0, 0
    to = lambda b: {k: v.to(device) for k, v in b.items()}  # noqa: E731
    t0 = time.time()
    while step < a.steps:
        if not batches:
            batches = hard_negative_batches(ctx, kinds, bs, group=tc["hard_negatives"] + 1, seed=a.seed + epoch)
            epoch += 1
        idx = batches.pop(0)
        if len(idx) < 4:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, min(tc["warmup"], a.steps // 10 + 1), tc["lr"])
        out = em_step(model, to(collate_events([items_tr[i] for i in idx])), tc["lam_g"], tc["lam_u"], lam_i=tc["lam_i"], lam_i1=tc.get("lam_i1", 0.0))
        opt.zero_grad()
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        q = out["q"]
        marginal = q.mean(0)
        run.log(step, {"loss": out["loss"].item(), **{k: out[k].item() for k in ("prior_nll", "nce1", "nce2", "recon", "scene", "usage_kl", "view_mi", "v1_mi", "assoc")},
                       "q_entropy": float(-(q * q.clamp_min(1e-8).log()).sum(1).mean()),
                       "z_used": int((marginal > 0.02).sum()), "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
                       "steps_per_s": (step + 1) / (time.time() - t0)})
        if (step + 1) % tc["val_every"] == 0 or step + 1 == a.steps:
            run.log(step, {f"val/{k}": v for k, v in eval_pol(model, items_va, device).items() if v is not None}, force=True)
        step += 1
    model = model.cpu().eval()
    with torch.no_grad():
        train_batch = collate_events(items_tr[:4096])
        model.set_banks(train_batch)
        v1 = torch.as_tensor(np.stack([it["v1"].mean(0) for it in items_tr]))
        sigma2 = v1.var(0) + 1e-6
        ctx_t = model.context(train_batch["pooled"], train_batch["act"])
        omega = model.omega(ctx_t, sigma2).numpy()
    omega0 = float(np.quantile(omega, float(load_config(ROOT / "configs/pol/variants/POL.yaml")["memory"]["omega_quantile"])))
    metrics = eval_pol(model, items_va, "cpu", sigma2=sigma2)
    return {"outcome_model": model}, {"outcome_model": kwargs}, {"omega0": omega0}, {"sigma2": sigma2.numpy()}, metrics


@torch.no_grad()
def eval_pol(model, items, device, sigma2=None):
    """Validation EM loss plus evaluator-only diagnostics: NMI of argmax q with simulator outcomes and
    AUROC of -omega for occluded vs visible outcome events."""
    from sklearn.metrics import normalized_mutual_info_score

    if not items:
        return {}
    model.eval()
    b = {k: v.to(device) for k, v in collate_events(items).items()}
    out = em_step(model, b)
    q = out["q"].cpu().numpy()
    lab = [(i, it["outcome"]) for i, it in enumerate(items) if it["outcome"] is not None]
    nmi = float(normalized_mutual_info_score([o for _, o in lab], q[[i for i, _ in lab]].argmax(1))) if len(lab) > 1 else None
    s2 = sigma2.to(device) if sigma2 is not None else b["v1"].mean(1).var(0) + 1e-6
    omega = model.omega(model.context(b["pooled"], b["act"]), s2).cpu().numpy()
    obs = [(i, it["observable"]) for i, it in enumerate(items) if it["observable"] is not None]
    om = auroc(-omega[[i for i, _ in obs]], [not o for _, o in obs]) if obs else None
    model.train()
    return {"loss": out["loss"].item(), "nmi": nmi, "omega_auroc": om}


# ---------------------------------------------------------------- baselines
def frame_data(a, store, proj, split):
    eps = []
    for task in a.tasks:
        all_eps = [e for e in load_episodes(Path(a.data) / task, with_rgb=False) if e["split"] in (None, "train")]
        train, val = split_seeds([e["seed"] for e in all_eps])
        keep = set(train if split == "train" else val)
        for ep in all_eps:
            if ep["seed"] in keep:
                pooled = np.asarray(store.pooled(task, ep["seed"]), np.float32)
                ep["P"] = proj.transform(pooled.reshape(len(pooled), -1)).astype(np.float32)
                ep["dpool"] = np.r_[0.0, np.linalg.norm(np.diff(pooled.reshape(len(pooled), -1), axis=0), axis=1)]
                eps.append(ep)
    return eps


def fit(module, sampler, steps, lr, run, prefix, device):
    module = module.to(device).train()
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        batch = {k: torch.as_tensor(v).to(device) for k, v in sampler(step).items()}
        loss = module.stage_a_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        run.log(step, {f"{prefix}/loss": loss.item()})
    return module.cpu().eval()


def train_baseline(a, variant, cfg, store, proj, items_tr, run, device):
    eps = frame_data(a, store, proj, "train")
    eps_val = frame_data(a, store, proj, "val")
    rng = np.random.default_rng(a.seed)
    B, lr = 512, cfg["train"]["lr"]
    P = np.concatenate([e["P"] for e in eps])
    steps = a.baseline_steps
    mods, kw, sc = {}, {}, {}

    def frames_of(fn):
        return np.concatenate([fn(e) for e in eps])

    if variant == "KEM":
        d = np.concatenate([e["dpool"] for e in eps])
        lab = d > np.quantile(d, 0.9)  # self-supervised keyframe label: large feature change
        mods["kem_scorer"] = fit(bm.KeyframeScorer(), lambda s: (lambda i: {"x": P[i], "label": lab[i]})(rng.integers(len(P), size=B)),
                                 steps, lr, run, "kem", device)
        sc["kem_threshold"] = 0.5
    elif variant in ("MU", "StateMem"):
        seqs = [e["P"][::8] for e in eps if len(e["P"][::8]) >= 13]

        def seq_batch(_):
            out = []
            for _ in range(64):
                s = seqs[rng.integers(len(seqs))]
                o = rng.integers(len(s) - 12)
                out.append(s[o:o + 12])
            return {"seq": np.stack(out)}

        if variant == "MU":
            mods["mu_net"], kw["mu_net"] = fit(bm.RecurrentMemory(K=8), seq_batch, steps, lr, run, "mu", device), {"K": 8}
        else:
            mods["statemem_net"] = fit(bm.StateMemCell(), seq_batch, steps, lr, run, "statemem", device)
    elif variant == "AURA":
        A = frames_of(lambda e: e["action"])
        net = fit(bm.ActionPredictor(), lambda s: (lambda i: {"x": P[i], "action": A[i]})(rng.integers(len(P), size=B)),
                  steps, lr, run, "aura", device)
        with torch.no_grad():
            pred = net(torch.as_tensor(P)).numpy()
        sc["aura_threshold"] = float(np.quantile(np.linalg.norm(np.diff(pred, axis=0), axis=1), 0.9))
        mods["aura_net"] = net
    elif variant in ("CheckVLA", "CheckFIFO"):
        h = 8
        idx = [(k, t) for k, e in enumerate(eps) for t in range(len(e["P"]) - h)]

        def check_batch(_):
            pick = [idx[i] for i in rng.integers(len(idx), size=B)]
            return {"x": np.stack([eps[k]["P"][t] for k, t in pick]),
                    "actions": np.stack([eps[k]["action"][t:t + h] for k, t in pick]),
                    "target": np.stack([eps[k]["P"][t + h] for k, t in pick])}

        net = fit(bm.FeaturePredictor(h), check_batch, steps, lr, run, "check", device)
        nominal = [e for e in eps_val + eps if is_nominal(e)] or eps  # CheckVLA calibrates on nominal runs
        res = []
        with torch.no_grad():
            for e in nominal[:50]:
                T = len(e["P"]) - h
                if T <= 0:
                    continue
                x, acts = torch.as_tensor(e["P"][:T]), torch.as_tensor(np.stack([e["action"][t:t + h] for t in range(T)]))
                res.append(np.linalg.norm(net(x, acts).numpy() - e["P"][h:h + T], axis=1))
        res = np.concatenate(res)
        alpha = float(load_config(ROOT / f"configs/pol/variants/{variant}.yaml")["memory"].get("conformal_alpha", 0.1))
        sc.update(check_mu=float(res.mean()), check_sd=float(res.std() + 1e-6),
                  conformal_q=float(np.quantile((res - res.mean()) / (res.std() + 1e-6), 1 - alpha)),
                  still_threshold=float(np.quantile(np.concatenate([np.linalg.norm(e["P"][h:] - e["P"][:-h], axis=1) for e in eps if len(e["P"]) > h]), 0.2)))
        mods["check_net"] = net
    elif variant == "AGM":
        X, y = [], []
        for it in items_tr:
            if it["kind"] == "R+" and it["failed"] is not None:
                pooled = np.asarray(store.pooled(it["task"], it["seed"]), np.float32)
                X.append(proj.transform(pooled[min(len(pooled) - 1, it["t"] + 8)].reshape(1, -1))[0])
                y.append(not it["failed"])
        X, y = np.asarray(X, np.float32), np.asarray(y)
        mods["verify_head"] = fit(bm.VerifyHead(), lambda s: (lambda i: {"x": X[i], "label": y[i]})(rng.integers(len(X), size=min(B, len(X)))),
                                  steps, lr, run, "agm", device)
    elif variant == "Markov":
        X, y = markov_xy(items_tr, proj)
        mods["markov_head"] = fit(bm.MarkovHead(), lambda s: (lambda i: {"x": X[i], "label": y[i]})(rng.integers(len(X), size=min(B, len(X)))),
                                  steps, lr, run, "markov", device)
    return mods, kw, sc, {}, {}


def is_nominal(ep):
    """No injected failure anywhere in the episode (evaluator log)."""
    return not any(r.get("failure") or r.get("spill_event") or r.get("reading_flip") or r.get("stick_event")
                   for r in ep["logs"])


def markov_xy(items, proj):
    """Markov checker data: frame + proprio just after the re-encounter, labelled with the gate target."""
    sel = [it for it in items if it["has_v2"] and it["target"] is not None]
    X = np.stack([np.r_[proj.transform(it["markov_pooled"].reshape(1, -1))[0], it["markov_proprio"]] for it in sel]).astype(np.float32) if sel else np.zeros((0, 57), np.float32)
    return X, np.array([it["target"] for it in sel], bool)


# --------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/pol/stage_a.yaml"))
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--features", default=str(ROOT / "data/features"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=None, help="POL steps (default: config)")
    p.add_argument("--baseline-steps", type=int, default=3000)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="POL event batch (default: config; lower on a shared GPU)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    overrides = {**({"model.M": a.M} if a.M else {}), **({"train.batch_size": a.batch_size} if a.batch_size else {})}
    cfg = load_config(a.config, overrides or None)
    a.steps = a.steps or cfg["train"]["steps"]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = FeatureStore(a.features)
    proj = PCA.load(store.root / "proj50.npz")
    task_cfgs = {t: load_config(ROOT / f"configs/pol/tasks/{t}.yaml") for t in a.tasks}
    prov = provenance(cfg, ROOT, {"variant": a.variant, "tasks": a.tasks, "seed": a.seed,
                                  "task_configs": task_cfgs})
    run = init_run("stage_a", a.variant, cfg, prov, K=8, seed=a.seed, log_every=cfg["train"]["log_every"], out_dir=a.out)
    items_tr = stage_a_items(a.data, store, a.tasks, "train", task_cfgs)
    items_va = stage_a_items(a.data, store, a.tasks, "val", task_cfgs)
    metrics = {"events_train": len(items_tr), "events_val": len(items_va)}
    if a.variant == "POL":
        mods, kw, sc, arr, m = train_pol(a, cfg, items_tr, items_va, run, device)
    elif a.variant in LEARNED:
        mods, kw, sc, arr, m = train_baseline(a, a.variant, cfg, store, proj, items_tr, run, device)
    else:
        mods, kw, sc, arr, m = {}, {}, {}, {}, {}
    metrics.update(m)
    save_assets(a.out, a.variant, mods, kw, sc, arr, {"provenance": prov, "metrics": metrics})
    Path(a.out, "metrics.json").write_text(json.dumps({"provenance": prov, **metrics, **sc}, indent=1, default=str))
    run.summary({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    run.finish()
    print(json.dumps({"variant": a.variant, **{k: v for k, v in metrics.items()}, **sc}, default=str))


if __name__ == "__main__":
    main()
