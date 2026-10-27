"""Save / load the Stage A outputs of one memory variant in the form MemoryProducer assets expect.

<dir>/assets.pt holds, per module, its class name, constructor kwargs and state dict, plus scalar
thresholds and arrays. load_assets() rebuilds the modules (eval mode, CPU) and adds the pooled
projection of the feature store, so `make_memory(variant, cfg, load_assets(...))` just works.
"""

from pathlib import Path

import numpy as np
import torch

from . import baseline_models as bm
from .features import PCA
from .outcome_model import OutcomeModel

CLASSES = {c.__name__: c for c in [OutcomeModel, bm.KeyframeScorer, bm.RecurrentMemory, bm.ActionPredictor,
                                    bm.FeaturePredictor, bm.StateMemCell, bm.VerifyHead, bm.MarkovHead]}


def save_assets(out, variant, modules, kwargs, scalars, arrays, meta):
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant,
                "modules": {k: (type(m).__name__, kwargs.get(k, {}), m.state_dict()) for k, m in modules.items()},
                "scalars": {k: float(v) for k, v in scalars.items()},
                "arrays": {k: np.asarray(v, np.float32) for k, v in arrays.items()}, "meta": meta},
               Path(out) / "assets.pt")


def project50_torch(proj):
    mean = torch.as_tensor(proj.mean, dtype=torch.float32)
    comp = torch.as_tensor(proj.components, dtype=torch.float32)
    return lambda pooled: (torch.as_tensor(np.asarray(pooled, np.float32)).reshape(-1) - mean) @ comp.T


def load_assets(stage_a_dir, store_root):
    blob = torch.load(Path(stage_a_dir) / "assets.pt", map_location="cpu", weights_only=False)
    proj = PCA.load(Path(store_root) / "proj50.npz")
    assets = {"proj50": proj.components.T.astype(np.float64), "proj_mean": proj.mean.astype(np.float64)}
    for name, (cls, kwargs, state) in blob["modules"].items():
        m = CLASSES[cls](**kwargs)
        for buf in ("bank1", "bank2"):  # reference banks are sized at the end of training
            if buf in state:
                setattr(m, buf, torch.zeros_like(state[buf]))
        m.load_state_dict(state)
        assets[name] = m.eval()
    assets.update(blob["scalars"])
    for k, v in blob["arrays"].items():
        assets[k] = torch.as_tensor(v)
    if "markov_head" in assets:
        head, p50 = assets["markov_head"], project50_torch(proj)

        def markov(pooled, proprio):
            with torch.no_grad():
                x = torch.cat([p50(pooled), torch.as_tensor(np.asarray(proprio, np.float32))])[None]
                return float(torch.sigmoid(head(x))[0])

        assets["markov"] = markov
    return assets
