"""Save trainable weights only; frozen 7B initialization stays pinned externally."""

from pathlib import Path
import torch


def adapter_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_adapter(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(model), path)


def load_adapter(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != names:
        raise RuntimeError("Adapter/trainable-scope mismatch")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or names.intersection(missing):
        raise RuntimeError("Incomplete adapter state")
