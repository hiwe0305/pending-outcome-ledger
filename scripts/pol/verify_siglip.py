"""Check the extracted SigLIP weights against openpi's own JAX SigLIP on the same images (policy venv).

    JAX_PLATFORMS=cpu python scripts/pol/verify_siglip.py --ckpt gs://openpi-assets/checkpoints/pi0_base

Runs the JAX module exactly as pi0 builds it (So400m/14, pool_type none, scan) and the PyTorch
SiglipVisionModel loaded from checkpoints/pi0_siglip.pt, both in float32 on CPU, and compares the
encoder outputs (JAX out["encoded"] vs. torch last_hidden_state). Exit code 1 if they differ.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/pol")]


def main():
    import jax
    import jax.numpy as jnp
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    import openpi.models.siglip as siglip
    from extract_siglip import restore_img
    from pol.features import VISION

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi0_base")
    p.add_argument("--weights", default=str(ROOT / "checkpoints/pi0_siglip.pt"))
    p.add_argument("--atol", type=float, default=2e-3)
    a = p.parse_args()

    import orbax.checkpoint as ocp

    path = f"{a.ckpt.rstrip('/')}/params"
    if path.startswith("gs://"):
        from orbax.checkpoint._src.checkpointers import checkpointer as _ck

        _ck.utils.is_checkpoint_finalized = lambda _path: True
    with ocp.PyTreeCheckpointer() as ck:
        meta = ck.metadata(path)
        item = {"params": {"PaliGemma": {"img": meta["params"]["PaliGemma"]["img"]}}}
        args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item)
        tree = ck.restore(path, ocp.args.PyTreeRestore(item=item, restore_args=args, transforms={}))
    params = jax.tree.map(lambda v: v["value"] if isinstance(v, dict) and set(v) == {"value"} else v,
                          tree["params"]["PaliGemma"]["img"], is_leaf=lambda v: isinstance(v, dict) and set(v) == {"value"})

    rng = np.random.default_rng(0)
    images = rng.uniform(-1, 1, size=(2, 224, 224, 3)).astype(np.float32)
    module = siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="float32")
    _, out = module.apply({"params": params}, jnp.asarray(images), train=False)
    ref = np.asarray(out["encoded"])

    model = SiglipVisionModel(SiglipVisionConfig(**VISION)).eval()
    model.load_state_dict(torch.load(a.weights, weights_only=True), strict=False)
    with torch.no_grad():
        got = model(pixel_values=torch.from_numpy(images).permute(0, 3, 1, 2)).last_hidden_state.numpy()
    diff = float(np.abs(ref - got).max())
    rel = float(np.abs(ref - got).mean() / (np.abs(ref).mean() + 1e-8))
    report = {"shape": list(got.shape), "max_abs_diff": diff, "mean_rel_diff": rel, "ok": diff < a.atol}
    print(json.dumps(report))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
