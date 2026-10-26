"""Extract only the SigLIP vision tower of pi0 from the openpi JAX checkpoint (policy venv).

    python scripts/pol/extract_siglip.py --ckpt gs://openpi-assets/checkpoints/pi0_base --out checkpoints/pi0_siglip.pt

Only the PaliGemma/img subtree is restored (orbax partial restore; with a gs:// path TensorStore reads
just those chunks), so this needs ~2 GB of RAM instead of the ~13 GB of the full conversion. The
mapping follows openpi's examples/convert_jax_model_to_pytorch.py (vision part) at the pinned commit.
The pooling head (img/head) is the multimodal projector of PaliGemma and is not kept.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def restore_img(ckpt):
    import orbax.checkpoint as ocp

    path = f"{ckpt.rstrip('/')}/params"
    if path.startswith("gs://"):
        # The released bucket has no commit_success.txt, which orbax requires for gs:// paths; the
        # checkpoint is a published, finalized release, so the check is skipped for it only.
        from orbax.checkpoint._src.checkpointers import checkpointer as _ck

        _ck.utils.is_checkpoint_finalized = lambda _path: True
    with ocp.PyTreeCheckpointer() as ck:
        meta = ck.metadata(path)
        item = {"params": {"PaliGemma": {"img": meta["params"]["PaliGemma"]["img"]}}}
        args = jax_tree_map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item)
        out = ck.restore(path, ocp.args.PyTreeRestore(item=item, restore_args=args, transforms={}))
    return flatten(out["params"]["PaliGemma"]["img"])


def jax_tree_map(fn, tree):
    import jax

    return jax.tree.map(fn, tree)


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = np.asarray(v)
    return {k[:-6] if k.endswith("/value") else k: v for k, v in out.items()}


def to_torch_state(img, hidden=1152, layers=27):
    import torch

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    enc = "Transformer/encoderblock"
    att = f"{enc}/MultiHeadDotProductAttention_0"
    sd = {
        "vision_model.embeddings.patch_embedding.weight": t(img["embedding/kernel"].transpose(3, 2, 0, 1)),
        "vision_model.embeddings.patch_embedding.bias": t(img["embedding/bias"]),
        "vision_model.embeddings.position_embedding.weight": t(img["pos_embedding"].reshape(-1, hidden)),
        "vision_model.post_layernorm.weight": t(img["Transformer/encoder_norm/scale"]),
        "vision_model.post_layernorm.bias": t(img["Transformer/encoder_norm/bias"]),
    }
    for i in range(layers):
        p = f"vision_model.encoder.layers.{i}"
        sd[f"{p}.layer_norm1.weight"] = t(img[f"{enc}/LayerNorm_0/scale"][i])
        sd[f"{p}.layer_norm1.bias"] = t(img[f"{enc}/LayerNorm_0/bias"][i])
        sd[f"{p}.layer_norm2.weight"] = t(img[f"{enc}/LayerNorm_1/scale"][i])
        sd[f"{p}.layer_norm2.bias"] = t(img[f"{enc}/LayerNorm_1/bias"][i])
        sd[f"{p}.mlp.fc1.weight"] = t(img[f"{enc}/MlpBlock_0/Dense_0/kernel"][i].transpose())
        sd[f"{p}.mlp.fc1.bias"] = t(img[f"{enc}/MlpBlock_0/Dense_0/bias"][i])
        sd[f"{p}.mlp.fc2.weight"] = t(img[f"{enc}/MlpBlock_0/Dense_1/kernel"][i].transpose())
        sd[f"{p}.mlp.fc2.bias"] = t(img[f"{enc}/MlpBlock_0/Dense_1/bias"][i])
        for jax_name, torch_name in [("key", "k_proj"), ("value", "v_proj"), ("query", "q_proj")]:
            sd[f"{p}.self_attn.{torch_name}.weight"] = t(img[f"{att}/{jax_name}/kernel"][i].reshape(-1, hidden).transpose())
            sd[f"{p}.self_attn.{torch_name}.bias"] = t(img[f"{att}/{jax_name}/bias"][i].reshape(-1))
        sd[f"{p}.self_attn.out_proj.weight"] = t(img[f"{att}/out/kernel"][i].reshape(-1, hidden).transpose())
        sd[f"{p}.self_attn.out_proj.bias"] = t(img[f"{att}/out/bias"][i].reshape(-1))
    return sd


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi0_base")
    p.add_argument("--out", default=str(ROOT / "checkpoints/pi0_siglip.pt"))
    a = p.parse_args()
    img = restore_img(a.ckpt)
    sd = to_torch_state(img)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, a.out)
    digest = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    meta = {"source": a.ckpt, "tensors": len(sd), "params": int(sum(v.numel() for v in sd.values())), "sha256": digest}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
