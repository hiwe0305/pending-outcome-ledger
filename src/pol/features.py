"""Frozen SigLIP features of pi0's own vision encoder, PCA, and the on-disk FeatureStore.

Layout of a FeatureStore root:
  pca.npz                         mean [1152], components [256, 1152] (fitted on train frames only)
  proj50.npz                      mean [2304], components [50, 2304] (pooled -> 50, for baselines)
  <task>/<seed:06d>.pooled.npy    [T, 2, 1152] float16, mean over 256 patches per camera, every frame
  <task>/<seed:06d>.tokens.npz    idx [n] int32, tok [n, 32, 256] float16: 4x4-pooled patches of both
                                  cameras after PCA, only at policy-call frames and event windows
"""

from pathlib import Path

import numpy as np

VISION = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
              image_size=224, patch_size=14)
GRID, POOLED_GRID = 16, 4


def pool_grid(tokens, out=POOLED_GRID):
    """[N, 256, D] patch tokens on a 16x16 grid -> [N, out*out, D] by average pooling."""
    n, p, d = tokens.shape
    g = int(round(p ** 0.5))
    k = g // out
    return tokens.reshape(n, out, k, out, k, d).mean((2, 4)).reshape(n, out * out, d)


class PCA:
    def __init__(self, mean, components):
        self.mean, self.components = np.asarray(mean, np.float32), np.asarray(components, np.float32)

    @classmethod
    def fit(cls, X, dim, seed=0, max_rows=200_000):
        X = np.asarray(X, np.float32)
        if len(X) > max_rows:
            X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
        mean = X.mean(0)
        _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
        return cls(mean, vt[:dim])

    def transform(self, X):
        return (np.asarray(X, np.float32) - self.mean) @ self.components.T

    def save(self, path):
        np.savez(path, mean=self.mean, components=self.components)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["components"])


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)

    def _base(self, task, seed):
        return self.root / task / f"{int(seed):06d}"

    def pooled(self, task, seed):
        return np.load(f"{self._base(task, seed)}.pooled.npy", mmap_mode="r")

    def tokens(self, task, seed):
        z = np.load(f"{self._base(task, seed)}.tokens.npz")
        return dict(zip(z["idx"].tolist(), z["tok"]))

    def write(self, task, seed, pooled, token_idx, tokens):
        base = self._base(task, seed)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{base}.pooled.npy", np.asarray(pooled, np.float16))
        np.savez(f"{base}.tokens.npz", idx=np.asarray(token_idx, np.int32), tok=np.asarray(tokens, np.float16))

    def has(self, task, seed):
        return Path(f"{self._base(task, seed)}.tokens.npz").exists()


class SiglipExtractor:
    """pi0's SigLIP So400m/14 vision tower (weights from scripts/pol/extract_siglip.py), frozen, fp16 on GPU.

    __call__(rgb [N,128,128,6] uint8, top then wrist) -> pooled [N,2,1152], grid [N,32,1152] (float32 numpy)
    """

    def __init__(self, weights, device="cuda", dtype="float16", batch=64):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        self.torch, self.device, self.batch = torch, device, batch
        self.dtype = getattr(torch, dtype)
        model = SiglipVisionModel(SiglipVisionConfig(**VISION))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "head" not in k]  # the pooling head is not used
        if missing or unexpected:
            raise ValueError(f"SigLIP weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model = model.to(device=device, dtype=self.dtype).eval()

    def _encode(self, images):
        """[N,128,128,3] uint8 -> [N,256,1152]: resize to 224 and scale to [-1, 1] as pi0 does."""
        torch = self.torch
        x = torch.as_tensor(images, device=self.device).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        x = (x / 127.5 - 1.0).to(self.dtype)
        with torch.no_grad():
            return self.model(pixel_values=x).last_hidden_state.float()

    def __call__(self, rgb):
        pooled, grid = [], []
        for s in range(0, len(rgb), self.batch):
            chunk = np.asarray(rgb[s:s + self.batch])
            cams = [self._encode(chunk[..., 3 * c:3 * c + 3]) for c in range(2)]
            pooled.append(np.stack([c.mean(1).cpu().numpy() for c in cams], 1))
            grid.append(np.concatenate([pool_grid(c.cpu().numpy()) for c in cams], 1))
        return np.concatenate(pooled), np.concatenate(grid)
