"""EXPLICIT SYNTHETIC FIXTURES. Never report these as OpenVLA or MIKASA performance."""

from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import Encoded, L1ActionHead
from .data import Episode


class TinyBackbone(nn.Module):
    def __init__(self, dim=16, chunk=8):
        super().__init__()
        self.dim = dim
        self.chunk = chunk
        self.image = nn.Linear(3, dim)
        self.camera = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.language = nn.Embedding(256, dim)
        self.proprio = nn.Linear(7, dim)
        self.action_queries = nn.Parameter(torch.randn(chunk * 7, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, 2, dim * 2, batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.head = L1ActionHead(dim, dim * 2, chunk)

    def encode(self, obs, normalized_proprio):
        p = next(self.parameters())
        x = (
            torch.as_tensor(obs.rgb.copy(), device=p.device).permute(0, 3, 1, 2).float()
            / 255
        )
        pooled = (
            F.adaptive_avg_pool2d(x, (2, 2))
            .permute(0, 2, 3, 1)
            .reshape(len(x), 4, 2, 3)
            .permute(0, 2, 1, 3)
        )
        patches = self.image(pooled) + self.camera[None, :, None]
        spatial = patches.reshape(len(x), 8, self.dim)
        language = torch.stack(
            [
                self.language(
                    torch.tensor(list(s.encode("utf-8")), device=p.device)
                ).mean(0)
                for s in obs.language
            ]
        )[:, None]
        return Encoded(
            spatial,
            language,
            torch.ones(len(x), 1, dtype=torch.bool, device=p.device),
            torch.zeros(len(x), 1, dtype=torch.bool, device=p.device),
            1,
            torch.as_tensor(normalized_proprio, device=p.device, dtype=p.dtype),
        )

    def decode(self, encoded, memory=None):
        parts = [
            encoded.spatial,
            self.proprio(encoded.proprio)[:, None],
            encoded.prompt,
        ]
        if memory is not None:
            parts.append(memory)
        parts.append(self.action_queries[None].expand(len(encoded.spatial), -1, -1))
        seq = torch.cat(parts, 1)
        out = self.transformer(seq)[:, -self.chunk * 7 :]
        return self.head(out)


def synthetic_episode(seed=17, length=25, roles=6):
    rng = np.random.default_rng(seed)
    ep = Episode(
        f"SYNTHETIC:{seed}",
        rng.integers(0, 256, size=(length, 128, 128, 6), dtype=np.uint8),
        rng.normal(size=(length, 7)).astype(np.float32),
        rng.uniform(-1, 1, size=(length, 7)).astype(np.float32),
        np.arange(length, dtype=np.float32) * 0.1,
        "Synthetic fixture: check batteries.",
    )
    # Arbitrary fixture targets verify losses/gradients only; no benchmark semantics.
    labels = {
        t: {
            "joint": np.full(roles, -1, np.int64),
            "location": np.arange(roles, dtype=np.int64) % 4,
            "evidence": np.arange(roles, dtype=np.int64) % 3,
            "mode": int(rng.integers(0, roles + 1)),
            "keyframes": rng.integers(0, 2, size=min(t, 8)).astype(np.int64),
        }
        for t in range(0, length, 8)
    }
    return ep, labels
