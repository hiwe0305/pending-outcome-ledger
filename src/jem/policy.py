"""Shared frontend and action interface. B0 has no history access.

All memory variants receive identical inputs: role features, the executed
action prefix and (when `evidence=True`) one evidence vector per executed
microstep from a shared lightweight frame encoder. The optional epistemic
gate adds one mode token to the memory prefix.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .gate import EpistemicGate, voi_mode
from .memory import MEMORIES, NEEDS_EVIDENCE, RoleAttention

VARIANTS = ("B0", *MEMORIES)


class EvidenceEncoder(nn.Module):
    """Cheap per-microstep encoder (both cameras, 64x64); the 7B backbone sees call frames only."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, frames):
        """frames uint8 [B,T,128,128,6] -> [B,T,dim]."""
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(64, 64), mode="area")
        return self.net(x).reshape(b, t, self.dim)


class JointPolicy(nn.Module):
    def __init__(self, backbone, variant, roles=6, evidence=True, gate=None, evidence_dim=64,
                 proprio_normalizer=None, action_normalizer=None, gate_kwargs=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
        if gate not in (None, "il", "voi") or (variant == "B0" and gate):
            raise ValueError("Gate must be None/'il'/'voi' and needs a memory")
        if variant in NEEDS_EVIDENCE and not evidence:
            raise ValueError(f"{variant} requires the evidence stream")
        self.backbone = backbone
        self.variant = variant
        self.roles = roles
        self.chunk_size = backbone.chunk
        self.gate_kind = gate
        self.gate_kwargs = gate_kwargs or {}
        self.proprio_normalizer = proprio_normalizer
        self.action_normalizer = action_normalizer
        memory = variant != "B0"
        # Frontend built before variant-specific modules so the same seed gives identical frontends.
        self.role = RoleAttention(backbone.dim, roles) if memory else None
        self.context = nn.Sequential(nn.Linear(backbone.dim + 7, backbone.dim), nn.GELU()) if memory else None
        self.evidence = EvidenceEncoder(evidence_dim) if memory and evidence else None
        self.memory = MEMORIES[variant](backbone.dim, evidence_dim if self.evidence else 0) if memory else None
        self.gate = EpistemicGate(backbone.dim, roles) if gate else None
        self.previous = None

    def reset(self, batch=1):
        p = self.role.queries if self.role is not None else next(self.parameters())
        self.previous = p.new_zeros((batch, self.roles, self.backbone.dim))
        if self.memory is not None:
            self.memory.reset(batch, self.roles)

    def detach(self):
        if self.previous is not None:
            self.previous = self.previous.detach()
        if self.memory is not None:
            self.memory.detach()

    def _prefix(self, executed, dt, frames, b, p):
        actions = torch.as_tensor(np.empty((b, 0, 7), np.float32) if executed is None else executed,
                                  device=p.device, dtype=p.dtype)
        actions = actions[None] if actions.ndim == 2 else actions
        elapsed = torch.as_tensor(np.empty((b, 0), np.float32) if dt is None else dt, device=p.device, dtype=p.dtype)
        elapsed = elapsed[None] if elapsed.ndim == 1 else elapsed
        mask = torch.ones(actions.shape[:2], device=p.device, dtype=torch.bool)
        evidence = evidence_mask = None
        if self.evidence is not None and frames is not None and actions.shape[1]:
            x = torch.as_tensor(np.asarray(frames), device=p.device)
            x = x[None] if x.ndim == 4 else x
            if x.shape[:2] != actions.shape[:2]:
                raise ValueError("One evidence frame per executed microstep required")
            evidence = self.evidence(x).to(p.dtype)
            evidence_mask = mask.clone()
        return actions, elapsed, mask, evidence, evidence_mask

    def step(self, obs, executed=None, dt=None, frames=None, mode=None, keyframe_teacher=None, active=None):
        normalized = self.proprio_normalizer.transform(obs.proprio) if self.proprio_normalizer else obs.proprio
        encoded = self.backbone.encode(obs, normalized)
        if self.previous is None:
            raise RuntimeError("Policy must be reset at episode boundary")
        if self.variant == "B0":
            return self.backbone.decode(encoded), None
        # Probability/frontend modules stay float32 even with a bfloat16 7B backbone.
        spatial = encoded.spatial.to(self.role.queries.dtype)
        text_mask = (~encoded.action_mask) & encoded.mask.bool()
        text = (encoded.prompt * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True).clamp_min(1)
        cweight = self.context[0].weight
        context = self.context(torch.cat([text, encoded.proprio.to(text.dtype)], -1).to(cweight.dtype))
        roles, _ = self.role(spatial, self.previous.to(spatial.dtype) + context[:, None].to(spatial.dtype))
        roles = roles + context[:, None].to(roles.dtype)
        b = len(spatial)
        p = next(self.memory.parameters())
        actions, elapsed, mask, evidence, evidence_mask = self._prefix(executed, dt, frames, b, p)
        active = torch.ones(b, device=p.device, dtype=torch.bool) if active is None else active
        teacher = None if keyframe_teacher is None else torch.as_tensor(keyframe_teacher, device=p.device)
        output = self.memory.update(self.previous.to(p.dtype), roles.to(p.dtype), actions, mask, elapsed, active,
                                    evidence=evidence, evidence_mask=evidence_mask, keyframe_teacher=teacher)
        self.previous = torch.where(active[:, None, None], roles.to(p.dtype), self.previous)
        tokens = output.tokens
        if self.gate is not None:
            logits = self.gate(tokens[:, -self.roles:], context.to(tokens.dtype))
            output.aux["mode_logits"] = logits
            if mode is not None:
                chosen = torch.as_tensor(mode, device=p.device).reshape(b)
            elif self.gate_kind == "voi":
                chosen = voi_mode(output.belief, **self.gate_kwargs)
            else:
                chosen = logits.argmax(-1)
            output.aux["mode"] = chosen
            tokens = torch.cat([tokens, self.gate.token(chosen).to(tokens.dtype)], 1)
        return self.backbone.decode(encoded, tokens), output

    @torch.no_grad()
    def inference(self, obs, executed=None, dt=None, frames=None):
        actions, _ = self.step(obs, executed, dt, frames)
        actions = actions.float().cpu().numpy()
        if self.action_normalizer:
            actions = self.action_normalizer.inverse(actions)
        return np.clip(actions, -1, 1).astype(np.float32)
