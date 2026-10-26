"""Targets stay outside the policy. Uncertified auxiliary labels are rejected.

Every memory variant receives the same auxiliary contract: joint/marginal NLL on
its belief, gate cross-entropy when a gate is present, and keyframe BCE for KEM.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import torch
from torch.nn import functional as F
from .memory import J, L, Q

LABEL_SCHEMA = "eem-env-log-v1"


@dataclass
class Targets:
    joint: torch.Tensor  # [B,N] in [0,J) or -1 unknown
    location: torch.Tensor  # [B,N] in [0,L) or -1
    evidence: torch.Tensor  # [B,N] in [0,Q) or -1
    mode: torch.Tensor | None = None  # [B] in [0,N] or -1 (0 = act, i+1 = verify role i)
    keyframes: torch.Tensor | None = None  # [B,T] in {0,1} or -1

    def validate(self, belief):
        shape = belief.shape[:-1]
        for x, n in [(self.joint, J), (self.location, L), (self.evidence, Q)]:
            if x.shape != shape or ((x < -1) | (x >= n)).any() or x.dtype not in (torch.int32, torch.int64):
                raise ValueError("Invalid label shape/range/dtype")
        if self.mode is not None and (self.mode.shape != shape[:1] or ((self.mode < -1) | (self.mode > shape[1])).any()):
            raise ValueError("Invalid mode labels")
        if self.keyframes is not None and ~torch.isin(self.keyframes, self.keyframes.new_tensor([-1, 0, 1])).all():
            raise ValueError("Invalid keyframe labels")


def mean_masked(loss, mask):
    return torch.where(mask, loss, torch.zeros_like(loss)).sum() / mask.sum().clamp_min(1)


def auxiliary_loss(output, targets, valid):
    targets.validate(output.belief)
    if valid.shape != (output.belief.shape[0],) or valid.dtype != torch.bool:
        raise ValueError("Valid shape")
    b = output.belief
    known = valid[:, None]
    joint_mask = known & (targets.joint >= 0)

    def nll(prob, index, mask):
        p = prob.gather(-1, index.clamp_min(0)[..., None]).squeeze(-1)
        return mean_masked(-p.clamp_min(1e-12).log(), mask)

    pair = b.reshape(*b.shape[:-1], L, Q)
    terms = {
        "joint": nll(b, targets.joint, joint_mask),
        # Marginals only where the joint is unknown, so no label is counted twice.
        "location": nll(pair.sum(-1), targets.location, known & (targets.location >= 0) & ~joint_mask),
        "evidence": nll(pair.sum(-2), targets.evidence, known & (targets.evidence >= 0) & ~joint_mask),
    }
    if "mode_logits" in output.aux and targets.mode is not None:
        m = valid & (targets.mode >= 0)
        ce = F.cross_entropy(output.aux["mode_logits"].float(), targets.mode.clamp_min(0), reduction="none")
        terms["mode"] = mean_masked(ce, m)
    if "keyframe_logits" in output.aux and targets.keyframes is not None:
        logits = output.aux["keyframe_logits"].float()
        k = targets.keyframes[:, : logits.shape[1]]
        m = valid[:, None] & (k >= 0)
        bce = F.binary_cross_entropy_with_logits(logits, k.clamp_min(0).float(), reduction="none")
        terms["keyframe"] = mean_masked(bce, m)
    return terms


def action_loss(predicted, target, mask, valid):
    if target.shape != predicted.shape or mask.shape != predicted.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Action loss schema")
    return mean_masked((predicted - target).abs(), (mask & valid[:, None])[..., None].expand_as(predicted))


def verify_certificate(path, dataset_revision, base_dir):
    """Require reviewed provenance and hash-matching evidence, not a config boolean."""
    d = json.loads(Path(path).read_text())
    if (
        d.get("review_status") != "verified"
        or d.get("dataset_revision") != dataset_revision
        or d.get("source_kind") not in {"annotation", "replay", "env_log"}
    ):
        raise RuntimeError("G3: trustworthy labels not verified")
    if not d.get("observability_verified") or d.get("label_schema") != LABEL_SCHEMA:
        raise RuntimeError("G3: label schema/observability audit pending")
    if not d.get("evidence_files"):
        raise RuntimeError("G3: missing provenance evidence")
    base = Path(base_dir).resolve()
    for item in d["evidence_files"]:
        f = (base / item["path"]).resolve()
        if not f.is_relative_to(base) or hashlib.sha256(f.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("G3: evidence hash mismatch")
    return d
