"""Episode-causal, bounded training. Real memory supervision requires certified labels (G3)."""

from contextlib import nullcontext
import time
import torch
from .data import calls
from .sequences import segments
from .supervision import Targets, action_loss, auxiliary_loss

LABEL_KEYS = ("joint", "location", "evidence")


def call_targets(item, device):
    """Per-call label dict -> Targets with a batch dimension of one."""
    def tensor(key):
        return torch.as_tensor(item[key], device=device)[None]

    return Targets(
        *[tensor(k) for k in LABEL_KEYS],
        mode=tensor("mode").reshape(1) if "mode" in item else None,
        keyframes=tensor("keyframes") if "keyframes" in item else None,
    )


def train_episode(policy, episode, optimizer, labels=None, burn_in=1, tbptt=4, aux_weight=1.0,
                  synthetic=False, certificate=None):
    if policy.variant != "B0" and not synthetic and (labels is None or certificate is None):
        raise RuntimeError("G3 blocks real memory training without certified labels")
    if labels is not None and not synthetic and certificate is None:
        raise RuntimeError("G3 blocks unverified auxiliary labels")
    stream = list(calls(episode, policy.chunk_size))
    policy.reset()
    policy.train()
    history = []
    device = next(policy.parameters()).device
    for segment in segments(len(stream), burn_in, tbptt):
        if segment.detach_before:
            policy.detach()
        if not segment.burn_in:
            optimizer.zero_grad(set_to_none=True)
        total, scalars = None, {}
        start = time.perf_counter()
        with torch.no_grad() if segment.burn_in else nullcontext():
            for index in range(segment.start, segment.stop):
                call = stream[index]
                item = labels.get(call.t) if labels is not None else None
                mode = item.get("mode") if item else None
                teacher = None
                if item and "keyframes" in item and len(call.executed):
                    teacher = torch.as_tensor(item["keyframes"], device=device)[None].clamp_min(0).bool()
                prediction, out = policy.step(call.observation, call.executed, call.dt, call.frames,
                                              mode=None if mode is None or mode < 0 else [mode],
                                              keyframe_teacher=teacher)
                target = call.target
                if policy.action_normalizer:
                    target = policy.action_normalizer.transform(target)
                target = torch.as_tensor(target[None], device=device, dtype=prediction.dtype)
                mask = torch.as_tensor(call.mask[None], device=device)
                valid = torch.ones(1, device=device, dtype=torch.bool)
                loss = action_loss(prediction, target, mask, valid)
                terms = {"action": loss}
                if item is not None and out is not None:
                    terms.update(auxiliary_loss(out, call_targets(item, device), valid))
                    loss = loss + aux_weight * sum(v for k, v in terms.items() if k != "action")
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                total = loss if total is None else total + loss
                for key, value in terms.items():
                    scalars[key] = scalars.get(key, 0.0) + float(value.detach())
        if not segment.burn_in:
            (total / (segment.stop - segment.start)).backward()
            params = [p for p in policy.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError("Nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        # State persists numerically and is detached only between optimization segments.
        policy.detach()
        n = segment.stop - segment.start
        history.append({"start": segment.start, "stop": segment.stop, "burn_in": segment.burn_in,
                        "losses": {k: v / n for k, v in scalars.items()},
                        "seconds": time.perf_counter() - start})
    return history


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
