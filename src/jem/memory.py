"""Causal memories for EEM and its baselines.

Every variant sees the same inputs: role features at the previous and current
policy call, the executed action prefix, and optional intra-chunk evidence
features (one vector per executed microstep). Belief-filter variants keep
log-probabilities so numerical underflow can never become a permanent zero.
Semantics of the states are only trustworthy with certified labels.
"""

from dataclasses import dataclass, field
import math
import torch
from torch import nn

L, Q = 4, 3  # location {slot, held, socket, elsewhere}; evidence {untested, working, broken}
J = L * Q  # joint state s = Q * l + q
UNTESTED, WORKING, BROKEN = 0, 1, 2


def reachable_evidence():
    """[Q, Q]: tested evidence may flip between working/broken but never returns to untested."""
    r = torch.ones(Q, Q, dtype=torch.bool)
    r[1:, UNTESTED] = False
    return r


def joint_support():
    q = torch.arange(J) % Q
    return reachable_evidence()[q][:, q]


def transport(state, association):
    """Exact for permutations; marginal approximation for uncertain association."""
    if association.shape != (*state.shape[:2], state.shape[1]):
        raise ValueError("Association shape")
    if not torch.isfinite(association).all() or (association < 0).any():
        raise ValueError("Invalid association")
    one = torch.ones_like(association.sum(-1))
    if not torch.allclose(association.sum(-1), one, atol=1e-5) or not torch.allclose(
        association.sum(-2), one, atol=1e-5
    ):
        raise ValueError("Association must be doubly stochastic")
    return torch.einsum("bij,bjd->bid", association, state)


@dataclass
class MemoryOutput:
    belief: torch.Tensor  # [B,N,J] probabilities used by auxiliary losses and the gate
    tokens: torch.Tensor  # [B,M,D] memory prefix for the action transformer
    aux: dict = field(default_factory=dict)  # variant-specific logits for auxiliary losses


class BaseMemory(nn.Module):
    def __init__(self, dim, evidence_dim=0):
        super().__init__()
        self.dim = dim
        self.evidence_dim = evidence_dim
        self.read = nn.Sequential(nn.Linear(J, dim), nn.LayerNorm(dim))
        self.log_belief = None

    @property
    def belief(self):
        return None if self.log_belief is None else self.log_belief.exp()

    def reset(self, batch, roles, device=None):
        if batch < 1 or roles < 1:
            raise ValueError("Empty memory")
        device = device or next(self.parameters()).device
        self.log_belief = torch.full((batch, roles, J), -math.log(J), device=device)

    def detach(self):
        self.log_belief = self.log_belief.detach()

    def output(self, aux=None):
        belief = self.belief
        return MemoryOutput(belief, self.read(belief), aux or {})

    def validate(self, prior, obs, actions, mask, dt, active, evidence, evidence_mask):
        if self.log_belief is None:
            raise RuntimeError("Call reset first")
        b, n = self.log_belief.shape[:2]
        if prior.shape != (b, n, self.dim) or obs.shape != prior.shape:
            raise ValueError("Role feature shape")
        if (
            actions.ndim != 3
            or actions.shape[0] != b
            or actions.shape[-1] != 7
            or mask.shape != actions.shape[:2]
            or dt.shape != mask.shape
        ):
            raise ValueError("Executed prefix shape")
        if active.shape != (b,) or mask.dtype != torch.bool or active.dtype != torch.bool:
            raise ValueError("Boolean batch masks required")
        if not all(torch.isfinite(x).all() for x in [prior, obs, actions, dt]) or (dt < 0).any():
            raise ValueError("Nonfinite/negative inputs")
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Executed mask must be a prefix")
        if evidence is not None:
            if not self.evidence_dim:
                raise ValueError("Memory built without an evidence stream")
            if evidence.shape != (*mask.shape, self.evidence_dim) or evidence_mask.shape != mask.shape:
                raise ValueError("Evidence must align with executed microsteps")
            if evidence_mask.dtype != torch.bool or (evidence_mask & ~mask).any():
                raise ValueError("Evidence only for executed microsteps")
            if not torch.isfinite(evidence).all():
                raise ValueError("Nonfinite evidence")

    def action_features(self, prior, action, dt):
        n = prior.shape[1]
        return torch.cat([prior, action[:, None].expand(-1, n, -1), dt[:, None, None].expand(-1, n, 1)], -1)

    def evidence_input(self, prior, evidence):
        return torch.cat([prior, evidence[:, None].expand(-1, prior.shape[1], -1)], -1)


def _where(cond, new, old):
    return torch.where(cond.reshape(-1, *([1] * (new.ndim - 1))), new, old)


class JointMemory(BaseMemory):
    """Log-space Bayes-style filter: predict with executed actions, correct with every frame."""

    def __init__(self, dim, evidence_dim=0):
        super().__init__(dim, evidence_dim)
        self.correction = nn.Linear(dim, J)
        self.evidence_correction = nn.Linear(dim + evidence_dim, J) if evidence_dim else None
        self.register_buffer("support", joint_support(), persistent=False)

    def log_kernel(self, x):  # [..., J, J], rows sum to one in probability space
        raise NotImplementedError

    def predict(self, log_b, x):
        return torch.logsumexp(log_b[..., :, None] + self.log_kernel(x), dim=-2)

    @staticmethod
    def correct(log_b, score):
        out = log_b + score.float()
        return out - torch.logsumexp(out, -1, keepdim=True)

    def update(self, prior, obs, actions, mask, dt, active, evidence=None, evidence_mask=None,
               observed=None, association=None, **_):
        self.validate(prior, obs, actions, mask, dt, active, evidence, evidence_mask)
        observed = active if observed is None else observed & active
        old = self.log_belief
        log_b = old
        if association is not None:
            log_b = transport(log_b.exp(), association).clamp_min(1e-30).log()
        for j in range(actions.shape[1]):
            step = mask[:, j] & active
            log_b = _where(step, self.predict(log_b, self.action_features(prior, actions[:, j], dt[:, j]).float()), log_b)
            if evidence is not None:
                score = self.evidence_correction(self.evidence_input(prior, evidence[:, j]))
                log_b = _where(step & evidence_mask[:, j], self.correct(log_b, score), log_b)
        log_b = _where(observed, self.correct(log_b, self.correction(obs)), log_b)
        self.log_belief = _where(active, log_b, old)
        return self.output()


class DirectJointMemory(JointMemory):
    """B2: unconstrained linear transition logits on the shared support."""

    def __init__(self, dim, evidence_dim=0):
        super().__init__(dim, evidence_dim)
        self.transition = nn.Linear(dim + 8, J * J)

    def log_kernel(self, x):
        logits = self.transition(x).reshape(*x.shape[:-1], J, J)
        return logits.masked_fill(~self.support, -torch.inf).log_softmax(-1)


class DirectJointMLP(DirectJointMemory):
    """B2-MLP: same support, nonlinear transition logits (function-class control)."""

    def __init__(self, dim, evidence_dim=0, hidden=256):
        super().__init__(dim, evidence_dim)
        self.transition = nn.Sequential(nn.Linear(dim + 8, hidden), nn.GELU(), nn.Linear(hidden, J * J))


class EEMMemory(JointMemory):
    """EEM: identifiable chain-rule kernel K(l',q'|s) = rho(q'|s) * D(l'|s,q').

    rho ranges only over evidence values reachable from the source state, so the
    factorization of any positive legal row is unique and rho aligns with labelled
    evidence transitions (unlike a latent event marginalised before correction).
    """

    def __init__(self, dim, evidence_dim=0):
        super().__init__(dim, evidence_dim)
        self.rho = nn.Linear(dim + 8, J * Q)
        self.physical = nn.Linear(dim + 8, J * Q * L)
        q = torch.arange(J) % Q
        self.register_buffer("reach", reachable_evidence()[q], persistent=False)  # [J, Q]

    def factors(self, x):
        rho = self.rho(x).reshape(*x.shape[:-1], J, Q).masked_fill(~self.reach, -torch.inf).log_softmax(-1)
        phys = self.physical(x).reshape(*x.shape[:-1], J, Q, L).log_softmax(-1)
        return rho, phys

    def log_kernel(self, x):
        rho, phys = self.factors(x)
        return (rho[..., None, :] + phys.transpose(-1, -2)).reshape(*x.shape[:-1], J, J)


class GRUMemory(BaseMemory):
    """Recurrent baseline with the same belief readout and auxiliary contract."""

    def __init__(self, dim, evidence_dim=0):
        super().__init__(dim, evidence_dim)
        self.cell = nn.GRUCell(dim + 8, dim)
        self.observe = nn.GRUCell(dim, dim)
        self.evidence_cell = nn.GRUCell(dim + evidence_dim, dim) if evidence_dim else None
        self.joint = nn.Linear(dim, J)
        self.hidden = None

    def reset(self, batch, roles, device=None):
        super().reset(batch, roles, device)
        self.hidden = self.log_belief.new_zeros((batch, roles, self.dim))

    def detach(self):
        super().detach()
        self.hidden = self.hidden.detach()

    def update(self, prior, obs, actions, mask, dt, active, evidence=None, evidence_mask=None,
               observed=None, association=None, **_):
        self.validate(prior, obs, actions, mask, dt, active, evidence, evidence_mask)
        observed = active if observed is None else observed & active
        h = self.hidden if association is None else transport(self.hidden, association)

        def step(cell, x, h):
            return cell(x.flatten(0, 1), h.flatten(0, 1)).reshape_as(h)

        for j in range(actions.shape[1]):
            live = mask[:, j] & active
            h = _where(live, step(self.cell, self.action_features(prior, actions[:, j], dt[:, j]), h), h)
            if evidence is not None:
                h = _where(live & evidence_mask[:, j], step(self.evidence_cell, self.evidence_input(prior, evidence[:, j]), h), h)
        h = _where(observed, step(self.observe, obs, h), h)
        self.hidden = _where(active, h, self.hidden)
        self.log_belief = _where(active, self.joint(self.hidden).float().log_softmax(-1), self.log_belief)
        return self.output()


class KeyframeMemory(BaseMemory):
    """KEM baseline in the style of EventVLA: write salient microstep evidence to a FIFO buffer.

    A keyframe head scores each executed microstep; frames above the commit threshold
    (or teacher-forced keyframe labels during training) are written. Roles read the
    buffer by attention. There is no transition model, so evidence that never forms a
    salient event (a lamp that stays off) is only seen if the policy-call frame shows it.
    The joint head is a decoder for the shared auxiliary contract, not a filter.
    """

    def __init__(self, dim, evidence_dim=0, capacity=5, commit=0.55):
        super().__init__(dim, evidence_dim)
        if not evidence_dim:
            raise ValueError("KEM needs the intra-chunk evidence stream")
        self.capacity, self.commit = capacity, commit
        self.keyframe = nn.Linear(evidence_dim, 1)
        self.key = nn.Linear(evidence_dim, dim)
        self.value = nn.Linear(evidence_dim, dim)
        self.query = nn.Linear(dim, dim)
        self.joint = nn.Linear(2 * dim, J)
        self.buffer = self.valid = None

    def reset(self, batch, roles, device=None):
        super().reset(batch, roles, device)
        self.buffer = self.log_belief.new_zeros((batch, self.capacity, self.evidence_dim))
        self.valid = torch.zeros(batch, self.capacity, dtype=torch.bool, device=self.log_belief.device)

    def detach(self):
        super().detach()
        self.buffer = self.buffer.detach()

    def write(self, item, keep):
        shifted = torch.cat([self.buffer[:, 1:], item[:, None]], 1)
        valid = torch.cat([self.valid[:, 1:], torch.ones_like(self.valid[:, :1])], 1)
        self.buffer = _where(keep, shifted, self.buffer)
        self.valid = _where(keep, valid, self.valid)

    def update(self, prior, obs, actions, mask, dt, active, evidence=None, evidence_mask=None,
               observed=None, association=None, keyframe_teacher=None, **_):
        self.validate(prior, obs, actions, mask, dt, active, evidence, evidence_mask)
        logits = []
        if evidence is not None:
            for j in range(actions.shape[1]):
                logit = self.keyframe(evidence[:, j]).squeeze(-1)
                logits.append(logit)
                decide = logit.sigmoid() >= self.commit if keyframe_teacher is None else keyframe_teacher[:, j]
                self.write(evidence[:, j], decide & evidence_mask[:, j] & mask[:, j] & active)
        q = self.query(obs)
        k, v = self.key(self.buffer), self.value(self.buffer)
        att = torch.einsum("bnd,bcd->bnc", q, k) / math.sqrt(self.dim)
        att = att.masked_fill(~self.valid[:, None], -torch.inf)
        read = torch.where(self.valid.any(1)[:, None, None], att.softmax(-1).nan_to_num(0.0), torch.zeros_like(att))
        context = torch.einsum("bnc,bcd->bnd", read, v)
        log_b = self.joint(torch.cat([obs, context], -1)).float().log_softmax(-1)
        self.log_belief = _where(active, log_b, self.log_belief)
        aux = {"keyframe_logits": torch.stack(logits, 1)} if logits else {}
        return MemoryOutput(self.belief, obs + context, aux)


class RecurrentTokenMemory(BaseMemory):
    """MU-lite: M recurrent memory tokens updated by attention to executed steps.

    Captures the idea of µVLA (recurrent tokens carried across calls) outside the
    language model; it is not a reproduction of µVLA, which routes tokens through
    the OFT transformer itself.
    """

    def __init__(self, dim, evidence_dim=0, tokens=8):
        super().__init__(dim, evidence_dim)
        self.init = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.step_in = nn.Linear(dim + 8 + evidence_dim, dim)
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.role_read = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.joint = nn.Linear(dim, J)
        self.memory = None

    def reset(self, batch, roles, device=None):
        super().reset(batch, roles, device)
        self.memory = self.init[None].expand(batch, -1, -1).to(self.log_belief.device).clone()

    def detach(self):
        super().detach()
        self.memory = self.memory.detach()

    def absorb(self, inputs, live):
        upd, _ = self.attn(self.memory, inputs, inputs, need_weights=False)
        self.memory = _where(live, self.norm(self.memory + upd), self.memory)

    def update(self, prior, obs, actions, mask, dt, active, evidence=None, evidence_mask=None,
               observed=None, association=None, **_):
        self.validate(prior, obs, actions, mask, dt, active, evidence, evidence_mask)
        pooled = prior.mean(1)
        for j in range(actions.shape[1]):
            parts = [pooled, actions[:, j], dt[:, j, None]]
            if self.evidence_dim:
                ev = evidence[:, j] if evidence is not None else pooled.new_zeros(len(pooled), self.evidence_dim)
                if evidence is not None:
                    ev = ev * evidence_mask[:, j, None]
                parts.append(ev)
            self.absorb(self.step_in(torch.cat(parts, -1))[:, None], mask[:, j] & active)
        self.absorb(obs, active)
        read, _ = self.role_read(obs, self.memory, self.memory, need_weights=False)
        self.log_belief = _where(active, self.joint(read).float().log_softmax(-1), self.log_belief)
        return MemoryOutput(self.belief, torch.cat([self.memory, read], 1), {})


class RoleAttention(nn.Module):
    """Learned roles with causal query context and a null token; not certified IDs."""

    def __init__(self, dim, roles=6):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(roles, dim) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, spatial, previous):
        if spatial.ndim != 3 or previous.shape != (spatial.shape[0], len(self.queries), spatial.shape[-1]):
            raise ValueError("Role frontend shape")
        keys = torch.cat([spatial, self.null.expand(spatial.shape[0], -1, -1)], 1)
        query = self.queries[None] + previous
        weights = torch.einsum("bnd,bpd->bnp", query, self.key(keys)) / (spatial.shape[-1] ** 0.5)
        weights = weights.softmax(-1)
        return torch.einsum("bnp,bpd->bnd", weights, self.value(keys)), weights


MEMORIES = {
    "GRU": GRUMemory,
    "B2": DirectJointMemory,
    "B2MLP": DirectJointMLP,
    "EEM": EEMMemory,
    "KEM": KeyframeMemory,
    "MU": RecurrentTokenMemory,
}
NEEDS_EVIDENCE = {"KEM"}
