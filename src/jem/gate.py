"""Epistemic gate: choose act or verify(role) from the belief.

Mode 0 is "act" (continue the task); mode i+1 is "verify role i" (re-test it).
EEM-IL learns the choice from belief-expert labels; EEM-VOI computes a myopic
value of information analytically from the evidence marginals.
"""

import torch
from torch import nn
from .memory import L, Q, UNTESTED, WORKING, BROKEN


class EpistemicGate(nn.Module):
    def __init__(self, dim, roles):
        super().__init__()
        self.roles = roles
        self.score = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.act = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.embed = nn.Embedding(roles + 1, dim)

    def forward(self, role_tokens, context):
        """role_tokens [B,N,D], context [B,D] -> mode logits [B,N+1]."""
        if role_tokens.shape[1] != self.roles:
            raise ValueError("Gate expects one token per role")
        n = role_tokens.shape[1]
        verify = self.score(torch.cat([role_tokens, context[:, None].expand(-1, n, -1)], -1)).squeeze(-1)
        return torch.cat([self.act(context), verify], -1)

    def token(self, mode):
        return self.embed(mode)[:, None]


def quality_posterior(evidence_marginal, p_working=0.5, flip=0.0):
    """P(working) per role from evidence marginals [...,Q] under a symmetric reading flip."""
    tw = p_working * (1 - flip) / (p_working * (1 - flip) + (1 - p_working) * flip)
    tb = p_working * flip / (p_working * flip + (1 - p_working) * (1 - flip))
    weights = evidence_marginal.new_tensor([p_working, tw, tb])
    return (evidence_marginal * weights).sum(-1)


def voi_verify(belief, p_working=0.5, flip=0.0, spill=0.0, cost=0.0):
    """Myopic value of one more test per role, in units of P(correct routing).

    belief [...,N,J] joint over (L,Q); returns [...,N]. A test spills with probability
    `spill` (no information) and otherwise yields one reading flipped with `flip`.
    """
    q = belief.reshape(*belief.shape[:-1], L, Q).sum(-2)
    p = quality_posterior(q, p_working, flip).clamp(1e-6, 1 - 1e-6)
    now = torch.maximum(p, 1 - p)
    on = p * (1 - flip) + (1 - p) * flip
    p_on = p * (1 - flip) / on
    p_off = p * flip / (1 - on)
    after = on * torch.maximum(p_on, 1 - p_on) + (1 - on) * torch.maximum(p_off, 1 - p_off)
    return (1 - spill) * after + spill * now - now - cost


def voi_mode(belief, **kw):
    """0 = act, i+1 = verify role i with the largest positive VOI."""
    v = voi_verify(belief, **kw)
    best, idx = v.max(-1)
    return torch.where(best > 0, idx + 1, torch.zeros_like(idx))


__all__ = ["EpistemicGate", "quality_posterior", "voi_verify", "voi_mode", "UNTESTED", "WORKING", "BROKEN"]
