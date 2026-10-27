"""Small learned modules of the baseline memories. All read the 50-d projection of pooled features
(memories.base.project50) and are trained in Stage A with self-supervised losses, except VerifyHead
(AGM-style), which is label-trained as in AGM. Batch formats are produced by pol.data."""

import torch
from torch import nn
import torch.nn.functional as F

D = 50


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class KeyframeScorer(nn.Module):
    """EventVLA-style keyframe score. Self-supervised label: large feature change between steps."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(D, 64, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def stage_a_loss(self, batch):  # x [B,50], label [B] in {0,1}
        return F.binary_cross_entropy_with_logits(self(batch["x"]), batch["label"].float())


class RecurrentMemory(nn.Module):
    """muVLA-style recurrent memory tokens: GRU over call frames, K tokens read out of the state."""

    def __init__(self, K, hidden=256):
        super().__init__()
        self.K = K
        self.cell = nn.GRUCell(D, hidden)
        self.readout = nn.Linear(hidden, K * D)
        self.future = nn.Linear(hidden, D)
        self.hidden = hidden

    def step(self, x, h):
        return self.cell(x, h)

    def read(self, h):
        return self.readout(h).view(-1, self.K, D)

    def stage_a_loss(self, batch):  # seq [B,T,50]: predict the next call frame
        seq = batch["seq"]
        h = seq.new_zeros(seq.shape[0], self.hidden)
        loss = 0.0
        for t in range(seq.shape[1] - 1):
            h = self.step(seq[:, t], h)
            loss = loss + F.mse_loss(self.future(h), seq[:, t + 1])
        return loss / (seq.shape[1] - 1)


class ActionPredictor(nn.Module):
    """AURA-style: a write happens when the predicted action changes."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(D, 128, 7)

    def forward(self, x):
        return self.net(x)

    def stage_a_loss(self, batch):  # x [B,50], action [B,7]
        return F.mse_loss(self(batch["x"]), batch["action"])


class FeaturePredictor(nn.Module):
    """CheckVLA-style point predictor of the feature `horizon` steps ahead, conditioned on actions."""

    def __init__(self, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.net = _mlp(D + horizon * 7, 256, D)

    def forward(self, x, actions):
        return self.net(torch.cat([x, actions.flatten(1)], -1))

    def stage_a_loss(self, batch):  # x [B,50], actions [B,h,7], target [B,50]
        return F.mse_loss(self(batch["x"], batch["actions"]), batch["target"])


class StateMemCell(nn.Module):
    """StateMem-style single memory token updated by a low-rank residual of the prediction error."""

    def __init__(self, rank=8):
        super().__init__()
        self.predict = nn.Linear(D, D)
        self.U = nn.Parameter(torch.randn(D, rank) * 0.05)
        self.V = nn.Parameter(torch.randn(D, rank) * 0.05)

    def step(self, m, x):
        err = x - self.predict(m)
        return m + (err @ self.V) @ self.U.T

    def stage_a_loss(self, batch):  # seq [B,T,50]
        seq = batch["seq"]
        m = torch.zeros_like(seq[:, 0])
        loss = 0.0
        for t in range(seq.shape[1] - 1):
            m = self.step(m, seq[:, t])
            loss = loss + F.mse_loss(self.predict(m), seq[:, t + 1])
        return loss / (seq.shape[1] - 1)


class VerifyHead(nn.Module):
    """AGM-style placement verifier on the post-release frame (label-trained, as in AGM)."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(D, 128, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def stage_a_loss(self, batch):  # x [B,50], label [B] 1 = placement achieved
        return F.binary_cross_entropy_with_logits(self(batch["x"]), batch["label"].float())


class MarkovHead(nn.Module):
    """Memoryless re-encounter checker: P(the event being re-encountered failed | frame, proprio).
    Label-trained on re-encounter frames; it is the 'Markov' baseline and the N1 gate reference."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(D + 7, 128, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def stage_a_loss(self, batch):  # x [B,57] = proj50 + proprio, label [B] 1 = failed
        return F.binary_cross_entropy_with_logits(self(batch["x"]), batch["label"].float())
