"""Exact OFT prefix/prompt helpers with tiny transformer; real model gate pending."""

import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from jem.backbone import OFTBackbone
from jem.data import PolicyObservation

ROOT = Path(__file__).resolve().parents[1]


class FakeProcessor:
    def __call__(self, prompt, image, return_tensors="pt"):
        x = (
            torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1).float()[None]
            / 255
        )
        return {
            "input_ids": torch.tensor([[1, 101]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": x,
        }


class TinyLanguageModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            dim, 2, dim * 2, batch_first=True, dropout=0.0
        )
        self.last = None

    def forward(self, **kwargs):
        assert (
            kwargs["position_ids"] is None
            and kwargs["past_key_values"] is None
            and not kwargs["use_cache"]
        )
        self.last = kwargs
        hidden = self.layer(kwargs["inputs_embeds"])
        return SimpleNamespace(hidden_states=(hidden,))


class FakeOFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = nn.Embedding(32000, 8)
        self.image = nn.Linear(3, 8)
        self.language_model = TinyLanguageModel(8)

    def get_input_embeddings(self):
        return self.embedding

    def _process_vision_features(self, pixel_values, language_embeddings, use_film):
        assert pixel_values.shape[1] == 6 and use_film is False
        x = (
            F.adaptive_avg_pool2d(pixel_values, (2, 2))
            .permute(0, 2, 3, 1)
            .reshape(1, 4, 2, 3)
            .reshape(1, 8, 3)
        )
        return self.image(x)


def attach_official_helpers():
    source = ROOT / "third_party/oft/prismatic/extern/hf/modeling_prismatic.py"
    if not source.exists():
        pytest.skip("Pinned OFT audit source absent")
    tree = ast.parse(source.read_text())
    names = {
        "_prepare_input_for_action_prediction",
        "_prepare_labels_for_action_prediction",
        "_build_multimodal_attention",
        "_process_action_masks",
        "_regression_or_discrete_prediction",
    }
    methods = [
        m
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for m in c.body
        if getattr(m, "name", None) in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    scope = {
        "torch": torch,
        "ACTION_DIM": 7,
        "NUM_ACTIONS_CHUNK": 8,
        "STOP_INDEX": 2,
        "ACTION_TOKEN_BEGIN_IDX": 31743,
        "IGNORE_INDEX": -100,
    }
    util_source = source.parents[2] / "training/train_utils.py"
    utility = ast.parse(util_source.read_text())
    funcs = ast.Module(
        body=[
            n
            for n in utility.body
            if getattr(n, "name", None)
            in {"get_current_action_mask", "get_next_actions_mask"}
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(funcs), str(util_source), "exec"), scope)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    for name in names:
        setattr(FakeOFT, name, scope[name])


def test_real_oft_helpers_offsets_masks_and_both_gradient_paths():
    torch.set_num_threads(2)
    attach_official_helpers()
    torch.manual_seed(2)
    model = FakeOFT()
    adapter = OFTBackbone(model, FakeProcessor(), 8, 16)
    obs = PolicyObservation(
        np.ones((1, 128, 128, 6), np.uint8) * 20,
        np.zeros((1, 7), np.float32),
        ("check batteries",),
    )
    encoded = adapter.encode(obs, obs.proprio)
    encoded.spatial.retain_grad()
    memory = torch.randn(1, 6, 8, requires_grad=True)
    prediction = adapter.decode(encoded, memory)
    assert prediction.shape == (1, 8, 7)
    mask = model.language_model.last["attention_mask"]
    assert mask.shape == (1, 3 + 57 + 8 + 1 + 6) and mask.all()
    prediction.square().sum().backward()
    assert memory.grad.abs().sum() > 0 and encoded.spatial.grad.abs().sum() > 0
    assert torch.isfinite(memory.grad).all()

    # Numeric token-index landmark proves action extraction moves with memory prefix.
    class Landmark(nn.Module):
        def forward(self, **kwargs):
            x = kwargs["inputs_embeds"]
            index = torch.arange(x.shape[1]).float()[None, :, None].expand_as(x)
            return SimpleNamespace(hidden_states=(index,))

    class Capture(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Identity(), nn.Linear(8, 8))
            self.states = None

        def forward(self, states):
            self.states = states
            return states.reshape(1, 8, 7, 8).mean(-1)

    model.language_model = Landmark()
    adapter.head = Capture()
    adapter.decode(encoded, memory)
    assert adapter.head.states[0, 0, 0] == 17  # spatial8+proprio1+memory6+prompt3-1 (official next-token shift)
    captured_with_memory = adapter.head.states.detach().clone()
    class OfficialHead:
        def predict_action(self, states):
            self.states = states
            return states.new_zeros((1, 8, 7))
    official_head = OfficialHead()
    prefix = torch.cat([encoded.spatial, adapter.proprio(encoded.proprio)[:, None], memory], 1)
    model._regression_or_discrete_prediction(
        encoded.prompt, encoded.action_mask, prefix, encoded.mask, None,
        prefix.shape[1], encoded.prompt_length - 1, official_head,
    )
    assert torch.equal(captured_with_memory, official_head.states)
    adapter.decode(encoded)
    assert adapter.head.states[0, 0, 0] == 11


def test_oft_chunk_mismatch_is_explicit():
    attach_official_helpers()
    adapter = OFTBackbone(FakeOFT(), FakeProcessor(), 4, 16)
    obs = PolicyObservation(
        np.zeros((1, 128, 128, 6), np.uint8), np.zeros((1, 7), np.float32), ("check",)
    )
    with pytest.raises(ValueError, match="chunk constants"):
        adapter.encode(obs, obs.proprio)


@pytest.mark.parametrize("variant", ["GRU", "B2", "B2MLP", "EEM", "KEM", "MU"])
def test_bfloat16_backbone_float32_memory_bridge(variant):
    from jem.policy import JointPolicy

    attach_official_helpers()
    torch.manual_seed(2)
    backbone = OFTBackbone(FakeOFT(), FakeProcessor(), 8, 16).to(dtype=torch.bfloat16)
    policy = JointPolicy(backbone, variant)
    policy.train()
    policy.reset()
    obs = PolicyObservation(
        np.ones((1, 128, 128, 6), np.uint8) * 20,
        np.zeros((1, 7), np.float32),
        ("check batteries",),
    )
    # One executed microstep with its evidence frame, written to KEM's buffer.
    output, memory = policy.step(
        obs, np.zeros((1, 7), np.float32), np.ones(1, np.float32) * 0.1,
        frames=obs.rgb, keyframe_teacher=[[True]],
    )
    assert output.dtype == torch.bfloat16 and memory.belief.dtype == torch.float32
    output.float().square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.memory.parameters()
    )
