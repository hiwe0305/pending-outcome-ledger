"""Differentiable adapter for the pinned OFT helpers; no LIBERO action path."""

from dataclasses import dataclass
from pathlib import Path
import importlib, importlib.metadata, json, subprocess, sys
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class L1ActionHead(nn.Module):
    """OFT layout: each action sees its seven transformer action token states."""

    def __init__(self, dim, hidden, chunk=8):
        super().__init__()
        self.chunk = chunk
        self.model = nn.Sequential(
            nn.LayerNorm(dim * 7),
            nn.Linear(dim * 7, hidden),
            nn.ReLU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 7),
        )

    def forward(self, states):
        if states.ndim != 3 or states.shape[1] != self.chunk * 7:
            raise ValueError("Expected K*7 action tokens")
        return self.model(states.reshape(states.shape[0], self.chunk, -1))


@dataclass
class Encoded:
    spatial: torch.Tensor
    prompt: torch.Tensor
    mask: torch.Tensor
    action_mask: torch.Tensor
    prompt_length: int
    proprio: torch.Tensor


class OFTBackbone(nn.Module):
    def __init__(self, model, processor, chunk=8, head_hidden=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.chunk = chunk
        self.dim = model.config.text_config.hidden_size
        self.proprio = nn.Sequential(
            nn.Linear(7, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim)
        )
        self.head = L1ActionHead(self.dim, head_hidden or self.dim, chunk)
        required = [
            "_process_vision_features",
            "_prepare_input_for_action_prediction",
            "_prepare_labels_for_action_prediction",
            "_process_action_masks",
            "_build_multimodal_attention",
        ]
        if any(not hasattr(model, n) for n in required):
            raise TypeError("Incompatible OFT API")

    def encode(self, obs, normalized_proprio):
        """Batch-one OFT processor path; training accumulates episodes causally."""
        if len(obs.rgb) != 1:
            raise ValueError(
                "Pinned OFT prompt adapter supports batch=1; accumulate gradients"
            )
        from PIL import Image

        prompt = (
            f"In: What action should the robot take to {obs.language[0].lower()}?\nOut:"
        )
        inputs = self.processor(
            prompt, Image.fromarray(obs.rgb[0, ..., :3]), return_tensors="pt"
        )
        wrist = self.processor(
            prompt, Image.fromarray(obs.rgb[0, ..., 3:]), return_tensors="pt"
        )
        p = next(self.model.parameters())
        ids = inputs["input_ids"].to(p.device)
        mask = inputs["attention_mask"].to(p.device)
        pixels = torch.cat([inputs["pixel_values"], wrist["pixel_values"]], dim=1).to(
            device=p.device, dtype=p.dtype
        )
        if not torch.all(ids[:, -1] == 29871):
            ids = torch.cat([ids, ids.new_full((len(ids), 1), 29871)], -1)
            mask = torch.cat([mask, mask.new_ones((len(ids), 1))], -1)
        prompt_length = ids.shape[1]
        labels = torch.full_like(ids, -100)
        ids, mask = self.model._prepare_input_for_action_prediction(ids, mask)
        labels = self.model._prepare_labels_for_action_prediction(labels, ids)
        action_mask = self.model._process_action_masks(labels)
        if (action_mask.sum(-1) != self.chunk * 7).any():
            raise ValueError("OFT chunk constants incompatible; no LIBERO fallback")
        embeddings = self.model.get_input_embeddings()(ids)
        language = embeddings[~action_mask].reshape(len(ids), -1, self.dim)
        spatial = self.model._process_vision_features(pixels, language, False)
        proprio = torch.as_tensor(normalized_proprio, device=p.device, dtype=p.dtype)
        return Encoded(spatial, embeddings, mask, action_mask, prompt_length, proprio)

    def decode(self, encoded, memory=None):
        spatial = encoded.spatial
        ptoken = self.proprio(
            encoded.proprio.to(dtype=self.proprio[0].weight.dtype)
        ).to(spatial.dtype)[:, None]
        prefix = torch.cat([spatial, ptoken], 1)
        if memory is not None:
            if memory.shape[0] != len(spatial) or memory.shape[-1] != self.dim:
                raise ValueError("Memory prefix shape")
            prefix = torch.cat([prefix, memory.to(prefix.dtype)], 1)
        prompt = encoded.prompt.masked_fill(encoded.action_mask[..., None], 0)
        embeds, mask = self.model._build_multimodal_attention(
            prompt, prefix, encoded.mask
        )
        out = self.model.language_model(
            input_ids=None,
            attention_mask=mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        offset = prefix.shape[1] + encoded.prompt_length - 1
        states = out.hidden_states[-1][:, offset : offset + self.chunk * 7]
        if states.shape[1] != self.chunk * 7:
            raise RuntimeError("Action indexing failed")
        return self.head(states.to(self.head.model[1].weight.dtype))


def load_local_oft(source_root, checkpoint_root, manifest, chunk=8, for_audit=False):
    """Explicit local-only heavyweight loader, refuses moving/unverified source."""
    root = Path(source_root).resolve()
    checkpoint = Path(checkpoint_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeError(
            "Complete pinned OFT git checkout required; selected audit sources cannot run 7B"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != manifest["sources"]["oft"]["revision"]:
        raise RuntimeError("OFT revision mismatch")
    if not for_audit and not manifest["checkpoint"]["compatibility_verified"]:
        raise RuntimeError(
            "Checkpoint compatibility gate pending; complete audit before allocation"
        )
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError("Local pinned checkpoint missing")
    if checkpoint.name != manifest["checkpoint"]["revision"]:
        raise RuntimeError(
            "Use local HF snapshots/<exact checkpoint revision> to pin initialization"
        )
    dist = importlib.metadata.distribution("transformers")
    direct = dist.read_text("direct_url.json")
    receipt = json.loads(direct) if direct else {}
    if (
        receipt.get("vcs_info", {}).get("commit_id")
        != manifest["sources"]["transformers_oft"]["revision"]
    ):
        raise RuntimeError(
            "Required pinned Transformers OFT fork missing; stock Transformers is not OFT recipe"
        )
    sys.path.insert(0, str(root))
    constants = importlib.import_module("prismatic.vla.constants")
    # Pin MIKASA explicitly, no argv-dependent LIBERO proprio/normalization.
    constants.NUM_ACTIONS_CHUNK = chunk
    constants.ACTION_DIM = 7
    constants.PROPRIO_DIM = 7
    module = importlib.import_module("prismatic.extern.hf.modeling_prismatic")
    module.NUM_ACTIONS_CHUNK = chunk
    module.ACTION_DIM = 7
    cls = module.OpenVLAForActionPrediction
    model = cls.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.vision_backbone.set_num_images_in_input(2)
    processing = importlib.import_module("prismatic.extern.hf.processing_prismatic")
    processor = processing.PrismaticProcessor.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    return OFTBackbone(model, processor, chunk).to(dtype=torch.bfloat16)


def configure_lora(backbone, rank=16):
    """Same explicit tuning recipe for core variants; inspect resulting counts."""
    from peft import LoraConfig, get_peft_model

    for p in backbone.model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    backbone.model.language_model = get_peft_model(backbone.model.language_model, cfg)
    for p in backbone.model.projector.parameters():
        p.requires_grad = True
