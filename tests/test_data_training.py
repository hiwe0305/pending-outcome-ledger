import numpy as np
import pytest
import torch
from jem.data import (
    Episode,
    TrainNormalizer,
    calls,
    episode_split,
    pack_cameras,
    policy_observation,
)
from jem.fixtures import synthetic_episode, TinyBackbone
from jem.policy import JointPolicy, VARIANTS
from jem.memory import GRUMemory
from jem.supervision import Targets, auxiliary_loss, action_loss, verify_certificate
from jem.sequences import segments, pad_sequences
from jem.training import train_episode


def test_camera_allowlist_and_train_only_normalization():
    top = np.full((1, 128, 128, 3), 11, np.uint8)
    wrist = np.full_like(top, 22)
    obs = policy_observation(
        {
            "rgb": pack_cameras(top, wrist),
            "proprio": np.zeros((1, 7)),
            "oracle_info": {"quality": 123},
            "task_cue": 123,
        },
        ("test",),
    )
    assert obs.rgb[0, 0, 0].tolist() == [11, 11, 11, 22, 22, 22]
    assert not hasattr(obs, "oracle_info") and not hasattr(obs, "task_cue")
    train, _ = synthetic_episode()
    test, _ = synthetic_episode(29)
    train.proprio[:, 0] = 3
    test.proprio[:] = 10000
    scaler = TrainNormalizer.fit({"a": train, "b": test}, ["a"], "proprio")
    assert scaler.low[0] == scaler.high[0] == 3 and scaler.train_ids == ("a",)
    assert (
        np.isfinite(scaler.transform(test.proprio)).all()
        and (scaler.transform(test.proprio)[:, 0] == 0).all()
    )
    x = train.proprio.copy()
    x[:, 0] = 3
    recovered = scaler.inverse(scaler.transform(x))
    assert np.max(np.abs(recovered[:, 0] - 3)) == 0
    action = TrainNormalizer.fit({"a": train}, ["a"], "actions")
    assert action.low[6] == -1 and action.high[6] == 1


def test_split_episode_and_future_target_isolation():
    ids = [f"{task}:{i}" for task in ["easy", "hard"] for i in range(10)]
    split = episode_split(ids)
    assert [len(split[k]) for k in ["train", "val", "test"]] == [16, 2, 2]
    assert not set(split["train"]) & set(split["test"]) and split == episode_split(
        list(reversed(ids))
    )
    ep, _ = synthetic_episode(length=19)
    first = list(calls(ep))[1]
    ep.actions[8:] += 0.01
    ep.actions[8:] = np.clip(ep.actions[8:], -1, 1)
    second = list(calls(ep))[1]
    assert np.array_equal(first.executed, second.executed) and np.array_equal(
        first.observation.rgb, second.observation.rgb
    )
    assert first.executed.shape == (8, 7)
    # One evidence frame per executed action; the last one is the call-time frame.
    assert first.frames.shape == (8, 128, 128, 6) and np.array_equal(first.frames[-1], ep.rgb[8])
    assert list(calls(ep))[0].frames.shape[0] == 0
    last = list(calls(ep))[-1]
    assert last.mask.tolist() == [True] * 3 + [False] * 5
    assert np.allclose(first.dt, 0.1, atol=1e-6)


def test_padding_burnin_and_no_mid_episode_reset():
    items = [torch.arange(5)[:, None].float(), torch.arange(3)[:, None].float()]
    x, valid, loss = pad_sequences(items, 2)
    assert valid.sum() == 8 and loss.sum() == 4 and not loss[:, :2].any()
    schedule = list(segments(9, 2, 3))
    assert [(s.start, s.stop) for s in schedule] == [(0, 2), (2, 5), (5, 8), (8, 9)]
    assert sum(s.reset for s in schedule) == 1 and schedule[1].detach_before


def test_unknown_masks_joint_priority_and_finite_losses():
    m = GRUMemory(4)
    m.reset(1, 2)
    out = m.output()
    unknown = torch.full((1, 2), -1, dtype=torch.long)
    targets = Targets(unknown, unknown, unknown)
    losses = auxiliary_loss(out, targets, torch.ones(1, dtype=torch.bool))
    assert all(v == 0 for v in losses.values())
    targets = Targets(torch.zeros_like(unknown), torch.zeros_like(unknown), torch.zeros_like(unknown))
    losses = auxiliary_loss(out, targets, torch.ones(1, dtype=torch.bool))
    assert losses["joint"] > 0 and losses["location"] == losses["evidence"] == 0
    pred = torch.ones(1, 2, 7, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])
    loss = action_loss(pred, target, mask, torch.ones(1, dtype=torch.bool))
    loss.backward()
    assert pred.grad[:, 1].sum() == 0 and pred.grad[:, 0].sum() > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_causal_bptt_works_and_real_memory_training_gated(variant):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    ep, labels = synthetic_episode()
    policy = JointPolicy(TinyBackbone(), variant)
    opt = torch.optim.Adam(policy.parameters(), lr=0.001)
    h = train_episode(
        policy, ep, opt, labels if variant != "B0" else None, synthetic=True, tbptt=1
    )
    assert (
        h[0]["burn_in"]
        and len(h) == 4
        and np.isfinite([r["losses"]["action"] for r in h]).all()
    )
    if variant != "B0":
        with pytest.raises(RuntimeError, match="G3"):
            train_episode(policy, ep, opt)


@pytest.mark.parametrize("variant", VARIANTS)
def test_action_depends_only_on_available_history(variant):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    p = JointPolicy(TinyBackbone(), variant)
    p.eval()
    ep, _ = synthetic_episode()
    streams = list(calls(ep))
    p.reset()
    left = []
    with torch.no_grad():
        for c in streams[:2]:
            left.append(p.step(c.observation, c.executed, c.dt, c.frames)[0].clone())
        # Perturb observations and commands strictly after the tested calls.
        ep.rgb[16:] = 255
        ep.actions[16:] = -1
        p.reset()
        right = []
        for c in list(calls(ep))[:2]:
            right.append(p.step(c.observation, c.executed, c.dt, c.frames)[0])
    assert all(torch.equal(a, b) for a, b in zip(left, right))
    if variant == "B0":
        p.reset()
        a = p.step(ep.observation(8), np.zeros((8, 7)), np.ones(8) * 0.1)[0]
        p.reset()
        b = p.step(ep.observation(8), np.ones((8, 7)), np.ones(8) * 0.5)[0]
        assert torch.equal(a, b)


def test_identical_backbone_head_and_role_initialization_across_variants():
    states = {}
    for variant in VARIANTS:
        torch.manual_seed(17)
        p = JointPolicy(TinyBackbone(), variant, evidence=variant != "B0")
        states[variant] = {
            "backbone": p.backbone.state_dict(),
            "role": p.role.state_dict() if p.role is not None else None,
            "context": p.context.state_dict() if p.context is not None else None,
            "evidence": p.evidence.state_dict() if p.evidence is not None else None,
        }
    for variant in VARIANTS[1:]:
        assert all(
            torch.equal(v, states[variant]["backbone"][k])
            for k, v in states["B0"]["backbone"].items()
        )
    for variant in VARIANTS[2:]:
        for part in ("role", "context", "evidence"):
            assert all(
                torch.equal(v, states[variant][part][k])
                for k, v in states["GRU"][part].items()
            )


def test_adapter_checkpoint_roundtrip_only_trainable_weights(tmp_path):
    from jem.checkpointing import save_adapter, load_adapter, adapter_state

    torch.manual_seed(17)
    policy = JointPolicy(TinyBackbone(), "EEM")
    for p in policy.backbone.image.parameters():
        p.requires_grad = False
    state = adapter_state(policy)
    assert "backbone.image.weight" not in state
    path = tmp_path / "adapter.pt"
    save_adapter(policy, path)
    clone = JointPolicy(TinyBackbone(), "EEM")
    for p in clone.backbone.image.parameters():
        p.requires_grad = False
    load_adapter(clone, path)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in state.items())
    with pytest.raises(RuntimeError):
        load_adapter(JointPolicy(TinyBackbone(), "B2"), path)


def test_label_certificate_fails_without_provenance(tmp_path):
    import json

    p = tmp_path / "certificate.json"
    p.write_text(
        json.dumps(
            {
                "review_status": "verified",
                "source_kind": "env_log",
                "dataset_revision": "abc",
                "observability_verified": True,
                "label_schema": "eem-env-log-v1",
                "evidence_files": [],
            }
        )
    )
    with pytest.raises(RuntimeError, match="provenance"):
        verify_certificate(p, "abc", tmp_path)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_memory_receives_current_proprio_and_language(variant):
    from jem.data import PolicyObservation

    torch.manual_seed(17)
    p = JointPolicy(TinyBackbone(), variant)
    p.eval()
    ep, _ = synthetic_episode()
    obs = ep.observation(0)
    with torch.no_grad():
        p.reset()
        original = p.step(obs)[1].belief.clone()
        changed = PolicyObservation(obs.rgb, obs.proprio + 1, obs.language)
        p.reset()
        proprio = p.step(changed)[1].belief.clone()
        changed = PolicyObservation(
            obs.rgb, obs.proprio, ("return every tested battery to its home",)
        )
        p.reset()
        language = p.step(changed)[1].belief.clone()
    assert not torch.allclose(original, proprio) and not torch.allclose(
        original, language
    )


def test_bounded_rgb_pilot_uses_exact_numeric_train_scalers(tmp_path):
    from jem.data import pilot_normalizers
    train, _ = synthetic_episode(17)
    more, _ = synthetic_episode(29)
    episodes = {'a':train,'b':more}
    TrainNormalizer.fit(episodes,['a','b'],'proprio').save(tmp_path/'proprio-normalizer.json')
    TrainNormalizer.fit(episodes,['a','b'],'actions').save(tmp_path/'actions-normalizer.json')
    prop, action = pilot_normalizers({'a':train},['a','b'],tmp_path)
    assert prop.train_ids==action.train_ids==('a','b')
    with pytest.raises(ValueError, match='train IDs'): pilot_normalizers({'a':train},['a','c'],tmp_path)
    with pytest.raises(ValueError): TrainNormalizer(np.zeros(7),np.full(7,np.nan),('a',))
