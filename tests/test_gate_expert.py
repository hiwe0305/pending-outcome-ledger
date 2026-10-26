import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from jem.envs.expert import BatteryBelief, ObservationModel, BROKEN, UNTESTED, WORKING, choose_verify, decide
from jem.gate import EpistemicGate, quality_posterior, voi_mode, voi_verify
from jem.memory import J, L, Q

ROOT = Path(__file__).resolve().parents[1]


def belief_with(q_marginal):
    b = torch.zeros(1, len(q_marginal), J)
    for i, qm in enumerate(q_marginal):
        b[0, i].reshape(L, Q)[0] = torch.tensor(qm)
    return b


def test_voi_prefers_untested_roles_and_respects_cost():
    b = belief_with([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    v = voi_verify(b)
    assert v[0, 0] == pytest.approx(0.5, abs=1e-4) and v[0, 1:].abs().max() < 1e-4
    assert voi_mode(b).item() == 1  # verify role 0
    assert voi_mode(b, cost=0.6).item() == 0  # act when testing is too expensive
    assert voi_verify(b, spill=1.0)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_quality_posterior_with_noisy_readings():
    q = torch.tensor([[0.0, 1.0, 0.0]])
    assert quality_posterior(q, flip=0.1).item() == pytest.approx(0.9)
    assert quality_posterior(torch.tensor([[1.0, 0, 0]]), p_working=0.3).item() == pytest.approx(0.3)


def test_gate_shapes_and_token():
    g = EpistemicGate(8, 3)
    logits = g(torch.randn(2, 3, 8), torch.randn(2, 8))
    assert logits.shape == (2, 4) and g.token(torch.tensor([0, 3])).shape == (2, 1, 8)


def test_expert_explaining_away_matches_the_abstract_witness():
    m = ObservationModel(spill=0.2, flip=0.0, occlusion=0.0)
    seated = BatteryBelief(m)
    seated.observe(True, False)  # lamp off and the battery was in the socket
    spilled = BatteryBelief(m)
    spilled.observe(False, False)  # lamp off, gripper came back empty
    assert seated.p_working == pytest.approx(0.0) and seated.evidence == BROKEN
    assert spilled.p_working == pytest.approx(0.5) and spilled.evidence == UNTESTED
    assert decide(seated, 2) == "bin" and decide(spilled, 2) == "retest" and decide(spilled, 0) in {"home", "bin"}
    lit = BatteryBelief(m)
    lit.observe(True, True)
    assert lit.p_working == pytest.approx(1.0) and lit.evidence == WORKING


def test_expert_value_of_test_and_deferred_choice():
    m = ObservationModel(spill=0.2, flip=0.1, occlusion=0.2)
    beliefs = [BatteryBelief(m) for _ in range(3)]
    beliefs[0].observe(True, True)
    beliefs[0].observe(True, True)
    beliefs[1].observe(True, None)  # occluded: still untested
    assert beliefs[1].value_of_test() > beliefs[0].value_of_test() >= -1e-9
    assert choose_verify(beliefs, 1) in (1, 2) and choose_verify(beliefs, 0) is None


def test_probe_detects_a_planted_shortcut_and_passes_on_noise():
    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts/probe_shortcuts.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    rgb = rng.integers(0, 256, (200, 128, 128, 6)).astype(np.uint8)
    noise = probe.logistic_probe(probe.features(rgb), y)
    rgb[y == 1, :16, :16] = 255  # leak quality into a corner of the image
    leak = probe.logistic_probe(probe.features(rgb), y)
    assert noise < 0.7 and leak > 0.9


def test_variant_configs_are_buildable():
    from jem.fixtures import TinyBackbone
    from jem.policy import JointPolicy

    for path in sorted((ROOT / "configs/variants").glob("*.json")):
        v = json.loads(path.read_text())
        JointPolicy(TinyBackbone(), v["variant"], evidence=v["evidence"], gate=v["gate"], gate_kwargs=v.get("gate_kwargs"))
