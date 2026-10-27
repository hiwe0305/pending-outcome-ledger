from pathlib import Path

import pytest

from pol.config import load_config, provenance, stable_hash

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "configs/pol/tasks"


def test_base_merge_and_dotted_override(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {x: 1, y: 2}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nb: {y: 3}\nc: 4\n")
    cfg = load_config(tmp_path / "child.yaml", {"b.x": 9, "d.e": "new"})
    assert cfg == {"a": 1, "b": {"x": 9, "y": 3}, "c": 4, "d": {"e": "new"}}


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert len(stable_hash({})) == 16


def test_task_configs_match_spec():
    bs3 = load_config(TASKS / "BatteriesSort-3.yaml")
    assert bs3["lamp_mode"] == "flash" and bs3["deferred"] is True
    assert bs3["injection"] == pytest.approx({"spill": 0.2, "occlusion": 0.15, "flip": 0.05, "stick": 0.05})
    assert bs3["heldout"] == ["stick"] and bs3["revealed"] is False
    sb5 = load_config(TASKS / "SealedBin-5.yaml")
    assert sb5["injection"] == pytest.approx({"deflect_side": 0.1, "deflect_behind": 0.1, "cue_hit": 0.9,
                                              "cue_false": 0.1, "cue_occluded": 0.2, "release_occluded": 0.05})
    assert sb5["heldout"] == ["release_occluded"]
    gi3 = load_config(TASKS / "GatherInject-3.yaml")
    assert gi3["revealed"] is True and gi3["injection"]["slip"] == 0.15
    for name in ["BatteriesSort-6", "SealedBin-3", "GatherInject-5"]:
        assert load_config(TASKS / f"{name}.yaml")["n_objects"] in (3, 5, 6)


def test_stage_defaults_match_spec():
    a = load_config(ROOT / "configs/pol/stage_a.yaml")
    assert a["model"]["M"] == 6 and a["model"]["tau"] == 0.1
    assert a["train"] == {"lr": 3e-4, "weight_decay": 0.01, "batch_size": 512, "steps": 20000,
                          "warmup": 1000, "lam_g": 1.0, "lam_u": 0.1, "lam_i": 1.0, "lam_i1": 1.0, "hard_negatives": 32,
                          "log_every": 20, "val_every": 1000, "seed": 0}
    b = load_config(ROOT / "configs/pol/stage_b.yaml")
    assert b["model"]["action_horizon"] == 16 and b["model"]["exec_steps"] == 8
    assert b["optim"]["lr_expert"] == 5e-5 and b["optim"]["lr_memory"] == 1e-4
    assert b["train"]["steps"] == 20000 and b["train"]["batch_size"] == 64
    assert b["train"]["checkpoint_candidates"] == [10000, 15000, 20000]


def test_every_variant_config_loads():
    names = {p.stem for p in (ROOT / "configs/pol/variants").glob("*.yaml")}
    assert names == {"B0", "Frames", "Random", "Markov", "AURA", "StateMem", "CheckVLA",
                     "CheckFIFO", "AGM", "KEM", "MU", "POL", "Oracle"}
    for n in names:
        cfg = load_config(ROOT / f"configs/pol/variants/{n}.yaml")
        assert cfg["memory"]["name"] == n and cfg["memory"]["K"] == 8


def test_provenance_fields():
    p = provenance({"x": 1}, ROOT, {"seed": 17})
    assert set(p) >= {"config_hash", "code_hash", "openpi_commit", "mikasa_commit", "seed"}
    assert p["openpi_commit"].startswith("215abfb2")
