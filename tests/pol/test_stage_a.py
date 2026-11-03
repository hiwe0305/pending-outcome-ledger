"""Stage A trainer, assets and gates on a tiny synthetic dataset written to disk (CPU, W&B disabled)."""
import json

import numpy as np
import pytest

from pol.gates import decide

PASS = {"unrevealed": {"pol": 0.9, "markov": 0.6, "pol_shuffled": 0.5}, "revealed": {"pol": 0.95, "markov": 0.94},
        "omega_auroc": 0.85}


def test_all_gates_pass():
    v = decide(PASS)
    assert (v["N1"], v["G1"], v["G2"], v["exit_code"]) == (True, True, True, 0)


@pytest.mark.parametrize("change,code,failed", [
    ({"unrevealed": {"pol": 0.9, "markov": 0.85, "pol_shuffled": 0.5}}, 2, "N1"),   # memoryless checker too good
    ({"unrevealed": {"pol": 0.75, "markov": 0.6, "pol_shuffled": 0.5}}, 2, "G1"),   # below 0.8
    ({"unrevealed": {"pol": 0.9, "markov": 0.6, "pol_shuffled": 0.7}}, 2, "G1"),    # works without the event's views
    ({"revealed": {"pol": 0.95, "markov": 0.80}}, 2, "N1"),                     # wins where it must not
    ({"omega_auroc": 0.7}, 3, "G2"),
    ({"omega_auroc": None}, 3, "G2"),
])
def test_each_failure_branch(change, code, failed):
    v = decide({**PASS, **change})
    assert v["exit_code"] == code and not v[failed]


def test_missing_revealed_control_is_noted_not_fatal():
    v = decide({k: v for k, v in PASS.items() if k != "revealed"})
    assert v["exit_code"] == 0 and any("revealed" in n for n in v["notes"])


def test_shuffle_evidence_swaps_views_within_task_and_kind():
    from pol.gates import shuffle_evidence

    items = [{"task": t, "kind": k, "pooled": np.full(2, i), "act": np.full(3, i), "v1": np.full(4, i),
              "v2": np.full(4, i), "w2": np.full(5, i), "pooled2": np.full(2, i), "has_v2": True,
              "act1": np.full(3, i), "act2": np.full(3, i), "pos1": np.full(3, i), "pos2": np.full(3, i)}
             for i, (t, k) in enumerate([("A", "R+")] * 3 + [("A", "R0")] * 2 + [("B", "R+")] * 2 + [("B", "R0")])]
    out = shuffle_evidence(items)
    for it, o in zip(items, out):
        src = int(o["v1"][0])
        assert (items[src]["task"], items[src]["kind"]) == (it["task"], it["kind"])
        assert all(int(o[k][0]) == src for k in ("v2", "w2", "pooled2", "act1", "act2", "pos1", "pos2"))  # the views travel together
        assert int(o["pooled"][0]) == int(it["pooled"][0]) and int(o["act"][0]) == int(it["act"][0])
        lone = sum((x["task"], x["kind"]) == (it["task"], it["kind"]) for x in items) == 1
        assert (src == int(it["v1"][0])) == lone  # never its own views unless it is alone in its group


def test_calibration_is_per_task():
    """z0 means positive in task A and negative in task B: one pooled P(target | z) scores both at chance."""
    from pol.gates import _auroc, calibrate, readout

    items = [{"task": t, "target": y} for t in "AB" for y in (True, False) * 10]
    q = np.array([[1.0, 0.0] if (it["target"] == (it["task"] == "A")) else [0.0, 1.0] for it in items])
    p = calibrate(q, items, ["A", "B"])
    assert p["A"][0] > 0.9 and p["B"][0] < 0.1
    for t in "AB":
        sel = [i for i, it in enumerate(items) if it["task"] == t]
        assert _auroc(readout(q[sel], [items[i] for i in sel], p), [items[i]["target"] for i in sel]) == 1.0


def test_group_score_is_the_mean_over_tasks():
    from pol.gates import macro

    g = macro([{"n": 10, "failures": 2, "pol": 0.9, "pol_shuffled": 0.5, "pol_fake_actions": None, "markov": 0.6},
               {"n": 20, "failures": 8, "pol": 0.7, "pol_shuffled": 0.6, "pol_fake_actions": None, "markov": None}])
    assert (g["n"], g["failures"]) == (30, 10)
    assert g["pol"] == pytest.approx(0.8) and g["pol_shuffled"] == pytest.approx(0.55)
    assert g["markov"] == pytest.approx(0.6) and g["pol_fake_actions"] is None


def write_dataset(root, n_eps=12, T=140, seed=0):
    """SealedBin-3-shaped episodes: grasp, release over the bin, re-encounter; outcome in the logs."""
    from pol.features import PCA, FeatureStore

    rng = np.random.default_rng(seed)
    data, feats = root / "data", root / "features"
    store = FeatureStore(feats)
    (data / "SealedBin-3").mkdir(parents=True)
    pooled_all = []
    for s in range(n_eps):
        w = np.full(T, 0.04)
        w[10:40] = 0.02
        w[70:100] = 0.02
        proprio = np.zeros((T, 7), np.float32)
        proprio[:, 6] = w
        out = int(rng.integers(2)) * 3  # 0 = in the bin, 3 = hidden behind it
        logs = [{"release_cube": -1, "failure": 0, "cube_location": [0, 0, 0], "shutter": bool(rng.random() < 0.3 and 40 <= t < 60)}
                for t in range(T)]
        logs[40]["release_cube"] = 0
        for t in range(41, T):
            logs[t]["cube_location"] = [out, 0, 0]
        np.savez(data / "SealedBin-3" / f"{s:06d}.npz", rgb=np.zeros((T, 128, 128, 6), np.uint8), proprio=proprio,
                 action=rng.normal(size=(T, 7)).astype(np.float32), timestamp=np.arange(T, dtype=np.float32), language="x")
        (data / "SealedBin-3" / f"{s:06d}.log.json").write_text(json.dumps(
            {"seed": s, "split": "train", "task": "SealedBin-3", "logs": logs}))
        pooled = rng.normal(size=(T, 2, 1152)).astype(np.float32)
        tok = rng.normal(size=(T, 32, 256)).astype(np.float32) + (out > 0)
        store.write("SealedBin-3", s, pooled, np.arange(T), tok)
        pooled_all.append(pooled.reshape(T, -1))
    PCA.fit(np.concatenate(pooled_all), 50).save(feats / "proj50.npz")
    (feats / "SealedBin-3" / "fsm.json").write_text(json.dumps({"w_open": 0.035, "w_empty": 0.005,
                                                                  "min_loaded_steps": 3, "min_open_steps": 2}))
    return str(data), str(feats)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("stage_a"))


@pytest.mark.parametrize("variant", ["POL", "Markov", "KEM", "MU", "AURA", "StateMem", "CheckVLA", "AGM", "B0"])
def test_train_save_load_and_produce(variant, dataset, tmp_path, monkeypatch):
    from pol.assets import load_assets
    from pol.memories import make_memory
    from pol.train_stage_a import main

    monkeypatch.setenv("WANDB_MODE", "disabled")
    data, feats = dataset
    out = tmp_path / variant
    main(["--variant", variant, "--data", data, "--features", feats, "--tasks", "SealedBin-3", "--out", str(out),
          "--steps", "6", "--baseline-steps", "4", "--M", "4"])
    assets = load_assets(out, feats)
    if variant == "Markov":
        assert 0.0 <= assets["markov"](np.zeros((2, 1152)), np.zeros(7)) <= 1.0
    else:
        mem = make_memory(variant, {"name": variant, "K": 8}, assets)
        assert mem.tokens()[0].shape == (8, 50)
    assert json.loads((out / "metrics.json").read_text())["events_train"] > 0


def test_gates_end_to_end(dataset, tmp_path, monkeypatch):
    from pol.gates import main as gates_main
    from pol.train_stage_a import main

    monkeypatch.setenv("WANDB_MODE", "disabled")
    data, feats = dataset
    for v in ["POL", "Markov"]:
        main(["--variant", v, "--data", data, "--features", feats, "--tasks", "SealedBin-3", "--out", str(tmp_path / v),
              "--steps", "6", "--baseline-steps", "4", "--M", "4"])
    code = gates_main(["--pol", str(tmp_path / "POL"), "--markov", str(tmp_path / "Markov"), "--data", data,
                       "--features", feats, "--tasks", "SealedBin-3", "--out", str(tmp_path / "gates.json")])
    report = json.loads((tmp_path / "gates.json").read_text())
    assert code in (0, 2, 3) and set(report["verdict"]) >= {"N1", "G1", "G2", "exit_code"}
    v = report["verdict"]
    assert v["G4"] and v["ECE"] and v["mode"] == "legacy"  # POL-V gates never fail an old run
    assert not {"frozen_v1", "agm_same_labels", "ece"} & set(report["scores"]["unrevealed"])
    u = report["scores"]["unrevealed"]
    assert {"n", "failures", "pol", "pol_shuffled", "pol_fake_actions", "markov"} <= set(u)
    assert set(report["scores"]["per_task"]) == {"SealedBin-3"}


POLV_PASS = {"unrevealed": {"pol": 0.9, "pol_shuffled": 0.5, "frozen_v1": 0.8}, "revealed": {"pol": 0.95},
             "omega_auroc": 0.9, "occluder_auroc": 0.85, "ece": 0.05}


def test_polv_all_gates_pass():
    v = decide(POLV_PASS, mode="polv")
    assert (v["G1"], v["G2"], v["G4"], v["ECE"], v["exit_code"]) == (True, True, True, True, 0)


@pytest.mark.parametrize("change,code,failed", [
    ({"unrevealed": {"pol": 0.9, "pol_shuffled": 0.5, "frozen_v1": 0.87}}, 2, "G4"),   # v2 adds < 0.05
    ({"unrevealed": {"pol": 0.9, "pol_shuffled": 0.5, "frozen_v1": None}}, 2, "G4"),
    ({"unrevealed": {"pol": 0.78, "pol_shuffled": 0.5, "frozen_v1": 0.6}}, 2, "G1"),
    ({"unrevealed": {"pol": 0.9, "pol_shuffled": 0.65, "frozen_v1": 0.8}}, 2, "G1"),
    ({"ece": 0.15}, 2, "ECE"),
    ({"ece": None}, 2, "ECE"),
    ({"omega_auroc": 0.84, "occluder_auroc": 0.5}, 3, "G2"),   # below 0.85
    ({"omega_auroc": 0.9, "occluder_auroc": 0.95}, 3, "G2"),   # loses to the occluder probe
    ({"occluder_auroc": None}, 3, "G2"),
])
def test_polv_failure_branches(change, code, failed):
    v = decide({**POLV_PASS, **change}, mode="polv")
    assert v["exit_code"] == code and not v[failed]


def test_polv_g2_ties_the_occluder_probe_and_g1_beats_g2():
    assert decide({**POLV_PASS, "omega_auroc": 0.85, "occluder_auroc": 0.85}, mode="polv")["exit_code"] == 0
    both = decide({**POLV_PASS, "omega_auroc": 0.5, "ece": 0.3}, mode="polv")
    assert both["exit_code"] == 2 and not both["G2"] and not both["ECE"]


def test_polv_scorers_on_synthetic_arrays():
    """P(target) from the heads' posterior, binary ECE, and the probe helpers, on tiny random heads."""
    import torch

    from pol.evidence import EvidenceHeads
    from pol.gates import _fit_probe, binary_ece, mean_v1_tokens, omega_input, polv_p_target

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    heads = EvidenceHeads({"T": 4}, E=2, d_tok=8, d_feat=12, n_time=2, n_tok=3, d_ctx=8).eval()
    items = [{"task": "T", "family": "sealed_bin", "kind": "R+", "target": bool(i % 2), "has_v2": i % 3 > 0,
              "pooled": rng.normal(size=(2, 12)).astype(np.float32), "act": rng.normal(size=(16, 7)).astype(np.float32),
              "v1": rng.normal(size=(2, 3, 8)).astype(np.float32), "act1": rng.normal(size=(16, 7)).astype(np.float32),
              "pos1": rng.normal(size=(16, 7)).astype(np.float32), "v2": rng.normal(size=(2, 3, 8)).astype(np.float32),
              "act2": rng.normal(size=(16, 7)).astype(np.float32), "pos2": rng.normal(size=(16, 7)).astype(np.float32),
              "w2": rng.normal(size=(16,)).astype(np.float32), "pooled2": rng.normal(size=(2, 12)).astype(np.float32),
              "early": rng.normal(size=(2, 2, 12)).astype(np.float32)} for i in range(12)]
    p1, p2 = polv_p_target(heads, items, use_v2=False), polv_p_target(heads, items)
    assert p1.shape == (12,) and ((0 <= p2) & (p2 <= 1)).all()
    no_v2 = [i for i, it in enumerate(items) if not it["has_v2"]]
    assert np.allclose(p1[no_v2], p2[no_v2], atol=1e-6)   # no re-encounter: v2 adds nothing
    assert 0.0 <= binary_ece(p2, [it["target"] for it in items]) <= 1.0
    assert binary_ece(np.array([1.0, 0.0]), [True, False]) == pytest.approx(0.0)
    X, y = mean_v1_tokens(items), [it["target"] for it in items]
    assert X.shape == (12, 8) and _fit_probe(X, y) is not None and _fit_probe(X, [True] * 12) is None
    assert omega_input(heads, items).shape[0] == 12


def test_polv_gates_end_to_end(dataset, tmp_path, monkeypatch):
    from pol.gates import main as gates_main
    from pol.train_stage_a import main

    monkeypatch.setenv("WANDB_MODE", "disabled")
    data, feats = dataset
    main(["--variant", "POLV", "--data", data, "--features", feats, "--tasks", "SealedBin-3", "--out", str(tmp_path / "POLV"),
          "--steps", "4"])
    code = gates_main(["--polv", str(tmp_path / "POLV"), "--data", data, "--features", feats, "--tasks", "SealedBin-3",
                       "--out", str(tmp_path / "gates.json")])
    rep = json.loads((tmp_path / "gates.json").read_text())
    assert code in (0, 2, 3) and set(rep["verdict"]) >= {"G1", "G2", "G4", "ECE", "exit_code"}
    sc = rep["scores"]
    # tiny synthetic val: every val event is a calibration id, so none may be scored
    assert sc["n_val_scored"] == sc["n_val_events"] - sum(sc["n_calibration_excluded"].values())
    assert sc["n_calibration_excluded"]["SealedBin-3"] > 0 and sc["per_task"]["SealedBin-3"]["n"] == 0
    assert "n" in sc["unrevealed"] or "pooled" in sc["unrevealed"]
    assert "omega_auroc" in rep["scores"] and "occluder_auroc" in rep["scores"] and "ece_per_task" in rep["scores"]


@pytest.mark.parametrize("missing", ["frozen_v1", "ece", "occluder_auroc", "omega_auroc", "pol", "pol_shuffled"])
def test_polv_mode_requires_every_key(missing):
    s = {**POLV_PASS, "unrevealed": dict(POLV_PASS["unrevealed"])}
    s["unrevealed"].pop(missing, None)
    s.pop(missing, None)
    v = decide(s, mode="polv")
    assert v["exit_code"] != 0 and f"missing {missing}" in v["notes"]


def test_legacy_mode_skips_are_noted():
    v = decide(PASS)
    assert v["exit_code"] == 0 and v["mode"] == "legacy"
    assert any("frozen_v1" in n for n in v["notes"]) and any("ece" in n for n in v["notes"])
    assert any("occluder_auroc" in n for n in v["notes"])


def _mk(task, i, target=True):
    rng = np.random.default_rng(i)
    return {"task": task, "seed": i, "t": 0, "target": target, "v1": rng.normal(size=(3, 4)).astype("float32")}


def test_calibration_ids_are_excluded_from_val_scoring():
    from pol.evidence import item_id
    from pol.gates import heldout_val

    va = [_mk("a", i) for i in range(10)] + [_mk("b", i) for i in range(10)]
    calib = {item_id(it) for it in va if it["task"] == "a" and it["seed"] < 4}
    held, n_ex = heldout_val(va, calib, ["a", "b"])
    assert n_ex == {"a": 4, "b": 0} and len(held) == 16
    assert not calib & {item_id(it) for it in held}


def test_agm_uses_all_48_labels_train_plus_val():
    from pol.evidence import item_id
    from pol.gates import fit_agm

    tr = [_mk("a", i, bool(i % 2)) for i in range(40)]
    va = [_mk("a", 100 + i, bool(i % 2)) for i in range(20)]
    ids = {item_id(it) for it in tr[:32]} | {item_id(it) for it in va[:16]}
    agm, n = fit_agm(tr + va, ids, ["a"])
    assert n == {"a": 48} and agm["a"] is not None
