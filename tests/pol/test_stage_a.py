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
              "v2": np.full(4, i), "w2": np.full(5, i), "pooled2": np.full(2, i), "has_v2": True}
             for i, (t, k) in enumerate([("A", "R+")] * 3 + [("A", "R0")] * 2 + [("B", "R+")] * 2 + [("B", "R0")])]
    out = shuffle_evidence(items)
    for it, o in zip(items, out):
        src = int(o["v1"][0])
        assert (items[src]["task"], items[src]["kind"]) == (it["task"], it["kind"])
        assert all(int(o[k][0]) == src for k in ("v2", "w2", "pooled2"))  # the views travel together
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
    u = report["scores"]["unrevealed"]
    assert {"n", "failures", "pol", "pol_shuffled", "pol_fake_actions", "markov"} <= set(u)
    assert set(report["scores"]["per_task"]) == {"SealedBin-3"}
