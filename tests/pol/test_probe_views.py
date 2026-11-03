import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/pol"))
from probe_views import probe_aurocs


def views(a, b, rng):
    n = len(a)
    return (np.c_[a, rng.normal(size=(n, 3))] + 0.3 * rng.normal(size=(n, 4)),
            np.c_[b, rng.normal(size=(n, 3))] + 0.3 * rng.normal(size=(n, 4)))


def test_joint_dependence_needs_both_views():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, 800), rng.integers(0, 2, 800)
    y = ((a + b) == 2).astype(int)
    X = views(a, b, rng)
    r = probe_aurocs((X[0][:400], X[1][:400]), (X[0][400:], X[1][400:]), y[:400], y[400:], 200, 5, 0)
    assert r["skipped"] == 0 and r["v12"]["n"] == 5
    assert r["v12"]["mean"] > max(r["v1"]["mean"], r["v2"]["mean"]) + 0.1


def test_single_class_subsample_is_skipped_not_raised():
    rng = np.random.default_rng(0)
    X = views(rng.integers(0, 2, 60), rng.integers(0, 2, 60), rng)
    ytr = np.zeros(30, int)  # train lacks a class
    r = probe_aurocs((X[0][:30], X[1][:30]), (X[0][30:], X[1][30:]), ytr, rng.integers(0, 2, 30), 10, 4, 0)
    assert r["skipped"] == 4 and r["v12"]["mean"] is None
    yv = np.zeros(30, int)  # val lacks a class
    r = probe_aurocs((X[0][:30], X[1][:30]), (X[0][30:], X[1][30:]), rng.integers(0, 2, 30), yv, 10, 4, 0)
    assert r["skipped"] == 4


def test_outcome_macro_three_classes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    X1 = np.eye(3)[y] + 0.5 * rng.normal(size=(600, 3))
    X2 = rng.normal(size=(600, 3))
    r = probe_aurocs((X1[:300], X2[:300]), (X1[300:], X2[300:]), y[:300], y[300:], 100, 3, 0)
    assert r["v1"]["mean"] > 0.8 and r["v2"]["mean"] < 0.65 and r["skipped"] == 0
