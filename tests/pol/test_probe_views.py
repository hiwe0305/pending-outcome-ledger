import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/pol"))
from probe_views import probe_aurocs


def test_joint_dependence_needs_both_views():
    rng = np.random.default_rng(0)
    n = 600
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = (a + b == 2).astype(int)
    X1 = np.c_[a, rng.normal(size=(n, 3))] + 0.3 * rng.normal(size=(n, 4))
    X2 = np.c_[b, rng.normal(size=(n, 3))] + 0.3 * rng.normal(size=(n, 4))
    r = probe_aurocs(X1, X2, y, n, 0)
    assert r["v12"] > max(r["v1"], r["v2"]) + 0.1
