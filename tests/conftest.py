"""Constrain fixture parallelism and verify downloaded helper provenance."""

import hashlib, json
from pathlib import Path
import pytest, torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root / "configs/source-hashes.json").read_text())["files"]
    for relative in [
        "third_party/mikasa/mikasa_robo_suite/vla/benchmarking.py",
        "third_party/oft/prismatic/extern/hf/modeling_prismatic.py",
        "third_party/oft/prismatic/training/train_utils.py",
    ]:
        path = root / relative
        if (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[relative]
        ):
            raise pytest.UsageError(f"Pinned source changed: {relative}")
