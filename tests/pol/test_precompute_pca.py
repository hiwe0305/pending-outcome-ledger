import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/pol"))
from precompute_features import adopt_pca  # noqa: E402


def test_adopt_pca_copies_pca_and_sibling_proj(tmp_path):
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1/pca.npz").write_bytes(b"P")
    (tmp_path / "r1/proj50.npz").write_bytes(b"J")
    adopt_pca(tmp_path / "r1/pca.npz", tmp_path / "r2")
    assert (tmp_path / "r2/pca.npz").read_bytes() == b"P"
    assert (tmp_path / "r2/proj50.npz").read_bytes() == b"J"


def test_adopt_pca_missing_and_same_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        adopt_pca(tmp_path / "nope.npz", tmp_path / "o")
    (tmp_path / "pca.npz").write_bytes(b"P")
    adopt_pca(tmp_path / "pca.npz", tmp_path)  # same dir: no-op, no error
    assert (tmp_path / "pca.npz").read_bytes() == b"P"
