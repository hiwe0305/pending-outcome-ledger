import numpy as np
import pytest

from jem.envs.injection import FailureSampler

SPEC = {"spill": 0.2, "occlusion": 0.15, "flip": 0.05, "stick": 0.05}


def test_rates_match_spec_within_two_sigma():
    rng = np.random.default_rng(0)
    s = FailureSampler(SPEC, heldout=["stick"], split="eval", uniform=rng.random)
    n = 20000
    for kind, p in SPEC.items():
        hits = sum(s.draw(kind) for _ in range(n))
        assert abs(hits / n - p) <= 2 * np.sqrt(p * (1 - p) / n) + 1e-9


def test_heldout_kind_never_drawn_in_train_split():
    s = FailureSampler(SPEC, heldout=["stick"], split="train", uniform=np.random.default_rng(1).random)
    assert s.p("stick") == 0.0 and not any(s.draw("stick") for _ in range(5000))
    assert s.p("spill") == 0.2


def test_unknown_kind_and_bad_split_are_errors():
    s = FailureSampler(SPEC, heldout=[], split="train", uniform=np.random.random)
    with pytest.raises(KeyError):
        s.draw("teleport")
    with pytest.raises(ValueError):
        FailureSampler(SPEC, heldout=[], split="test", uniform=np.random.random)


def test_fraction_draw_for_sub_kinds():
    rng = np.random.default_rng(2)
    s = FailureSampler({"deflect": 0.2, "deflect_behind_frac": 0.5}, heldout=[], split="train", uniform=rng.random)
    assert s.p("deflect_behind_frac") == 0.5  # fractions are not failure kinds but share the lookup
