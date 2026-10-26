import pytest

from jem.envs.expert_bins import BinBelief, BinModel, decide_after_release, decide_before_confirm

M = BinModel(side=0.1, behind=0.1, cue_hit=0.9, cue_false=0.1)


def belief(*evidence, lid_open=False):
    b = BinBelief(M)
    for e in evidence:
        b.observe_release(e, lid_open=lid_open)
    return b


def test_visible_failure_is_retrieved_immediately():
    b = belief("seen_out")
    assert b.p_out == 1.0 and decide_after_release(b) == "retrieve"


@pytest.mark.parametrize("evidence,expected", [
    ("cue", 0.1 * 0.9 / (0.1 * 0.9 + 0.8 * 0.1)),
    ("no_cue", 0.1 * 0.1 / (0.1 * 0.1 + 0.8 * 0.9)),
    ("occluded", 0.1 / (0.1 + 0.8)),
])
def test_posterior_per_evidence(evidence, expected):
    b = belief(evidence)
    assert b.p_out == pytest.approx(expected) and decide_after_release(b) == "continue"


def test_confirm_depends_on_history():
    assert decide_before_confirm([belief("no_cue") for _ in range(5)]) == "confirm"
    assert decide_before_confirm([belief("no_cue") for _ in range(4)] + [belief("cue")]) == "open_lid"
    assert decide_before_confirm([belief("no_cue") for _ in range(4)] + [belief("occluded")]) == "open_lid"


def test_lid_view_resolves_every_cube():
    b = belief("cue")
    b.observe_inside(False)
    assert b.p_out == 1.0
    b.observe_release("no_cue", lid_open=True)  # with the lid open a re-drop is seen directly
    assert b.p_out == 0.0


def test_revealed_task_never_verifies():
    model = BinModel(side=0.15, behind=0.0, cue_hit=0.0, cue_false=0.0)
    beliefs = [BinBelief(model) for _ in range(5)]
    for b in beliefs:
        b.observe_release("no_cue")
    assert decide_before_confirm(beliefs) == "confirm"
