import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from collect_batteriessort import lamp_reading  # noqa: E402


def rows(occluded_before, occluded_at, lit, t_insert=5, T=20):
    """A test's rows: approach (shutter flag of the previous test), insert at t_insert, then the lamp window.
    lamp_on is the true reading, lamp_shown what the cameras showed."""
    out = []
    for t in range(T):
        occ = occluded_before if t < t_insert else occluded_at
        on = lit and t >= t_insert
        out.append({"log": {"lamp_occluded": occ, "lamp_on": on, "lamp_shown": on and not occ,
                            "insert_event": t == t_insert, "spill_event": False}})
    return out


def test_visible_test_after_an_occluded_one_is_read():
    assert lamp_reading(rows(True, False, lit=True), 0, persistent=False) is True
    assert lamp_reading(rows(True, False, lit=False), 0, persistent=False) is False


def test_occluded_test_after_a_visible_one_is_not_read_through_the_shutter():
    """Round 3: 105 of 110 occluded tests were read from lamp_on because the first rows held the old flag."""
    assert lamp_reading(rows(False, True, lit=True), 0, persistent=False) is None
    assert lamp_reading(rows(False, True, lit=False), 0, persistent=False) is None
