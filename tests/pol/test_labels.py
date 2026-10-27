import numpy as np

from pol.events import Event
from pol.labels import OUTCOMES, event_observable, event_outcomes


def bin_logs(T, release_t, cube, final_loc, n=3):
    logs = [{"release_cube": -1, "failure": 0, "cube_location": [0] * n} for _ in range(T)]
    logs[release_t]["release_cube"] = cube
    for t in range(release_t + 1, T):
        logs[t]["cube_location"] = [final_loc if i == cube else 0 for i in range(n)]
    return logs


def test_bin_release_outcomes():
    ev = [Event(10, "R+", np.zeros(3))]
    for loc, name in [(0, "in"), (3, "hidden_out"), (2, "visible_out")]:
        (o,) = event_outcomes("sealed_bin", ev, bin_logs(60, 10, 1, loc))
        assert OUTCOMES["sealed_bin"][o] == name


def test_non_release_events_have_no_outcome():
    ev = [Event(5, "G+", np.zeros(3))]
    assert event_outcomes("sealed_bin", ev, bin_logs(60, 10, 1, 0)) == [None]


def battery_logs(T, t_event, kind, working):
    logs = [{"insert_event": False, "spill_event": False, "working": [working, False, False],
             "location": [0, 0, 0]} for _ in range(T)]
    logs[t_event][f"{kind}_event"] = True
    if kind == "insert":
        for t in range(t_event, T):
            logs[t]["location"] = [2, 0, 0]
    return logs


def test_battery_insert_outcomes():
    ev = [Event(20, "R+", np.zeros(3))]
    assert OUTCOMES["batteries_sort"][event_outcomes("batteries_sort", ev, battery_logs(50, 21, "insert", True))[0]] == "seated_working"
    assert OUTCOMES["batteries_sort"][event_outcomes("batteries_sort", ev, battery_logs(50, 21, "insert", False))[0]] == "seated_broken"
    assert OUTCOMES["batteries_sort"][event_outcomes("batteries_sort", ev, battery_logs(50, 22, "spill", True))[0]] == "spill"
    assert event_outcomes("batteries_sort", ev, battery_logs(50, 45, "insert", True)) == [None]


def test_slip_during_transport_labels_the_following_r0():
    """GatherInject slip (pilot episode 3): the cube leaves the gripper mid-transport (sim release at t=107),
    the gripper closes on nothing and the FSM reports R0 at t=126, not R+."""
    ev = [Event(90, "G+", np.zeros(3)), Event(126, "R0", np.zeros(3)), Event(160, "G+", np.zeros(3))]
    out = event_outcomes("gather_inject", ev, bin_logs(200, 107, 1, 2))
    assert [None if o is None else OUTCOMES["gather_inject"][o] for o in out] == [None, "visible_out", None]
    assert event_observable("gather_inject", ev, bin_logs(200, 107, 1, 2)) == [None, True, None]


def test_empty_close_without_release_has_no_outcome():
    ev = [Event(90, "G+", np.zeros(3)), Event(126, "R0", np.zeros(3))]
    assert event_outcomes("gather_inject", ev, bin_logs(200, 150, 1, 2)) == [None, None]  # release after the R0
    assert event_outcomes("gather_inject", ev, bin_logs(200, 60, 1, 2)) == [None, None]   # release before the grasp
