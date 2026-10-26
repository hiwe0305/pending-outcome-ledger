import numpy as np

from pol.events import (Event, FsmThresholds, GripperFSM, calibrate_thresholds, detect_events,
                        event_windows, reencounter_index)

TH = FsmThresholds(w_open=0.035, w_empty=0.005)


def trace(*segments):
    return np.concatenate([np.full(n, w, np.float64) for w, n in segments])


def run(widths, x=None):
    x = np.zeros((len(widths), 3)) if x is None else x
    return detect_events(widths, x, TH)


def test_grasp_and_release_of_an_object():
    ev = run(trace((0.04, 10), (0.02, 10), (0.04, 10)))
    assert [e.kind for e in ev] == ["G+", "R+"]
    assert ev[0].t == 12  # third consecutive loaded step
    assert ev[1].t == 21  # second consecutive open step


def test_closing_on_nothing_then_opening():
    ev = run(trace((0.04, 5), (0.0, 6), (0.04, 5)))
    assert [e.kind for e in ev] == ["R0"]


def test_short_loaded_blip_is_not_a_grasp():
    ev = run(trace((0.04, 5), (0.02, 2), (0.04, 5)))
    assert ev == []


def test_noise_inside_each_state_adds_no_events():
    rng = np.random.default_rng(0)
    w = trace((0.04, 30), (0.02, 30), (0.04, 30), (0.0, 30), (0.04, 30))
    w = w + rng.uniform(-0.002, 0.002, len(w))
    assert [e.kind for e in run(w)] == ["G+", "R+", "R0"]


def test_single_open_spike_while_loaded_is_ignored():
    ev = run(trace((0.04, 5), (0.02, 8), (0.04, 1), (0.02, 8), (0.04, 5)))
    assert [e.kind for e in ev] == ["G+", "R+"]


def test_fsm_is_incremental_and_resettable():
    fsm = GripperFSM(TH)
    w = trace((0.04, 5), (0.02, 5))
    out = [fsm.step(t, float(v), np.zeros(3)) for t, v in enumerate(w)]
    assert [e.kind for e in out if e] == ["G+"]
    fsm.reset()
    assert fsm.step(0, 0.02, np.zeros(3)) is None


def test_event_carries_end_effector_position():
    w = trace((0.04, 5), (0.02, 5))
    x = np.arange(len(w) * 3, dtype=float).reshape(-1, 3)
    (e,) = run(w, x)
    assert np.array_equal(e.x_ee, x[e.t])


def test_windows_are_clipped():
    pre, post = event_windows(3, 10, pre=8, post=16)
    assert (pre.start, pre.stop, post.start, post.stop) == (0, 3, 3, 10)


def test_reencounter_within_radius_only():
    z = np.zeros(3)
    events = [Event(0, "R+", z), Event(5, "G+", np.array([0.2, 0, 0])), Event(9, "G+", np.array([0.05, 0, 0]))]
    assert reencounter_index(events, 0, rho=0.06) == 2
    assert reencounter_index(events, 1, rho=0.06) is None
    assert reencounter_index(events, 2, rho=0.06) is None


def test_calibration_recovers_band_between_modes():
    rng = np.random.default_rng(1)
    w = np.concatenate([rng.normal(0.04, 0.001, 500), rng.normal(0.02, 0.001, 300), rng.normal(0.0, 0.0005, 200)])
    th = calibrate_thresholds([w])
    assert 0.006 < th.w_empty < 0.014 and 0.026 < th.w_open < 0.034
