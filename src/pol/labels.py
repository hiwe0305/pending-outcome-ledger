"""Evaluator-only outcome labels of gripper events, read from the simulator logs.

Used for metrics (silent-failure AUROC, NMI of learned outcomes), the revealed/unrevealed test, the
shortcut probes, Oracle tokens and the label-trained heads of Markov / AGM-style. Never an input
of POL or of the self-supervised baselines.
"""

OUTCOMES = {
    "batteries_sort": ("seated_working", "seated_broken", "spill"),
    "sealed_bin": ("in", "hidden_out", "visible_out", "stuck"),
    "gather_inject": ("in", "hidden_out", "visible_out", "stuck"),
}
FAILED = {"batteries_sort": {2}, "sealed_bin": {1, 2, 3}, "gather_inject": {1, 2, 3}}
_BIN_LOC = {0: 0, 3: 1, 2: 2, 1: 3}  # bins.AT_TARGET, HIDDEN, ELSEWHERE, HELD -> outcome index
SETTLE = 24  # steps after a release at which the cube's location is read (bounces are applied at +12)
INSERT_WINDOW = (-6, 10)  # an insert/spill event this close to a release belongs to it
RELEASE_LAG = (-6, 3)  # the FSM reports R+ a few steps after the gripper starts opening (hysteresis)


def _battery(e, logs):
    lo, hi = e.t + INSERT_WINDOW[0], min(len(logs), e.t + INSERT_WINDOW[1])
    for t in range(max(0, lo), hi):
        if logs[t].get("spill_event"):
            return 2
        if logs[t].get("insert_event"):
            seated = [i for i, l in enumerate(logs[t]["location"]) if l == 2]
            return 0 if seated and logs[t]["working"][seated[0]] else 1
    return None


def _bin(e, logs):
    window = range(max(0, e.t + RELEASE_LAG[0]), min(len(logs), e.t + RELEASE_LAG[1]))
    cubes = [logs[t]["release_cube"] for t in window if logs[t]["release_cube"] >= 0]
    if not cubes:
        return None
    t = min(len(logs) - 1, e.t + SETTLE)
    return _BIN_LOC[logs[t]["cube_location"][cubes[0]]]


def event_outcomes(family, events, logs):
    """Outcome index per event (into OUTCOMES[family]) or None for events without an outcome."""
    fn = _battery if family == "batteries_sort" else _bin
    return [fn(e, logs) if e.kind == "R+" else None for e in events]
