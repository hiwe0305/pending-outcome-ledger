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
# Decision target (spec 4, 10.1): (negative, positive) outcomes of the event the decision depends on. Positive =
# the outcome that changes the decision and is hidden at decision time. Visible failures (BatteriesSort spill,
# SealedBin visible_out) stay in the evaluation as negatives, so the gate tests late combination of v1 and v2,
# not memory of a one-frame cue. GatherInject is the revealed control: any failure vs in. Stuck is left out.
DECISION_TARGET = {"batteries_sort": ({0, 2}, {1}), "sealed_bin": ({0, 2}, {1}), "gather_inject": ({0}, {1, 2, 3})}
GATE_TARGET = DECISION_TARGET  # old name
_BIN_LOC = {0: 0, 3: 1, 2: 2, 1: 3}  # bins.AT_TARGET, HIDDEN, ELSEWHERE, HELD -> outcome index
SETTLE = 24  # steps after a release at which the cube's location is read (bounces are applied at +12)
INSERT_WINDOW = (-6, 10)  # an insert/spill event this close to a release belongs to it
RELEASE_LAG = (-6, 3)  # the FSM reports R+ a few steps after the gripper starts opening (hysteresis)


def decision_target(family, outcome):
    """True / False for the positive / negative outcomes, None for events left out."""
    neg, pos = DECISION_TARGET[family]
    return True if outcome in pos else False if outcome in neg else None


gate_target = decision_target  # old name


def _insert_step(e, logs):
    """Step of the insert or spill that belongs to the release event e, or None."""
    lo, hi = e.t + INSERT_WINDOW[0], min(len(logs), e.t + INSERT_WINDOW[1])
    return next((t for t in range(max(0, lo), hi) if logs[t].get("spill_event") or logs[t].get("insert_event")), None)


def _battery(e, logs):
    t = _insert_step(e, logs)
    if t is None:
        return None
    if logs[t].get("spill_event"):
        return 2
    seated = [i for i, l in enumerate(logs[t]["location"]) if l == 2]
    return 0 if seated and logs[t]["working"][seated[0]] else 1


def _bin(e, logs, since=0):
    """R+: the cube released at the gripper opening. R0: a cube that left the gripper while it was still
    loaded (a slip during transport: the gripper then closes on nothing and only reports R0 when it
    reopens), i.e. released after the previous event `since`; its location is read SETTLE steps after
    that release."""
    if e.kind == "R+":
        window = range(max(0, e.t + RELEASE_LAG[0]), min(len(logs), e.t + RELEASE_LAG[1]))
    else:
        window = range(max(0, since), min(len(logs), e.t + RELEASE_LAG[1]))
    released = [(t, logs[t]["release_cube"]) for t in window if logs[t]["release_cube"] >= 0]
    if not released:
        return None
    t_rel, cube = released[0]
    t = min(len(logs) - 1, (e.t if e.kind == "R+" else t_rel) + SETTLE)
    return _BIN_LOC[logs[t]["cube_location"][cube]]


def event_outcomes(family, events, logs):
    """Outcome index per event (into OUTCOMES[family]) or None for events without an outcome.
    BatteriesSort: R+ only. Bins: R+, and R0 after a slip (see _bin)."""
    out = []
    for i, e in enumerate(events):
        if family == "batteries_sort":
            out.append(_battery(e, logs) if e.kind == "R+" else None)
        else:
            out.append(_bin(e, logs, events[i - 1].t + 1 if i else 0) if e.kind in ("R+", "R0") else None)
    return out


def event_observable(family, events, logs, post=16):
    """Evaluator-only: was the outcome region visible after the event (True), hidden by a static
    occluder (False), or not an outcome event (None)? BatteriesSort: the lamp shutter; SealedBin: the
    lid shutter. GatherInject has no occluder, so its outcome events are all visible."""
    out = []
    for e, o in zip(events, event_outcomes(family, events, logs)):
        if o is None:
            out.append(None)
            continue
        window = logs[max(0, e.t + RELEASE_LAG[0]):min(len(logs), e.t + post)]
        if family == "batteries_sort":
            # lamp_occluded is drawn at each insert and kept until the next one: read it at this test's insert step
            out.append(not logs[_insert_step(e, logs)].get("lamp_occluded"))
        elif family == "sealed_bin":
            out.append(not any(r.get("shutter") for r in window))
        else:
            out.append(True)
    return out
