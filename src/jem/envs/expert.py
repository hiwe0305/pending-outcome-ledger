"""Belief-expert decisions for BatteriesSort (pure Python, no simulator needed).

The collector knows the simulator state, but the expert must act only on what the
policy could have observed; otherwise demonstrations teach the policy to guess.
For each battery the expert keeps the exact posterior P(working | observed tests),
where one test yields (held after reaching in, lamp reading or None if occluded),
then routes when confident or re-tests when the value of information is positive.
"""

from dataclasses import dataclass, field

SLOT, HELD, SOCKET, ELSEWHERE = range(4)  # location labels, matching jem.memory
UNTESTED, WORKING, BROKEN = range(3)  # evidence labels


@dataclass
class ObservationModel:
    spill: float = 0.2  # insertion misses the socket and the battery lands in the catch tray
    flip: float = 0.0  # faulty-tester reading flip
    occlusion: float = 0.1  # lamp shutter lowered for this test (reading unavailable)
    p_working: float = 0.5

    def likelihood(self, held, reading, working):
        """P(held, reading | working). reading is True/False, or None when the lamp was occluded."""
        p_held = (1 - self.spill) if held else self.spill
        if reading is None:
            return p_held * self.occlusion  # occlusion is independent of quality
        lit = held and working  # a lamp lights only for a seated working battery
        p_reading = (1 - self.flip) if reading == lit else self.flip
        return p_held * (1 - self.occlusion) * p_reading


@dataclass
class BatteryBelief:
    model: ObservationModel
    tests: list = field(default_factory=list)  # [(held, reading)]

    def observe(self, held, reading):
        self.tests.append((bool(held), reading))

    @property
    def p_working(self):
        a, b = self.model.p_working, 1 - self.model.p_working
        for held, reading in self.tests:
            a *= self.model.likelihood(held, reading, True)
            b *= self.model.likelihood(held, reading, False)
        return a / (a + b)

    @property
    def evidence(self):
        """Evidence label: latest observed reading of a seated test (not hidden quality)."""
        for held, reading in reversed(self.tests):
            if held and reading is not None:
                return WORKING if reading else BROKEN
        return UNTESTED

    def value_of_test(self):
        """Expected gain in P(correct routing) from one more test."""
        p = self.p_working
        now = max(p, 1 - p)
        after = sum(
            max(p * self.model.likelihood(h, r, True), (1 - p) * self.model.likelihood(h, r, False))
            for h in (True, False)
            for r in (True, False, None)
        )
        return after - now


def decide(belief, retests_left, threshold=0.99, cost=0.05):
    """'home' | 'bin' | 'retest' for one battery after a test."""
    p = belief.p_working
    if max(p, 1 - p) >= threshold or retests_left <= 0 or belief.value_of_test() <= cost:
        return "home" if p >= 0.5 else "bin"
    return "retest"


def choose_verify(beliefs, retests_left, cost=0.05):
    """Deferred sorting: index of the battery most worth re-testing, or None."""
    if retests_left <= 0:
        return None
    gains = [(b.value_of_test(), i) for i, b in enumerate(beliefs)]
    best, i = max(gains)
    return i if best > cost else None
