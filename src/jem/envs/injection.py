"""Failure injection shared by the U-Suite envs (pure logic, no simulator).

Rates come from configs/pol/tasks/*.yaml. Held-out failure kinds are switched off in the train
split, so demonstrations never contain them and evaluation can test generalization to them.
`uniform` is the env's seeded U(0,1) source (torch.rand under ManiSkill's per-episode seeding).
"""

SPLITS = ("train", "eval")


class FailureSampler:
    def __init__(self, spec, heldout, split, uniform):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.spec, self.heldout, self.split, self.uniform = dict(spec), set(heldout), split, uniform

    def p(self, kind):
        if kind not in self.spec:
            raise KeyError(f"unknown failure kind {kind!r}")
        return 0.0 if (self.split == "train" and kind in self.heldout) else float(self.spec[kind])

    def draw(self, kind):
        return float(self.uniform()) < self.p(kind)
