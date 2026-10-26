"""Belief-expert decisions for SealedBin and GatherInject (pure Python, no simulator).

Like the BatteriesSort expert, it acts only on what the cameras could show. Evidence per drop:
  "seen_out"  a cube visibly left the target (sideways bounce, slip, drift): retrieve now
  "cue"       the lid jiggled: ambiguous (backwards bounces usually jiggle it, clean drops sometimes)
  "no_cue"    the lid was visible and still
  "occluded"  a shutter hid the lid: no evidence, the prior of an unseen bounce applies
Opening the lid shows the inside of the bin and resolves every cube at once. The expert opens it
only when the drops so far make "all inside" unlikely, so the decision depends on the history.
"""

from dataclasses import dataclass, field


@dataclass
class BinModel:
    side: float = 0.1  # visible bounce
    behind: float = 0.1  # unseen bounce under the canopy
    cue_hit: float = 0.9  # P(jiggle | unseen bounce)
    cue_false: float = 0.1  # P(jiggle | clean drop)


@dataclass
class BinBelief:
    model: BinModel
    p_out: float = 1.0  # before the first drop the cube is not in the bin
    history: list = field(default_factory=list)

    def observe_release(self, evidence, lid_open=False):
        self.history.append(evidence)
        m = self.model
        if evidence == "seen_out":
            self.p_out = 1.0
            return
        if lid_open:
            self.p_out = 0.0  # the inside is visible: a cube that stayed is seen
            return
        like_out = {"cue": m.cue_hit, "no_cue": 1 - m.cue_hit, "occluded": 1.0}[evidence]
        like_in = {"cue": m.cue_false, "no_cue": 1 - m.cue_false, "occluded": 1.0}[evidence]
        out, clean = m.behind * like_out, (1 - m.side - m.behind) * like_in
        self.p_out = out / (out + clean) if out > 0 else 0.0

    def observe_inside(self, inside):
        self.p_out = 0.0 if inside else 1.0


def decide_after_release(belief):
    return "retrieve" if belief.p_out >= 1.0 else "continue"


def decide_before_confirm(beliefs, threshold=0.9):
    p_all_in = 1.0
    for b in beliefs:
        p_all_in *= 1.0 - b.p_out
    return "confirm" if p_all_in >= threshold else "open_lid"
