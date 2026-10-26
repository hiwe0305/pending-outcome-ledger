"""SealedBin and GatherInject: MIKASA GatherAndRecall scenes in which placements can fail.

Both keep the GatherAndRecall flow (move every cube to the target, a lamp flashes once, then press
the button of the flash colour) and its evaluator, which registers a press only while every cube is
on the target. They differ only in whether a failed placement can be seen:

  GatherInject (revealed control): the open disc. Cubes slip during transport or drift off the disc
      on release, in plain view.
  SealedBin (unrevealed): the disc is the floor of a bin with walls and an opaque visual-only lid.
      A cube can bounce out after the gripper has left the bin (so the wrist camera no longer looks
      inside): sideways onto the table (visible) or backwards under a visual-only canopy (seen by no
      camera). A backwards bounce usually jiggles the lid, but clean drops sometimes do too, so the
      jiggle is ambiguous evidence; on some drops a shutter hides the lid, so there is no evidence at
      all. Whether to check the bin before confirming therefore depends on the history of drops.
      Pressing the red handle on the bin slides the lid away so the top camera sees the inside.

Failure rates come from configs/pol/tasks/*.yaml; held-out kinds are off in the train split.
Evaluator-only labels are returned in info["pol_log"] and never reach the VLA observation.
"""

import numpy as np
import sapien
import torch
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from mikasa_robo_suite.vla.memory_envs.gather_and_recall_vla import GatherAndRecallVLABaseEnv

from .injection import FailureSampler

AT_TARGET, HELD, ELSEWHERE, HIDDEN = range(4)  # evaluator-only cube location labels
FAILURE_CODES = {"": 0, "deflect_side": 1, "deflect_behind": 2, "release_occluded": 4,
                 "slip": 5, "place_drift": 6, "stick": 7}
VISIBLE_OUT = (1, 5, 6)  # failures a camera sees as they happen
GRAY = np.array([150, 150, 155, 255], dtype=np.float32) / 255.0


class _PlacementTaskBase(GatherAndRecallVLABaseEnv):
    INJECTION = {}
    HELDOUT = ()
    STICK_KIND = "stick"
    STICK_STEPS = 15
    RELEASE_RADIUS = 0.10  # a release within this xy distance of the target counts as a placement attempt

    def __init__(self, *args, split="train", **kwargs):
        self.split = split
        # Own numpy RNG, reseeded from the episode seed at reset (the torch RNG is shared with the planner).
        self.inj_rng = np.random.default_rng(0)
        self.failures = FailureSampler(self.INJECTION, self.HELDOUT, split, uniform=lambda: self.inj_rng.random())
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ scene
    def _box(self, name, half, color, collision, pose=None):
        b = self.scene.create_actor_builder()
        if collision:
            b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=color))
        return _build_by_type(b, name=name, body_type="kinematic", initial_pose=pose or sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]))

    def _load_scene(self, options):
        super()._load_scene(options)
        n, N, d = self.num_envs, self.N_CUBES, self.device
        self.held_prev = torch.zeros((n, N), dtype=torch.bool, device=d)
        self.stick_left = torch.zeros(n, dtype=torch.int64, device=d)
        self.stick_cube = torch.full((n,), -1, dtype=torch.int64, device=d)
        self.pending = []  # (env, cube, step to apply, kind)
        self.failure_now = torch.zeros(n, dtype=torch.int64, device=d)
        self.release_now = torch.full((n,), -1, dtype=torch.int64, device=d)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        self.inj_rng = np.random.default_rng([int(self._episode_seed[0]), 7919])
        self.held_prev[env_idx] = False
        self.stick_left[env_idx] = 0
        self.stick_cube[env_idx] = -1
        self.pending = [p for p in self.pending if p[0] not in env_idx.tolist()]

    # -------------------------------------------------------------- dynamics
    def _place_cube(self, ei, i, xyz, vel=None):
        raw = self.cubes[i].pose.raw_pose.clone()
        raw[ei, :3] = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        self.cubes[i].pose = raw
        v = self.cubes[i].linear_velocity.clone()
        v[ei] = 0 if vel is None else torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        self.cubes[i].set_linear_velocity(v)

    def _on_grasp(self, ei, i, t):
        pass

    def _on_release(self, ei, i, t):
        """Return the failure kind applied to this release, or ''."""
        return ""

    def _apply_pending(self, ei, i, kind):
        pass

    def _cube_location(self, ei, i, held):
        raise NotImplementedError

    def evaluate(self):
        n, N = self.num_envs, self.N_CUBES
        t = self.elapsed_steps.to(torch.int64).reshape(-1)
        self.failure_now[:] = 0
        self.release_now[:] = -1
        held = torch.stack([self.agent.is_grasping(c) for c in self.cubes], 1)
        for ei in range(n):
            if int(self.stick_left[ei]) > 0:  # the cube follows the gripper, then drops where it is
                i = int(self.stick_cube[ei])
                self._place_cube(ei, i, self.agent.tcp.pose.p[ei] - torch.tensor([0, 0, 0.03], device=self.device))
                self.stick_left[ei] -= 1
                held[ei, i] = True
                continue
            for i in range(N):
                if held[ei, i] and not self.held_prev[ei, i]:
                    self._on_grasp(ei, i, int(t[ei]))
                if self.held_prev[ei, i] and not held[ei, i]:
                    self.release_now[ei] = i
                    kind = self._on_release(ei, i, int(t[ei]))
                    if kind == self.STICK_KIND or kind == "release_occluded":
                        self.stick_left[ei], self.stick_cube[ei] = self.STICK_STEPS, i
                    self.failure_now[ei] = FAILURE_CODES[kind]
        due = [p for p in self.pending if int(t[p[0]]) >= p[2]]
        for p in due:
            self.pending.remove(p)
            self._apply_pending(p[0], p[1], p[3])
            if p[3] in FAILURE_CODES:
                self.failure_now[p[0]] = FAILURE_CODES[p[3]]
        self.held_prev = held
        info = super().evaluate()
        loc = torch.tensor([[self._cube_location(ei, i, bool(held[ei, i])) for i in range(N)] for ei in range(n)],
                           device=self.device)
        info["pol_log"] = {"release_cube": self.release_now.clone(), "failure": self.failure_now.clone(),
                           "cube_location": loc, "flash_active": info["flash_active"],
                           "pressed_button": info["pressed_button"], **self._extra_log()}
        return info

    def _extra_log(self):
        return {}

    def _near_target(self, ei, i):
        p = self.cubes[i].pose.p[ei]
        return float(torch.linalg.norm(p[:2] - self.disc_xy[ei])) < self.RELEASE_RADIUS


class GatherInjectBaseEnv(_PlacementTaskBase):
    """Revealed control: the open disc of GatherAndRecall with in-view failures."""

    INJECTION = {"slip": 0.15, "place_drift": 0.1, "stick": 0.05}
    HELDOUT = ("stick",)
    SLIP_DELAY = (10, 30)
    DRIFT_SPEED = 0.8

    def _on_grasp(self, ei, i, t):
        if self.failures.draw("slip"):
            lo, hi = self.SLIP_DELAY
            self.pending.append((ei, i, t + int(self.inj_rng.integers(lo, hi)), "slip"))

    def _apply_pending(self, ei, i, kind):
        if kind == "slip" and bool(self.agent.is_grasping(self.cubes[i])[ei]):
            self._place_cube(ei, i, self.cubes[i].pose.p[ei] - torch.tensor([0, 0, 0.045], device=self.device))

    def _on_release(self, ei, i, t):
        self.pending = [p for p in self.pending if not (p[0] == ei and p[1] == i and p[3] == "slip")]
        if not self._near_target(ei, i):
            return ""
        if self.failures.draw("stick"):
            return "stick"
        if self.failures.draw("place_drift"):
            a = float(self.inj_rng.random()) * 2 * np.pi
            self._place_cube(ei, i, self.cubes[i].pose.p[ei],
                             vel=[self.DRIFT_SPEED * np.cos(a), self.DRIFT_SPEED * np.sin(a), 0.0])
            return "place_drift"
        return ""

    def _cube_location(self, ei, i, held):
        if held:
            return HELD
        p = self.cubes[i].pose.p[ei]
        return AT_TARGET if float(torch.linalg.norm(p[:2] - self.disc_xy[ei])) < self.DISC_ON_THRESH else ELSEWHERE


class SealedBinBaseEnv(_PlacementTaskBase):
    """Unrevealed: cubes go into an opaque lidded bin; a bounce-out can land where no camera sees."""

    INJECTION = {"deflect_side": 0.1, "deflect_behind": 0.1, "cue_hit": 0.9, "cue_false": 0.1,
                 "cue_occluded": 0.2, "release_occluded": 0.05}
    HELDOUT = ("release_occluded",)
    STICK_KIND = "release_occluded"
    CUBE_COLORS = [GRAY]
    DISC_RADIUS = 0.09
    DISC_ON_THRESH = 0.085
    WALL_HALF = 0.09
    WALL_THICK = 0.004
    WALL_HEIGHT = 0.05
    LID_Z = 0.058
    BOUNCE_DELAY = 12  # steps after release: the gripper is above the lid by then
    SIDE_OFFSET = (0.0, 0.16)  # where a sideways bounce lands (visible), relative to the bin centre
    BEHIND_OFFSET = (0.0, -0.17)  # under the canopy, between the bin and the cube cluster (seen by no camera)
    CANOPY_HALF = (0.06, 0.07, 0.045)
    HANDLE_OFFSET = (-0.12, 0.07)
    HANDLE_RADIUS = 0.03
    LANGUAGE_INSTRUCTION = (
        "Put all cubes into the covered bin. A lamp will briefly flash while you work. "
        "When every cube is inside, press the button matching the flash color."
    )

    def _load_scene(self, options):
        super()._load_scene(options)
        dark, red = np.array([70, 70, 80, 255]) / 255, np.array([200, 40, 40, 255]) / 255
        h, th, hh = self.WALL_HALF, self.WALL_THICK, self.WALL_HEIGHT / 2
        self.walls = [self._box(f"bin_wall_{k}", half, dark, True) for k, half in
                      enumerate([(th, h, hh), (th, h, hh), (h, th, hh), (h, th, hh)])]
        self.lid = self._box("bin_lid", (h + th, h + th, 0.004), dark, False)
        self.canopy = self._box("bin_canopy", self.CANOPY_HALF, dark, False)
        self.handle = self._box("bin_handle", (0.015, 0.015, 0.02), red, True)
        self.shutter = self._box("bin_shutter", (self.WALL_HALF + 0.02, self.WALL_HALF + 0.02, 0.006), np.array([40, 40, 48, 255]) / 255, False)
        n, d = self.num_envs, self.device
        self.lid_open = torch.zeros(n, dtype=torch.bool, device=d)
        self.jiggle_left = torch.zeros(n, dtype=torch.int64, device=d)
        self.handle_xy = torch.zeros((n, 2), device=d)
        self.shutter_left = torch.zeros(n, dtype=torch.int64, device=d)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            q = torch.tensor([1.0, 0, 0, 0]).repeat(b, 1)
            c = torch.zeros((b, 3))
            c[:, :2] = self.disc_xy[env_idx]
            h, hh = self.WALL_HALF, self.WALL_HEIGHT / 2
            for wall, (dx, dy) in zip(self.walls, [(h, 0), (-h, 0), (0, h), (0, -h)]):
                p = c.clone()
                p[:, 0] += dx
                p[:, 1] += dy
                p[:, 2] = hh
                wall.set_pose(Pose.create_from_pq(p=p, q=q))
            self.lid_open[env_idx] = False
            self.jiggle_left[env_idx] = 0
            self.shutter_left[env_idx] = 0
            self._set_lid(env_idx)
            cp = c.clone()
            cp[:, 0] += self.BEHIND_OFFSET[0]
            cp[:, 1] += self.BEHIND_OFFSET[1]
            cp[:, 2] = self.CANOPY_HALF[2]
            self.canopy.set_pose(Pose.create_from_pq(p=cp, q=q))
            hp = c.clone()
            hp[:, 0] += self.HANDLE_OFFSET[0]
            hp[:, 1] += self.HANDLE_OFFSET[1]
            hp[:, 2] = 0.02
            self.handle.set_pose(Pose.create_from_pq(p=hp, q=q))
            self.handle_xy[env_idx] = hp[:, :2]

    def _set_lid(self, env_idx):
        raw = self.lid.pose.raw_pose.clone()
        raw[env_idx, :2] = self.disc_xy[env_idx]
        raw[env_idx, 2] = torch.where(self.lid_open[env_idx], torch.full_like(raw[env_idx, 2], self.HEIGHT_OFFSET),
                                      torch.full_like(raw[env_idx, 2], self.LID_Z))
        jig = self.jiggle_left[env_idx] > 0
        raw[env_idx, 0] += jig.float() * 0.004
        raw[env_idx, 3:7] = torch.tensor([1.0, 0, 0, 0], device=self.device)
        self.lid.pose = raw
        sh = self.shutter.pose.raw_pose.clone()
        sh[env_idx, :2] = self.disc_xy[env_idx]
        sh[env_idx, 2] = torch.where(self.shutter_left[env_idx] > 0, torch.full_like(sh[env_idx, 2], self.LID_Z + 0.03),
                                     torch.full_like(sh[env_idx, 2], self.HEIGHT_OFFSET))
        sh[env_idx, 3:7] = torch.tensor([1.0, 0, 0, 0], device=self.device)
        self.shutter.pose = sh

    def _on_release(self, ei, i, t):
        if not self._near_target(ei, i):
            return ""
        if self.failures.draw("release_occluded"):
            return "release_occluded"
        u = float(self.inj_rng.random())
        side, behind = self.failures.p("deflect_side"), self.failures.p("deflect_behind")
        kind = "deflect_side" if u < side else ("deflect_behind" if u < side + behind else "")
        if kind:
            self.pending.append((ei, i, t + self.BOUNCE_DELAY, kind))
        if kind != "deflect_side":  # the lid cue of an unseen outcome
            self.shutter_left[ei] = self.BOUNCE_DELAY + 6 if self.failures.draw("cue_occluded") else 0
            if not self.shutter_left[ei] and self.failures.draw("cue_hit" if kind == "deflect_behind" else "cue_false"):
                self.pending.append((ei, i, t + self.BOUNCE_DELAY, "jiggle"))
        return ""

    def _apply_pending(self, ei, i, kind):
        c = self.disc_xy[ei]
        z = self.CUBE_HALF_SIZE + 0.002
        if kind == "jiggle":
            self.jiggle_left[ei] = 4  # seen, but does not tell whether the cube left
            return
        off = self.SIDE_OFFSET if kind == "deflect_side" else self.BEHIND_OFFSET
        jitter = (self.inj_rng.random(2) - 0.5) * 0.03
        self._place_cube(ei, i, [float(c[0]) + off[0] + float(jitter[0]), float(c[1]) + off[1] + float(jitter[1]), z])

    def evaluate(self):
        tcp = self.agent.tcp.pose.p
        near_handle = (torch.linalg.norm(tcp[:, :2] - self.handle_xy, dim=1) < self.HANDLE_RADIUS) & (tcp[:, 2] < 0.08)
        self.lid_open = self.lid_open | near_handle
        self.jiggle_left = torch.clamp(self.jiggle_left - 1, min=0)
        self.shutter_left = torch.clamp(self.shutter_left - 1, min=0)
        self._set_lid(torch.arange(self.num_envs, device=self.device))
        return super().evaluate()

    def _cube_location(self, ei, i, held):
        if held:
            return HELD
        p = self.cubes[i].pose.p[ei]
        c = self.disc_xy[ei]
        if bool((torch.abs(p[:2] - c) < self.WALL_HALF).all()) and float(p[2]) < self.WALL_HEIGHT:
            return AT_TARGET
        b = c + torch.tensor(self.BEHIND_OFFSET[:2], device=self.device)
        if bool((torch.abs(p[:2] - b) < torch.tensor(self.CANOPY_HALF[:2], device=self.device)).all()):
            return HIDDEN
        return ELSEWHERE

    def _extra_log(self):
        return {"lid_open": self.lid_open.clone(), "lid_jiggle": self.jiggle_left > 0, "shutter": self.shutter_left > 0}


def _register(name, base, n_cubes, steps):
    cls = type(name.replace("-", "_"), (base,), {"N_CUBES": n_cubes})
    return register_env(name, max_episode_steps=steps)(cls)


# Provisional budgets: scripts/pol/calibrate_budget.py replaces them with P95(expert) x 1.15.
for _n, _steps in ((3, 900), (5, 1400)):
    _register(f"SealedBin-{_n}-VLA-v0", SealedBinBaseEnv, _n, _steps)
    _register(f"GatherInject-{_n}-VLA-v0", GatherInjectBaseEnv, _n, _steps)


def register_wrappers():
    """Reuse the GatherAndRecall VLA wrapper stack without info overlays."""
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS, VLAWrapperConfig

    for n in (3, 5):
        base = VLA_WRAPPER_CONFIGS[f"GatherAndRecall{n}-VLA-v0"]
        for fam in ("SealedBin", "GatherInject"):
            VLA_WRAPPER_CONFIGS[f"{fam}-{n}-VLA-v0"] = VLAWrapperConfig(
                overlays=(), curriculum_wrapper=base.curriculum_wrapper, expected_control_mode=base.expected_control_mode)
