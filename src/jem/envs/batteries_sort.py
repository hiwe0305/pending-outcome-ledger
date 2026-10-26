"""BatteriesSort: MIKASA BatteriesCheckerHard extended so test outcomes change the next action.

UNTESTED IN THIS REPOSITORY: needs mani_skill/sapien and the pinned MIKASA install.
Run `python -m jem.envs.check_batteries_sort` on the server before collecting data.

Changes relative to BatteriesCheckerHard (see the proposal, section 4):
  M1  working batteries go back to their tray slot, broken ones into a reject bin;
  M2  a visual-only hood hides whether a battery is seated; an insertion misses with
      probability SPILL_PROB and the battery lands in a visible catch tray (untested);
      a lamp shutter hides the reading of a test with probability OCCLUSION_PROB;
      a faulty tester flips a reading with probability FLIP_PROB;
  M3  a tighter step budget, so re-testing every battery does not fit.
Variants: Hard (lamp lit while seated), Flash (lamp lit for LAMP_AFTERGLOW_STEPS after a
working insert), Compound (Flash + deferred sorting + occlusion 0.2 + flip 0.1).
Working count is drawn per episode (Bernoulli(P_WORKING) per battery) so counting cannot
infer the last battery. Event logs are evaluator-only (info["eem_log"]).
"""

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from mikasa_robo_suite.vla.memory_envs.batteries_checker_hard_vla import BatteriesCheckerVLABaseEnv, _build_by_type

SLOT, HELD, SOCKET, ELSEWHERE = range(4)


class BatteriesSortBaseEnv(BatteriesCheckerVLABaseEnv):
    SPILL_PROB = 0.2
    OCCLUSION_PROB = 0.1
    FLIP_PROB = 0.0
    STICK_PROB = 0.0  # POL held-out failure: the battery stays in the gripper on release
    STICK_STEPS = 15
    HELDOUT = ()
    COVER_CATCH = False  # POL: a visual-only cover hides the catch tray, so a spill is never seen
    COVER_HALF_SIZE = (0.042, 0.042, 0.03)
    P_WORKING = 0.5
    LAMP_MODE = "persistent"  # or "flash"
    DEFERRED = False
    BIN_X_OFFSET_FROM_SOCKET = 0.0
    BIN_Y_OFFSET_FROM_SOCKET = 0.16
    BIN_HALF_SIZE = (0.045, 0.045, 0.012)
    CATCH_Y_OFFSET_FROM_SOCKET = -0.12
    CATCH_HALF_SIZE = (0.035, 0.035, 0.006)
    HOOD_HALF_SIZE = (0.03, 0.03, 0.035)
    LANGUAGE_INSTRUCTION = (
        "Test each battery in the socket and read the lamp. Put working batteries back in their "
        "tray slot and broken batteries in the red bin, pressing the button after each one."
    )

    def __init__(self, *args, split="train", **kwargs):
        from jem.envs.injection import FailureSampler

        self.split = split
        spec = {"spill": self.SPILL_PROB, "occlusion": self.OCCLUSION_PROB, "flip": self.FLIP_PROB, "stick": self.STICK_PROB}
        # Injection draws use their own numpy RNG, reseeded from the episode seed at every reset: the torch
        # RNG is also consumed by the planner/renderer, which made injections irreproducible per seed.
        self.inj_rng = np.random.default_rng(0)
        self.failures = FailureSampler(spec, self.HELDOUT, split, uniform=lambda: self.inj_rng.random())
        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------------- scene
    def _box(self, name, half_size, color, collision=True, initial_pose=None):
        builder = self.scene.create_actor_builder()
        if collision:
            builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=sapien.render.RenderMaterial(base_color=color))
        return _build_by_type(builder, name=name, body_type="kinematic", initial_pose=initial_pose)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        hidden = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])
        self.reject_bin = self._box("reject_bin", self.BIN_HALF_SIZE, np.array([200, 40, 40, 255]) / 255, True, hidden)
        self.catch_tray = self._box("catch_tray", self.CATCH_HALF_SIZE, np.array([90, 90, 90, 255]) / 255, True, hidden)
        # Visual-only occluders: cameras cannot see seating or (sometimes) the lamp, the gripper passes through.
        self.hood = self._box("socket_hood", self.HOOD_HALF_SIZE, np.array([60, 60, 70, 255]) / 255, False, hidden)
        self.shutter = self._box("lamp_shutter", (0.02, 0.02, 0.03), np.array([60, 60, 70, 255]) / 255, False, hidden)
        self.catch_cover = self._box("catch_cover", self.COVER_HALF_SIZE, np.array([60, 60, 70, 255]) / 255, False, hidden)
        n, d = self.num_envs, self.device
        self.stick_left = torch.zeros(n, dtype=torch.int64, device=d)
        self.new_stick_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.sticks = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.int64, device=d)
        self.bin_pos = torch.zeros((n, 3), device=d)
        self.catch_pos = torch.zeros((n, 3), device=d)
        self.reading_flip = torch.zeros(n, dtype=torch.bool, device=d)
        self.lamp_occluded = torch.zeros(n, dtype=torch.bool, device=d)
        self.tests = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.int64, device=d)
        self.spills = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.int64, device=d)
        self.occlusions = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.int64, device=d)
        self.flips = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.int64, device=d)
        self.new_spill_event = torch.zeros(n, dtype=torch.bool, device=d)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.inj_rng = np.random.default_rng([int(self._episode_seed[0]), 7919])
        with torch.device(self.device):
            b = len(env_idx)
            # Per-battery Bernoulli quality instead of a fixed working count.
            active = self.active_mask[env_idx]
            working = (torch.rand((b, self.NUM_BATTERIES)) < self.P_WORKING) & active
            self.working_mask[env_idx] = working
            self.target_working_count[env_idx] = working.sum(1)
            self.oracle_info[env_idx] = 255  # never exposed to the policy; keep it uninformative anyway
            socket = self.socket_slot_pos[env_idx]
            q = torch.tensor(euler2quat(0, 0, np.pi / 2), dtype=torch.float32).repeat(b, 1)
            bin_pos = socket.clone()
            bin_pos[:, 0] += self.BIN_X_OFFSET_FROM_SOCKET
            bin_pos[:, 1] += self.BIN_Y_OFFSET_FROM_SOCKET
            bin_pos[:, 2] = self.BIN_HALF_SIZE[2]
            catch = socket.clone()
            catch[:, 1] += self.CATCH_Y_OFFSET_FROM_SOCKET
            catch[:, 2] = self.CATCH_HALF_SIZE[2]
            hood = socket.clone()
            hood[:, 2] += self.HOOD_HALF_SIZE[2] * 0.6
            self.reject_bin.set_pose(Pose.create_from_pq(p=bin_pos, q=q))
            self.catch_tray.set_pose(Pose.create_from_pq(p=catch, q=q))
            self.hood.set_pose(Pose.create_from_pq(p=hood, q=q))
            cover = catch.clone()
            cover[:, 2] = self.COVER_HALF_SIZE[2] if self.COVER_CATCH else self.HEIGHT_OFFSET
            self.catch_cover.set_pose(Pose.create_from_pq(p=cover, q=q))
            self.stick_left[env_idx] = 0
            self.sticks[env_idx] = 0
            self._set_shutter(env_idx, torch.zeros(b, dtype=torch.bool))
            self.bin_pos[env_idx] = bin_pos
            self.catch_pos[env_idx] = catch
            self.reading_flip[env_idx] = False
            self.lamp_occluded[env_idx] = False
            self.tests[env_idx] = 0
            self.spills[env_idx] = 0
            self.occlusions[env_idx] = 0
            self.flips[env_idx] = 0

    def _set_shutter(self, env_idx, down):
        pose = self.shutter.pose.raw_pose.clone()
        lamp = self.lamp_on_pos[env_idx].clone()
        lamp[:, 2] = torch.where(down, lamp[:, 2] + 0.03, lamp[:, 2] + self.HEIGHT_OFFSET)
        pose[env_idx, :3] = lamp
        self.shutter.pose = pose

    # -------------------------------------------------------------- dynamics
    def _in_box(self, pos, center, half, z_extra=0.03):
        return ((pos[..., :2] - center[:, None, :2]).abs() <= torch.tensor(half[:2], device=pos.device)).all(-1) & (
            pos[..., 2] <= center[:, None, 2] + half[2] + z_extra
        )

    def locations(self, battery_pos, grasp_mask, in_socket):
        """Evaluator-only location label per battery (SLOT/HELD/SOCKET/ELSEWHERE)."""
        home = torch.linalg.norm(battery_pos[..., :2] - self.tray_slot_positions[..., :2], dim=-1) <= self.SLOT_RETURN_XY_TOL * 3
        loc = torch.full(battery_pos.shape[:2], ELSEWHERE, dtype=torch.int64, device=battery_pos.device)
        loc = torch.where(home, SLOT, loc)
        loc = torch.where(in_socket, SOCKET, loc)
        return torch.where(grasp_mask, HELD, loc)

    def evaluate(self):
        n, d = self.num_envs, self.device
        action_mask = torch.ones(n, dtype=torch.bool, device=d)
        self.new_insert_event[:] = False
        self.new_return_event[:] = False
        self.new_confirm_event[:] = False
        self.new_spill_event[:] = False

        battery_pos, battery_raw_pose, grasp_mask = self._stack_battery_poses()
        dist_xy = torch.linalg.norm(battery_pos[:, :, :2] - self.socket_slot_pos[:, None, :2], dim=2)
        dist_z = torch.abs(battery_pos[:, :, 2] - self.socket_slot_pos[:, None, 2])
        in_socket = (dist_xy <= self.SOCKET_INSERT_XY_TOL) & (dist_z <= self.SOCKET_INSERT_Z_TOL) & ~grasp_mask & self.active_mask
        socket_has_battery = in_socket.any(1)
        socket_idx = torch.argmax(in_socket.to(torch.int64), dim=1)
        self.lamp_afterglow_steps = torch.clamp(self.lamp_afterglow_steps - 1, min=0)

        insert = (self.action_stage == self.STAGE_INSERT) & socket_has_battery
        for ei in torch.where(insert)[0].tolist():
            bat = int(socket_idx[ei])
            self.tests[ei, bat] += 1
            raw = self.batteries[bat].pose.raw_pose.clone()
            zero = torch.zeros((n, 3), device=d)
            occluded = self.failures.draw("occlusion")
            self.lamp_occluded[ei] = occluded
            self.occlusions[ei, bat] += int(occluded)
            self._set_shutter(torch.tensor([ei], device=d), torch.tensor([occluded], device=d))
            if self.failures.draw("spill"):
                # Missed insertion: the battery lands in the catch tray, untested; stage stays INSERT.
                raw[ei, :3] = self.catch_pos[ei] + torch.tensor([0, 0, self.CATCH_HALF_SIZE[2] + self.BATTERY_HALF_HEIGHT], device=d)
                self.spills[ei, bat] += 1
                self.new_spill_event[ei] = True
                in_socket[ei, bat] = False
            else:
                raw[ei, :3] = self.socket_slot_pos[ei]
                self.new_insert_event[ei] = True
                self.action_stage[ei] = self.STAGE_RETURN
                self.active_battery_idx[ei] = bat
                self.reading_flip[ei] = self.failures.draw("flip")
                self.flips[ei, bat] += int(self.reading_flip[ei])
                if bool(self.working_mask[ei, bat]) != bool(self.reading_flip[ei]):
                    self.lamp_afterglow_steps[ei] = self.LAMP_AFTERGLOW_STEPS
            raw[ei, 3:7] = self.battery_quat
            self.batteries[bat].pose = raw
            self.batteries[bat].set_linear_velocity(zero)
            self.batteries[bat].set_angular_velocity(zero)
        socket_has_battery = in_socket.any(1)

        # Lamp: shown reading = (seated & working) XOR flip; "flash" keeps only the afterglow.
        seated_working = (in_socket & self.working_mask).any(1)
        shown = torch.where(socket_has_battery, seated_working ^ self.reading_flip, torch.zeros_like(seated_working))
        lamp_on = self.lamp_afterglow_steps > 0
        if self.LAMP_MODE == "persistent":
            lamp_on = lamp_on | shown
        off_pose = self.lamp_bulb_off.pose.raw_pose.clone()
        on_pose = self.lamp_bulb_on.pose.raw_pose.clone()
        off_pose[lamp_on, :3] = self.lamp_off_pos[lamp_on]
        off_pose[~lamp_on, :3] = self.lamp_on_pos[~lamp_on]
        on_pose[lamp_on, :3] = self.lamp_on_pos[lamp_on]
        on_pose[~lamp_on, :3] = self.lamp_off_pos[~lamp_on]
        self.lamp_bulb_off.pose = off_pose
        self.lamp_bulb_on.pose = on_pose

        self._update_button(action_mask)

        # Return stage: a seated-then-extracted battery is released at its slot or in the bin.
        at_slot = torch.linalg.norm(battery_pos[..., :2] - self.tray_slot_positions[..., :2], dim=-1) <= self.SLOT_RETURN_XY_TOL
        in_bin = self._in_box(battery_pos, self.bin_pos, self.BIN_HALF_SIZE, z_extra=2.2 * self.BATTERY_HALF_HEIGHT)  # upright or lying on the bin
        placed = (at_slot if self.DEFERRED else (at_slot | in_bin)) & ~grasp_mask & ~in_socket
        ret = self.action_stage == self.STAGE_RETURN
        self.new_stick_event[:] = False
        for ei in torch.where(ret & (self.active_battery_idx >= 0))[0].tolist():
            bat = int(self.active_battery_idx[ei])
            if int(self.stick_left[ei]) > 0:
                # Held-out failure: the battery follows the gripper, then drops wherever the gripper is.
                raw = self.batteries[bat].pose.raw_pose.clone()
                raw[ei, :3] = self.agent.tcp.pose.p[ei] - torch.tensor([0, 0, 0.02], device=d)
                self.batteries[bat].pose = raw
                self.batteries[bat].set_linear_velocity(torch.zeros((n, 3), device=d))
                self.stick_left[ei] -= 1
                continue
            if bool(placed[ei, bat]):
                if int(self.sticks[ei, bat]) == 0 and self.failures.draw("stick"):
                    self.sticks[ei, bat] += 1
                    self.stick_left[ei] = self.STICK_STEPS
                    self.new_stick_event[ei] = True
                    continue
                self.new_return_event[ei] = True
                self.action_stage[ei] = self.STAGE_CONFIRM

        confirm = (self.action_stage == self.STAGE_CONFIRM) & self.new_button_press_event
        self.new_confirm_event = confirm
        for ei in torch.where(confirm)[0].tolist():
            bat = int(self.active_battery_idx[ei])
            if bat >= 0:
                self.checked_mask[ei, bat] = True
            self.action_stage[ei] = self.STAGE_INSERT
            self.active_battery_idx[ei] = -1
        self.checked_count = (self.checked_mask & self.active_mask).sum(1)

        # Success: every active battery checked and at the destination its true quality requires.
        correct = torch.where(self.working_mask, at_slot, in_bin) & ~grasp_mask
        all_checked = self.checked_count >= self.active_battery_count
        success = all_checked & ((correct | ~self.active_mask).all(1))
        self.found_working_count = (self.checked_mask & self.working_mask).sum(1)

        self.obj_to_goal_pos = self.socket_slot_pos - self.agent.tcp.pose.p
        self.battery_raw_pose = battery_raw_pose
        self.socket_battery_idx = socket_idx
        self.socket_has_working = seated_working
        return {
            "success": success,
            "action_mask": action_mask,
            "cue_mask": torch.zeros(n, dtype=torch.bool, device=d),
            "action_stage": self.action_stage,
            "active_battery_idx": self.active_battery_idx,
            "socket_has_battery": socket_has_battery,
            "socket_battery_idx": socket_idx,
            "socket_has_working": seated_working,
            "checked_count": self.checked_count,
            "active_battery_count": self.active_battery_count,
            "target_working_count": self.target_working_count,
            "found_working_count": self.found_working_count,
            "all_checked": all_checked,
            "new_insert_event": self.new_insert_event,
            "new_return_event": self.new_return_event,
            "new_confirm_event": self.new_confirm_event,
            "new_button_press_event": self.new_button_press_event,
            "button_press_depth": self.button_press_depth,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": None,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
            # Evaluator-only log for labels and strata; jem.data.policy_observation drops it.
            "eem_log": {
                "lamp_on": lamp_on,
                "lamp_occluded": self.lamp_occluded,
                "reading_flip": self.reading_flip,
                "spill_event": self.new_spill_event,
                "stick_event": self.new_stick_event,
                "insert_event": self.new_insert_event,
                "location": self.locations(battery_pos, grasp_mask, in_socket),
                "working": self.working_mask,
                "tests": self.tests,
                "spills": self.spills,
                "occlusions": self.occlusions,
                "flips": self.flips,
            },
        }

    def _update_button(self, action_mask):
        """Button press detection copied from the base evaluate() (unchanged semantics)."""
        tcp = self.agent.tcp.pose.p
        xy = torch.linalg.norm(tcp[:, :2] - self.button_xy, dim=1)
        depth = torch.clamp(self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp[:, 2], min=0.0, max=self.BUTTON_CAP_TRAVEL)
        depth = depth * (xy <= self.BUTTON_PRESS_XY_RADIUS).float()
        self.button_press_depth = depth
        cap = self.button_cap.pose.raw_pose.clone()
        cap[:, 0:2] = self.button_xy
        cap[:, 2] = self.button_cap_unpressed_z - depth
        cap[:, 3:7] = self.button_cap_quat.repeat(cap.shape[0], 1)
        self.button_cap.pose = cap
        pressed = depth >= self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO
        released = depth <= self.BUTTON_CAP_TRAVEL * self.BUTTON_RELEASE_READY_RATIO
        self.press_ready = self.press_ready | (~self.press_ready & released & action_mask)
        self.new_button_press_event = pressed & self.press_ready & action_mask
        self.press_ready = self.press_ready & ~self.new_button_press_event


def _variant(name, n, steps, **attrs):
    # Sized like the MIKASA env BatteriesCheckerHard-{n}; WORKING_BATTERY_COUNT only shapes
    # the unused oracle buffer because quality is redrawn per battery in _initialize_episode.
    cls = type(name, (BatteriesSortBaseEnv,), {"ACTIVE_BATTERY_COUNT": n, "WORKING_BATTERY_COUNT": n, **attrs})
    return register_env(name, max_episode_steps=steps)(cls)


COMPOUND = dict(LAMP_MODE="flash", DEFERRED=True, OCCLUSION_PROB=0.2, FLIP_PROB=0.1,
                LANGUAGE_INSTRUCTION=("Test every battery in the socket and remember each lamp result, return each "
                                      "to its slot and press the button; then move the broken ones to the red bin."))
for _n, _steps in ((3, 1300), (6, 2600)):
    _variant(f"BatteriesSortHard-{_n}-VLA-v0", _n, _steps)
    _variant(f"BatteriesSortFlash-{_n}-VLA-v0", _n, _steps, LAMP_MODE="flash")
    _variant(f"BatteriesSortCompound-{_n}-VLA-v0", _n, int(_steps * 1.3), **COMPOUND)

# POL U-Suite ids (configs/pol/tasks/BatteriesSort-*.yaml): flash lamp, sorting deferred until every
# battery is tested, covered catch tray, held-out sticking release. Step budgets are provisional until
# scripts/pol/calibrate_budget.py writes P95(expert) x 1.15 into the task yaml.
POL_BATTERIES = dict(COMPOUND, OCCLUSION_PROB=0.15, FLIP_PROB=0.05, SPILL_PROB=0.2, STICK_PROB=0.05,
                     HELDOUT=("stick",), COVER_CATCH=True)
for _n, _steps in ((3, 1700), (6, 3400)):
    _variant(f"BatteriesSort-{_n}-VLA-v0", _n, _steps, **POL_BATTERIES)


def episode_strata(env):
    """Evaluator-only stratum of a finished episode (env 0), for per-stratum reporting."""
    u = env.unwrapped
    active = u.active_mask[0]
    if not hasattr(u, "spills"):
        return "released_task"
    if int(u.flips[0][active].sum()):
        return "noisy"
    if int(u.spills[0][active].sum()):
        return "spill"
    if int(u.occlusions[0][active].sum()):
        return "occluded"
    return "clean"


def register_wrappers():
    """Reuse the BatteriesCheckerHard VLA wrapper stack without info overlays."""
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS, VLAWrapperConfig

    base = VLA_WRAPPER_CONFIGS["BatteriesCheckerHard-3-VLA-v0"]
    for n in (3, 6):
        for kind in ("Hard", "Flash", "Compound", ""):
            VLA_WRAPPER_CONFIGS[f"BatteriesSort{kind}-{n}-VLA-v0"] = VLAWrapperConfig(
                overlays=(), curriculum_wrapper=base.curriculum_wrapper, expected_control_mode=base.expected_control_mode)
