"""Motion primitives for SealedBin / GatherInject demonstrations (sim venv only).

Follows the pattern of MIKASA's GatherAndRecall oracle planner (MIT): ManiSkill's
PandaArmMotionPlanningSolver with screw motions, a fast path follower that subsamples waypoints
(stride 2) and stops at the step budget, top-down grasps, and pd_joint_pos hold actions.
Motion uses the true object poses (as MIKASA's planners do); *decisions* come from
jem.envs.expert_bins, which only uses observable evidence.
"""

import types

import numpy as np
import sapien
import torch

WAYPOINT_STRIDE = 2
GRIPPER_CLOSE_STEPS, GRIPPER_OPEN_STEPS, SETTLE_STEPS = 5, 5, 3


def _hold_action(env_u, gripper_state):
    qpos = env_u.agent.robot.get_qpos()[..., :-2].detach().cpu().numpy()
    grip = np.full((qpos.shape[0], 1), float(np.clip(gripper_state, -1.0, 1.0)), np.float32)
    return np.concatenate([qpos, grip], 1).astype(np.float32)[0]


def make_bin_primitives(env, max_env_steps):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    env_u = env.unwrapped
    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False, base_pose=env_u.agent.robot.pose,
                                           visualize_target_grasp_pose=False, print_env_info=False,
                                           joint_vel_limits=2.0, joint_acc_limits=2.0)
    done = {"value": False}

    def elapsed():
        return int(env_u.elapsed_steps.reshape(-1)[0].item())

    def follow(self, result, refine_steps=0):  # noqa: ARG001
        if done["value"] or result["position"].shape[0] <= 0:
            return None, None, done["value"], False, None
        n = result["position"].shape[0]
        idxs = list(range(0, n, WAYPOINT_STRIDE))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        out = (None,) * 5
        for i in idxs:
            if elapsed() >= max_env_steps - 1:
                done["value"] = True
                break
            out = self.env.step(np.hstack([result["position"][i], self.gripper_state]))
            self.elapsed_steps += 1
        return out

    planner.follow_path = types.MethodType(follow, planner)
    tcp_q = env_u.agent.tcp.pose.raw_pose[0, 3:].detach().cpu().numpy()
    button_top_z = float(env_u.button_top_z[0].item())
    z_lift = max(0.24, button_top_z + 0.12)
    cube_hs = float(env_u.CUBE_HALF_SIZE)
    z_place = float(env_u.DISC_HALF_HEIGHT) * 2 + cube_hs + 0.003

    def move(x, y, z, retries=3):
        for _ in range(retries):
            if done["value"]:
                return False
            res = planner.move_to_pose_with_screw(sapien.Pose(p=[x, y, z], q=tcp_q), dry_run=False, refine_steps=0)
            if res is not None and res != -1 and res[0] is not None:
                return True
            cur = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
            planner.move_to_pose_with_screw(sapien.Pose(p=[float(cur[0]), float(cur[1]), z_lift], q=tcp_q),
                                            dry_run=False, refine_steps=0)
        return False

    def hold(k):
        for _ in range(k):
            if done["value"] or elapsed() >= max_env_steps - 1:
                done["value"] = True
                return
            env.step(_hold_action(env_u, planner.gripper_state))

    def grasping_any():
        return any(bool(env_u.agent.is_grasping(c)[0].item()) for c in env_u.cubes)

    def pick_cube(i):
        p = env_u.cubes[i].pose.p[0].detach().cpu().numpy()
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        planner.gripper_state = 1
        move(x, y, max(z + 0.14, z_lift - 0.02))
        move(x, y, z + 0.003)
        planner.gripper_state = -1
        hold(GRIPPER_CLOSE_STEPS)
        move(x, y, z_lift)
        return grasping_any()

    def place_at(x, y):
        move(x, y, z_lift)
        move(x, y, z_place)
        planner.gripper_state = 1
        hold(GRIPPER_OPEN_STEPS)
        move(x, y, z_lift)
        hold(SETTLE_STEPS)

    def press(x, y, z_top):
        planner.gripper_state = -1
        hold(3)
        z_app = max(z_top + 0.10, z_lift - 0.02)
        move(x, y, z_app)
        move(x, y, z_top - float(env_u.BUTTON_CAP_TRAVEL) * 0.6)
        hold(5)
        move(x, y, z_app)

    def look_at(x, y, z=0.20, k=6):
        move(x, y, z)
        hold(k)

    def buttons_xy():
        return [tuple(env_u.buttons_xy[b, 0].detach().cpu().numpy().tolist()) for b in range(3)]

    def place_targets():
        cx, cy = env_u.disc_xy[0].detach().cpu().numpy().tolist()
        n = int(env_u.N_CUBES)
        r = float(np.clip(float(env_u.DISC_RADIUS) - cube_hs - 0.012, 0.05, 0.075))
        return [(cx + r * np.cos(np.pi / 2 + 2 * np.pi * i / n), cy + r * np.sin(np.pi / 2 + 2 * np.pi * i / n))
                for i in range(n)]

    return types.SimpleNamespace(planner=planner, done=done, elapsed=elapsed, move=move, hold=hold,
                                 pick_cube=pick_cube, place_at=place_at, press=press, look_at=look_at,
                                 buttons_xy=buttons_xy, place_targets=place_targets, z_lift=z_lift,
                                 button_top_z=button_top_z, grasping_any=grasping_any)
