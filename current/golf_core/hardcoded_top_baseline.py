import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import CLUB_PRESETS
from golf_core.right_arm_cem import JOINT_NAMES, address_pose, interpolate_pose
from golf_core.right_arm_static import (
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    deg,
    hand_sign,
    make_static_model,
    normalize_hand,
    print_setup_report,
)


HOLD_SETUP_STEPS = 90
TAKEAWAY_STEPS = 240
HOLD_TAKEAWAY_STEPS = 10_000

TAKEAWAY_SHOULDER_TURN_DEG = 45.0
TAKEAWAY_FOREARM_SUPINATION_DEG = 45.0


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def takeaway_pose(hand=DEFAULT_HAND):
    sign = hand_sign(hand)
    supination_sign = -sign
    pose = list(address_pose(hand))
    pose[0] = sign * deg(TAKEAWAY_SHOULDER_TURN_DEG)
    pose[6] = supination_sign * deg(TAKEAWAY_FOREARM_SUPINATION_DEG)
    return tuple(pose)


def hardcoded_target_pose(step, hand=DEFAULT_HAND):
    address = address_pose(hand)
    takeaway = takeaway_pose(hand)

    if step < HOLD_SETUP_STEPS:
        return address

    takeaway_step = step - HOLD_SETUP_STEPS
    if takeaway_step < TAKEAWAY_STEPS:
        return interpolate_pose(address, takeaway, takeaway_step / TAKEAWAY_STEPS)

    return takeaway


def apply_kinematic_pose(model, data, pose):
    for joint_name, value in zip(JOINT_NAMES, pose):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_index = model.jnt_qposadr[joint_id]
        data.qpos[qpos_index] = value
        dof_index = model.jnt_dofadr[joint_id]
        data.qvel[dof_index] = 0.0
    mujoco.mj_forward(model, data)


def format_deg(pose):
    return tuple(round(math.degrees(value), 1) for value in pose)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a hardcoded takeaway baseline.")
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    parser.add_argument("--speed", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    hand = normalize_hand(args.hand)
    model = make_static_model(args.club, hand, club_contact=True, include_actuators=False)
    data = mujoco.MjData(model)
    apply_setup_pose(model, data, hand)

    print("Hardcoded takeaway baseline")
    print("Club:", args.club)
    print("Hand:", hand)
    print("Joint names:", JOINT_NAMES)
    print("Address pose deg:", format_deg(address_pose(hand)))
    print("Takeaway pose deg:", format_deg(takeaway_pose(hand)))
    print_setup_report(model, data, hand)

    step = 0
    total_motion_steps = HOLD_SETUP_STEPS + TAKEAWAY_STEPS
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            pose = hardcoded_target_pose(step, hand)
            apply_kinematic_pose(model, data, pose)

            if step % 100 == 0 or step in (HOLD_SETUP_STEPS, total_motion_steps):
                print("step:", step, "pose_deg:", format_deg(pose))

            viewer.sync()
            step = min(step + 1, total_motion_steps + HOLD_TAKEAWAY_STEPS)
            time.sleep(model.opt.timestep * args.speed)


if __name__ == "__main__":
    main()
