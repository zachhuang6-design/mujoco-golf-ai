"""Animated two-arm chest backswing-to-top visual.

This builds on the approved address/takeaway model and extends it to a
top-of-backswing checkpoint. It is still a kinematic sketch, not a trained
physics controller.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import CLUB_PRESETS, get_club_preset
from golf_core import two_arm_chest_takeaway as takeaway
from golf_core.two_arm_chest_static import (
    DEFAULT_CLUB,
    DEFAULT_HAND,
    add_vec,
    chest_axis_points,
    deg,
    dot_vec,
    length_vec,
    scale_vec,
    setup_points,
    sub_vec,
)
from golf_core.right_arm_static import cross_vec, normalize_vec


CHEST_TOP_TURN_DEG = 90.0
TAKEAWAY_PHASE = 0.55
LEFT_TOP_ELBOW_FLEX_DEG = 0.0

SETUP_HOLD_SECONDS = 3.0
BACKSWING_STEPS = 260
PAUSE_STEPS = 10_000


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def rotate_vec(vec, axis, angle):
    axis = normalize_vec(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = cross_vec(axis, vec)
    dot = dot_vec(axis, vec)
    return add_vec(
        add_vec(scale_vec(vec, cos_a), scale_vec(cross, sin_a)),
        scale_vec(axis, dot * (1.0 - cos_a)),
    )


def rotate_point(point, origin, axis, angle):
    return add_vec(origin, rotate_vec(sub_vec(point, origin), axis, angle))


def grip_center(points):
    return takeaway.grip_center(points)


def elbow_angle_deg(shoulder, elbow, wrist):
    upper = normalize_vec(sub_vec(shoulder, elbow))
    forearm = normalize_vec(sub_vec(wrist, elbow))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_vec(upper, forearm)))))


def elbow_flex_deg(shoulder, elbow, wrist):
    return 180.0 - elbow_angle_deg(shoulder, elbow, wrist)


def elbow_from_flexion(shoulder, wrist, reference_elbow, flex_deg, upper_len, forearm_len):
    upper_len = max(upper_len, 1e-6)
    forearm_len = max(forearm_len, 1e-6)
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = max(length_vec(shoulder_to_wrist), 1e-6)

    if flex_deg <= 1e-6:
        total = upper_len + forearm_len
        return add_vec(shoulder, scale_vec(shoulder_to_wrist, upper_len / total))

    axis = normalize_vec(shoulder_to_wrist)
    distance = min(max(distance, abs(upper_len - forearm_len) + 1e-6), upper_len + forearm_len - 1e-6)
    along = (upper_len * upper_len - forearm_len * forearm_len + distance * distance) / (2.0 * distance)
    height = math.sqrt(max(upper_len * upper_len - along * along, 0.0))

    reference_offset = sub_vec(reference_elbow, shoulder)
    bend_direction = sub_vec(reference_offset, scale_vec(axis, dot_vec(reference_offset, axis)))
    if length_vec(bend_direction) < 1e-6:
        bend_direction = cross_vec(axis, (0.0, 0.0, 1.0))
    if length_vec(bend_direction) < 1e-6:
        bend_direction = cross_vec(axis, (0.0, 1.0, 0.0))
    bend_direction = normalize_vec(bend_direction)

    return add_vec(
        add_vec(shoulder, scale_vec(axis, along)),
        scale_vec(bend_direction, height),
    )


def pose_points(hand=DEFAULT_HAND, progress=0.0):
    progress = max(0.0, min(1.0, progress))
    if progress <= TAKEAWAY_PHASE:
        return takeaway.pose_points(hand, progress / TAKEAWAY_PHASE)

    base = setup_points(hand)
    start = takeaway.pose_points(hand, 1.0)
    top_progress = smoothstep((progress - TAKEAWAY_PHASE) / (1.0 - TAKEAWAY_PHASE))
    chest_axis_bottom, chest_axis_top = chest_axis_points(start["chest_center"])
    chest_axis = (0.0, 0.0, 1.0)
    turn_sign = -1.0 if hand == "right" else 1.0
    extra_turn_deg = CHEST_TOP_TURN_DEG - takeaway.CHEST_TAKEAWAY_TURN_DEG
    chest_angle = turn_sign * deg(extra_turn_deg) * top_progress

    moved = {
        key: rotate_point(value, start["chest_center"], chest_axis, chest_angle)
        for key, value in start.items()
        if key.endswith("_shoulder")
        or key.endswith("_elbow")
        or key.endswith("_wrist")
        or key in {"chest_center", "clubhead"}
    }
    moved["chest_center"] = start["chest_center"]
    moved["left_elbow_turn_dir"] = rotate_vec(start["left_elbow_turn_dir"], chest_axis, chest_angle)
    moved["right_elbow_turn_dir"] = rotate_vec(start["right_elbow_turn_dir"], chest_axis, chest_angle)

    start_left_flex = elbow_flex_deg(
        start["left_shoulder"],
        start["left_elbow"],
        start["left_wrist"],
    )
    left_flex_deg = start_left_flex + (LEFT_TOP_ELBOW_FLEX_DEG - start_left_flex) * top_progress

    moved["left_elbow"] = elbow_from_flexion(
        moved["left_shoulder"],
        moved["left_wrist"],
        moved["left_elbow"],
        left_flex_deg,
        length_vec(sub_vec(base["left_elbow"], base["left_shoulder"])),
        length_vec(sub_vec(base["left_wrist"], base["left_elbow"])),
    )

    supination_sign = -1.0 if hand == "right" else 1.0
    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        supination_sign * deg(takeaway.RIGHT_FOREARM_SUPINATION_DEG),
    )
    moved["progress"] = 1.0
    return moved


def print_backswing_report(hand, club_name=DEFAULT_CLUB):
    preset = get_club_preset(club_name)
    top = pose_points(hand, 1.0)
    parts = takeaway.club_visual_parts(top, club_name, hand)
    shaft_axis = normalize_vec(sub_vec(parts["heel"], parts["grip"]))
    x_parallel_angle = min(
        takeaway.angle_between_deg(shaft_axis, (1.0, 0.0, 0.0)),
        takeaway.angle_between_deg(shaft_axis, (-1.0, 0.0, 0.0)),
    )
    shaft_horizontal_angle = takeaway.angle_between_deg(shaft_axis, (shaft_axis[0], shaft_axis[1], 0.0))

    print("Two-arm chest backswing-to-top visual")
    print("Club:", preset["label"])
    print("First phase uses valid takeaway:", True)
    print("Takeaway phase share:", TAKEAWAY_PHASE)
    print("Takeaway chest turn degrees:", takeaway.CHEST_TAKEAWAY_TURN_DEG)
    print("Chest turn degrees:", CHEST_TOP_TURN_DEG)
    print("Continuation after takeaway:", "chest turn only")
    print("Left elbow flex target degrees:", LEFT_TOP_ELBOW_FLEX_DEG)
    print("Top left elbow flex degrees:", round(elbow_flex_deg(top["left_shoulder"], top["left_elbow"], top["left_wrist"]), 3))
    print("Top right elbow flex degrees:", round(elbow_flex_deg(top["right_shoulder"], top["right_elbow"], top["right_wrist"]), 3))
    print("Top shaft angle from x-axis degrees:", round(x_parallel_angle, 3))
    print("Top shaft angle above floor degrees:", round(shaft_horizontal_angle, 3))
    print("Top grip center:", [round(v, 4) for v in grip_center(top)])
    print("Top clubhead:", [round(v, 4) for v in top["clubhead"]])
    print("Clubhead rigidly locked to shaft:", True)
    print("Address pause seconds:", SETUP_HOLD_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", default=DEFAULT_CLUB, choices=sorted(CLUB_PRESETS))
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["right", "left"])
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    print_backswing_report(args.hand, args.club)
    model = takeaway.make_takeaway_model(args.club, args.hand)
    data = mujoco.MjData(model)
    takeaway.apply_pose(model, data, pose_points(args.hand, 0.0), args.club, args.hand)

    sleep_time = max(0.001, 0.01 / max(args.speed, 0.1))
    setup_hold_steps = max(1, int(SETUP_HOLD_SECONDS / sleep_time))
    step = 0
    total_steps = setup_hold_steps + BACKSWING_STEPS + PAUSE_STEPS
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if step < setup_hold_steps:
                progress = 0.0
            elif step < setup_hold_steps + BACKSWING_STEPS:
                progress = (step - setup_hold_steps) / BACKSWING_STEPS
            else:
                progress = 1.0

            takeaway.apply_pose(model, data, pose_points(args.hand, progress), args.club, args.hand)
            viewer.sync()
            step = (step + 1) % total_steps
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
