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
from golf_core import two_arm_chest_top_static as top_static
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


TAKEAWAY_PHASE = 0.55
CONTINUATION_LABEL = "valid takeaway to legal arm arc toward top"

MAX_ADDED_WRIST_ROLL_DEG = 12.0
TRAIL_ELBOW_FLEX_LIMIT_DEG = 90.0
LEAD_ELBOW_FLEX_LIMIT_DEG = 5.0

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


def lerp_vec(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def grip_center(points):
    return takeaway.grip_center(points)


def elbow_angle_deg(shoulder, elbow, wrist):
    upper = normalize_vec(sub_vec(shoulder, elbow))
    forearm = normalize_vec(sub_vec(wrist, elbow))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_vec(upper, forearm)))))


def elbow_flex_deg(shoulder, elbow, wrist):
    return 180.0 - elbow_angle_deg(shoulder, elbow, wrist)


def angle_delta_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def shortest_angle_lerp(a, b, t):
    return a + angle_delta_deg(b, a) * t


def signed_angle_about_axis(a, b, axis):
    a = normalize_vec(sub_vec(a, scale_vec(axis, dot_vec(a, axis))))
    b = normalize_vec(sub_vec(b, scale_vec(axis, dot_vec(b, axis))))
    axis = normalize_vec(axis)
    return math.atan2(dot_vec(axis, cross_vec(a, b)), dot_vec(a, b))


def rotate_direction_toward(start_vec, target_vec, progress):
    start_len = length_vec(start_vec)
    target_len = length_vec(target_vec)
    if start_len < 1e-8 or target_len < 1e-8:
        return lerp_vec(start_vec, target_vec, progress)

    start_dir = normalize_vec(start_vec)
    target_dir = normalize_vec(target_vec)
    axis, angle = takeaway.rotation_between(start_dir, target_dir)
    direction = rotate_vec(start_dir, axis, angle * progress)
    length = start_len * (1.0 - progress) + target_len * progress
    return scale_vec(direction, length)


def two_stage_direction(start_vec, guide_vec, target_vec, progress):
    if progress < 0.55:
        return rotate_direction_toward(start_vec, guide_vec, smoothstep(progress / 0.55))
    return rotate_direction_toward(guide_vec, target_vec, smoothstep((progress - 0.55) / 0.45))


def two_stage_transport(vec, start_axis, guide_axis, target_axis, progress):
    first_axis, first_angle = takeaway.rotation_between(start_axis, guide_axis)
    if progress < 0.55:
        return rotate_vec(vec, first_axis, first_angle * smoothstep(progress / 0.55))

    carried_to_guide = rotate_vec(vec, first_axis, first_angle)
    second_axis, second_angle = takeaway.rotation_between(guide_axis, target_axis)
    return rotate_vec(
        carried_to_guide,
        second_axis,
        second_angle * smoothstep((progress - 0.55) / 0.45),
    )


def legal_shaft_guide_direction(hand):
    side = 1.0 if hand == "right" else -1.0
    return normalize_vec((-0.30 * side, 0.55, 0.78))


def rotate_pose_around_spine(start, target, top_progress, base):
    chest_axis_bottom, chest_axis_top = chest_axis_points(base["chest_center"])
    chest_axis = normalize_vec(sub_vec(chest_axis_top, chest_axis_bottom))
    start_bar = sub_vec(start["left_shoulder"], start["right_shoulder"])
    target_bar = sub_vec(target["left_shoulder"], target["right_shoulder"])
    chest_turn = signed_angle_about_axis(start_bar, target_bar, chest_axis)

    moved = {"chest_center": start["chest_center"]}
    for key in (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "clubhead",
    ):
        moved[key] = rotate_point(
            start[key],
            start["chest_center"],
            chest_axis,
            chest_turn * top_progress,
        )

    moved["left_elbow_turn_dir"] = rotate_vec(
        start["left_elbow_turn_dir"],
        chest_axis,
        chest_turn * top_progress,
    )
    moved["right_elbow_turn_dir"] = rotate_vec(
        start["right_elbow_turn_dir"],
        chest_axis,
        chest_turn * top_progress,
    )
    return moved


def elbow_from_lengths(shoulder, wrist, reference_elbow, upper_len, forearm_len):
    upper_len = max(upper_len, 1e-6)
    forearm_len = max(forearm_len, 1e-6)
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = max(length_vec(shoulder_to_wrist), 1e-6)

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


def arm_lengths(base, side):
    return (
        length_vec(sub_vec(base[f"{side}_elbow"], base[f"{side}_shoulder"])),
        length_vec(sub_vec(base[f"{side}_wrist"], base[f"{side}_elbow"])),
    )


def blended_arm_lengths(start, base, side, top_progress):
    start_upper_len, start_forearm_len = arm_lengths(start, side)
    base_upper_len, base_forearm_len = arm_lengths(base, side)
    return (
        start_upper_len * (1.0 - top_progress) + base_upper_len * top_progress,
        start_forearm_len * (1.0 - top_progress) + base_forearm_len * top_progress,
    )


def clamp_wrist_to_reach(shoulder, wrist, upper_len, forearm_len):
    reach = upper_len + forearm_len - 1e-6
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = length_vec(shoulder_to_wrist)
    if distance <= reach:
        return wrist
    return add_vec(shoulder, scale_vec(normalize_vec(shoulder_to_wrist), reach))


def blend_points(start, target, top_progress, base, club_name, hand):
    moved = rotate_pose_around_spine(start, target, top_progress, base)
    residual_progress = smoothstep(top_progress)

    moved["left_shoulder"] = lerp_vec(
        moved["left_shoulder"],
        target["left_shoulder"],
        residual_progress,
    )
    moved["right_shoulder"] = lerp_vec(
        moved["right_shoulder"],
        target["right_shoulder"],
        residual_progress,
    )

    for side in ("left", "right"):
        shoulder_key = f"{side}_shoulder"
        wrist_key = f"{side}_wrist"
        current_arm = sub_vec(moved[wrist_key], moved[shoulder_key])
        target_arm = sub_vec(target[wrist_key], target[shoulder_key])
        moved[wrist_key] = add_vec(
            moved[shoulder_key],
            rotate_direction_toward(current_arm, target_arm, residual_progress),
        )

    left_upper_len, left_forearm_len = blended_arm_lengths(start, base, "left", top_progress)
    right_upper_len, right_forearm_len = blended_arm_lengths(start, base, "right", top_progress)
    moved["left_wrist"] = clamp_wrist_to_reach(
        moved["left_shoulder"],
        moved["left_wrist"],
        left_upper_len,
        left_forearm_len,
    )
    moved["right_wrist"] = clamp_wrist_to_reach(
        moved["right_shoulder"],
        moved["right_wrist"],
        right_upper_len,
        right_forearm_len,
    )
    left_arm_axis = normalize_vec(sub_vec(moved["left_wrist"], moved["left_shoulder"]))
    moved["left_wrist"] = add_vec(
        moved["left_shoulder"],
        scale_vec(left_arm_axis, left_upper_len + left_forearm_len - 1e-6),
    )

    moved["left_elbow"] = add_vec(moved["left_shoulder"], scale_vec(left_arm_axis, left_upper_len))
    moved["right_elbow"] = elbow_from_lengths(
        moved["right_shoulder"],
        moved["right_wrist"],
        lerp_vec(start["right_elbow"], target["right_elbow"], top_progress),
        right_upper_len,
        right_forearm_len,
    )
    moved["left_elbow_turn_dir"] = normalize_vec(
        rotate_direction_toward(
            moved["left_elbow_turn_dir"],
            target["left_elbow_turn_dir"],
            residual_progress,
        )
    )
    moved["right_elbow_turn_dir"] = normalize_vec(
        rotate_direction_toward(
            moved["right_elbow_turn_dir"],
            target["right_elbow_turn_dir"],
            residual_progress,
        )
    )

    start_grip = grip_center(start)
    target_grip = grip_center(target)
    moved_grip = grip_center(moved)
    start_shaft_axis = normalize_vec(sub_vec(start["clubhead"], start_grip))
    target_shaft_axis = normalize_vec(sub_vec(target["clubhead"], target_grip))
    guide_shaft_axis = legal_shaft_guide_direction(base.get("hand", DEFAULT_HAND))
    shaft_axis = normalize_vec(
        two_stage_direction(
            start_shaft_axis,
            guide_shaft_axis,
            target_shaft_axis,
            top_progress,
        )
    )
    shaft_len = (
        length_vec(sub_vec(start["clubhead"], start_grip)) * (1.0 - top_progress)
        + length_vec(sub_vec(target["clubhead"], target_grip)) * top_progress
    )
    moved["clubhead"] = add_vec(moved_grip, scale_vec(shaft_axis, shaft_len))

    start_parts = takeaway.club_visual_parts(
        start,
        club_name,
        hand,
        apply_wrist_supination=False,
    )
    start_visual_shaft = normalize_vec(sub_vec(start_parts["heel"], start_parts["grip"]))
    moved["club_face_normal"] = two_stage_transport(
        start_parts["face_normal"],
        start_visual_shaft,
        guide_shaft_axis,
        shaft_axis,
        top_progress,
    )
    moved["club_toe_axis"] = two_stage_transport(
        start_parts["toe_axis"],
        start_visual_shaft,
        guide_shaft_axis,
        shaft_axis,
        top_progress,
    )
    moved["club_top_axis"] = two_stage_transport(
        start_parts["top_axis"],
        start_visual_shaft,
        guide_shaft_axis,
        shaft_axis,
        top_progress,
    )

    max_added_roll = deg(MAX_ADDED_WRIST_ROLL_DEG)
    moved["club_roll_angle"] = shortest_angle_lerp(
        start.get("club_roll_angle", 0.0),
        start.get("club_roll_angle", 0.0) + max_added_roll,
        top_progress,
    )
    return moved


def pose_points(hand=DEFAULT_HAND, progress=0.0, club_name=DEFAULT_CLUB):
    progress = max(0.0, min(1.0, progress))
    if progress <= TAKEAWAY_PHASE:
        return takeaway.pose_points(hand, progress / TAKEAWAY_PHASE)

    base = setup_points(hand)
    base["hand"] = hand
    start = dict(takeaway.pose_points(hand, 1.0))
    start["club_roll_angle"] = takeaway.takeaway_wrist_supination_roll_angle(club_name, hand)
    target = top_static.top_static_points(hand, club_name)
    top_progress = smoothstep((progress - TAKEAWAY_PHASE) / (1.0 - TAKEAWAY_PHASE))
    moved = blend_points(start, target, top_progress, base, club_name, hand)
    supination_sign = -1.0 if hand == "right" else 1.0
    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        supination_sign * deg(takeaway.RIGHT_FOREARM_SUPINATION_DEG),
    )
    moved["progress"] = 1.0 - top_progress
    return moved


def print_backswing_report(hand, club_name=DEFAULT_CLUB):
    preset = get_club_preset(club_name)
    start = dict(takeaway.pose_points(hand, 1.0))
    start["club_roll_angle"] = takeaway.takeaway_wrist_supination_roll_angle(club_name, hand)
    top = pose_points(hand, 1.0, club_name)
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
    print("Continuation after takeaway:", CONTINUATION_LABEL)
    print("Top left elbow flex degrees:", round(elbow_flex_deg(top["left_shoulder"], top["left_elbow"], top["left_wrist"]), 3))
    print("Top right elbow flex degrees:", round(elbow_flex_deg(top["right_shoulder"], top["right_elbow"], top["right_wrist"]), 3))
    print("Top shaft angle from x-axis degrees:", round(x_parallel_angle, 3))
    print("Top shaft angle above floor degrees:", round(shaft_horizontal_angle, 3))
    print("Club roll from takeaway to top degrees:", round(math.degrees(top.get("club_roll_angle", 0.0) - start.get("club_roll_angle", 0.0)), 3))
    print("Max added wrist/club roll degrees:", MAX_ADDED_WRIST_ROLL_DEG)
    print("Lead elbow flex limit degrees:", LEAD_ELBOW_FLEX_LIMIT_DEG)
    print("Trail elbow flex limit degrees:", TRAIL_ELBOW_FLEX_LIMIT_DEG)
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
    takeaway.apply_pose(model, data, pose_points(args.hand, 0.0, args.club), args.club, args.hand)

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

            takeaway.apply_pose(model, data, pose_points(args.hand, progress, args.club), args.club, args.hand)
            viewer.sync()
            step = (step + 1) % total_steps
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
