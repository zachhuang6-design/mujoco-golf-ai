"""Static two-arm top-of-backswing pose for visual inspection.

This is an endpoint sketch only. It defines the top position directly so we can
inspect whether the geometry is legal before asking a controller to move there.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import CLUB_PRESETS, ball_x, ball_z, get_club_preset
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


CHEST_TOP_TURN_DEG = 95.0
PREFER_HIGH_HANDS = 1.0
PREFER_TRAIL_SIDE = 0.18
PREFER_RIGHT_ELBOW_FLEX_DEG = 75.0
LEFT_ARM_FRONT_VIEW_COUNTERCLOCKWISE_DEG = 15.0
LEFT_ARM_FRONT_VIEW_WEIGHT = 0.025
TOP_CLUBFACE_CLOCKWISE_ROLL_DEG = 90.0
LEFT_WRIST_TOP_LABEL = "neutral"
RIGHT_WRIST_TOP_LABEL = "fully extended"


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


def swing_plane_axes(points):
    target_axis = (1.0, 0.0, 0.0)
    shoulder_mid = points["chest_center"]
    shoulder_to_ball = (
        ball_x - shoulder_mid[0],
        -shoulder_mid[1],
        ball_z - shoulder_mid[2],
    )
    plane_depth_axis = normalize_vec((0.0, shoulder_to_ball[1], shoulder_to_ball[2]))
    plane_normal = normalize_vec(cross_vec(target_axis, plane_depth_axis))
    plane_depth_axis = normalize_vec(cross_vec(plane_normal, target_axis))
    return (ball_x, 0.0, ball_z), target_axis, plane_depth_axis, plane_normal


def point_plane_distance(point, plane_point, plane_normal):
    return dot_vec(sub_vec(point, plane_point), plane_normal)


def project_point_to_plane(point, plane_point, plane_normal):
    distance = point_plane_distance(point, plane_point, plane_normal)
    return sub_vec(point, scale_vec(plane_normal, distance))


def elbow_angle_deg(shoulder, elbow, wrist):
    upper = normalize_vec(sub_vec(shoulder, elbow))
    forearm = normalize_vec(sub_vec(wrist, elbow))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_vec(upper, forearm)))))


def elbow_flex_deg(shoulder, elbow, wrist):
    return 180.0 - elbow_angle_deg(shoulder, elbow, wrist)


def front_view_angle_deg(vector):
    return math.degrees(math.atan2(vector[2], vector[0]))


def angle_delta_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def elbow_from_lengths(shoulder, wrist, reference_elbow, upper_len, forearm_len):
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = max(length_vec(shoulder_to_wrist), 1e-6)
    distance = min(max(distance, abs(upper_len - forearm_len) + 1e-6), upper_len + forearm_len - 1e-6)
    axis = normalize_vec(shoulder_to_wrist)
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


def choose_left_wrist_on_plane(points, base, hand):
    plane_point, target_axis, plane_depth_axis, plane_normal = swing_plane_axes(base)
    left_shoulder = points["left_shoulder"]
    projection = project_point_to_plane(left_shoulder, plane_point, plane_normal)
    plane_distance = abs(point_plane_distance(left_shoulder, plane_point, plane_normal))

    left_upper_len = length_vec(sub_vec(base["left_elbow"], base["left_shoulder"]))
    left_forearm_len = length_vec(sub_vec(base["left_wrist"], base["left_elbow"]))
    right_upper_len = length_vec(sub_vec(base["right_elbow"], base["right_shoulder"]))
    right_forearm_len = length_vec(sub_vec(base["right_wrist"], base["right_elbow"]))
    left_reach = left_upper_len + left_forearm_len
    hand_spacing = length_vec(sub_vec(base["left_wrist"], base["right_wrist"]))

    circle_radius = math.sqrt(max(left_reach * left_reach - plane_distance * plane_distance, 0.0))
    trail_x_sign = -1.0 if hand == "right" else 1.0
    candidates = []

    for index in range(721):
        theta = 2.0 * math.pi * index / 720.0
        direction = normalize_vec(
            add_vec(
                scale_vec(target_axis, math.cos(theta)),
                scale_vec(plane_depth_axis, math.sin(theta)),
            )
        )
        left_wrist = add_vec(projection, scale_vec(direction, circle_radius))
        right_wrist = add_vec(left_wrist, scale_vec(target_axis, trail_x_sign * hand_spacing))
        right_reach = length_vec(sub_vec(right_wrist, points["right_shoulder"]))
        right_max = right_upper_len + right_forearm_len
        if right_reach > right_max - 1e-5:
            continue

        temp_right_elbow = elbow_from_lengths(
            points["right_shoulder"],
            right_wrist,
            points["right_elbow"],
            right_upper_len,
            right_forearm_len,
        )
        right_flex = elbow_flex_deg(points["right_shoulder"], temp_right_elbow, right_wrist)
        score = (
            PREFER_HIGH_HANDS * left_wrist[2]
            + PREFER_TRAIL_SIDE * trail_x_sign * (left_wrist[0] - points["left_shoulder"][0])
            - 0.002 * abs(right_flex - PREFER_RIGHT_ELBOW_FLEX_DEG)
        )
        front_angle = front_view_angle_deg(sub_vec(left_wrist, points["left_shoulder"]))
        candidates.append((score, front_angle, left_wrist, right_wrist, temp_right_elbow))

    if not candidates:
        raise ValueError("Could not find a legal top position on the swing plane")

    natural = max(candidates, key=lambda candidate: candidate[0])
    target_front_angle = natural[1] + LEFT_ARM_FRONT_VIEW_COUNTERCLOCKWISE_DEG
    best = max(
        candidates,
        key=lambda candidate: candidate[0]
        - LEFT_ARM_FRONT_VIEW_WEIGHT * abs(angle_delta_deg(candidate[1], target_front_angle)),
    )
    return best[2], best[3], best[4]


def top_static_points(hand=DEFAULT_HAND, club_name=DEFAULT_CLUB):
    base = setup_points(hand)
    chest_axis_bottom, chest_axis_top = chest_axis_points(base["chest_center"])
    chest_axis = normalize_vec(sub_vec(chest_axis_top, chest_axis_bottom))
    turn_sign = -1.0 if hand == "right" else 1.0
    chest_angle = turn_sign * deg(CHEST_TOP_TURN_DEG)

    moved = {
        key: rotate_point(value, base["chest_center"], chest_axis, chest_angle)
        for key, value in base.items()
        if key.endswith("_shoulder")
        or key.endswith("_elbow")
        or key.endswith("_wrist")
        or key in {"chest_center", "clubhead"}
    }
    moved["chest_center"] = base["chest_center"]
    moved["left_elbow_turn_dir"] = rotate_vec(base["left_elbow_turn_dir"], chest_axis, chest_angle)
    moved["right_elbow_turn_dir"] = rotate_vec(base["right_elbow_turn_dir"], chest_axis, chest_angle)

    left_wrist, right_wrist, right_elbow = choose_left_wrist_on_plane(moved, base, hand)
    moved["left_wrist"] = left_wrist
    moved["right_wrist"] = right_wrist

    left_upper_len = length_vec(sub_vec(base["left_elbow"], base["left_shoulder"]))
    left_forearm_len = length_vec(sub_vec(base["left_wrist"], base["left_elbow"]))
    left_reach = left_upper_len + left_forearm_len
    left_axis = normalize_vec(sub_vec(left_wrist, moved["left_shoulder"]))
    moved["left_elbow"] = add_vec(moved["left_shoulder"], scale_vec(left_axis, left_upper_len))
    moved["right_elbow"] = right_elbow

    address_parts = takeaway.club_visual_parts(base, club_name, hand, apply_wrist_supination=False)
    shaft_len = length_vec(sub_vec(address_parts["heel"], address_parts["grip"]))
    grip = takeaway.grip_center(moved)
    shaft_sign = 1.0 if hand == "right" else -1.0
    moved["clubhead"] = add_vec(grip, scale_vec((1.0, 0.0, 0.0), shaft_sign * shaft_len))
    moved["club_roll_angle"] = deg(TOP_CLUBFACE_CLOCKWISE_ROLL_DEG)

    desired_heel = add_vec(grip, scale_vec((1.0, 0.0, 0.0), shaft_sign * shaft_len))
    for _ in range(8):
        parts = takeaway.club_visual_parts(moved, club_name, hand)
        heel_error = sub_vec(desired_heel, parts["heel"])
        moved["clubhead"] = add_vec(moved["clubhead"], heel_error)

    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        0.0,
    )
    moved["progress"] = 0.0
    moved["left_full_reach"] = left_reach
    return moved


def print_top_report(points, club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    preset = get_club_preset(club_name)
    plane_point, target_axis, plane_depth_axis, plane_normal = swing_plane_axes(setup_points(hand))
    parts = takeaway.club_visual_parts(points, club_name, hand)
    shaft_axis = normalize_vec(sub_vec(parts["heel"], parts["grip"]))
    shaft_x_angle = min(
        takeaway.angle_between_deg(shaft_axis, (1.0, 0.0, 0.0)),
        takeaway.angle_between_deg(shaft_axis, (-1.0, 0.0, 0.0)),
    )
    shaft_above_floor = takeaway.angle_between_deg(shaft_axis, (shaft_axis[0], shaft_axis[1], 0.0))
    grip = takeaway.grip_center(points)
    chest_axis_bottom, chest_axis_top = chest_axis_points(points["chest_center"])
    chest_axis = normalize_vec(sub_vec(chest_axis_top, chest_axis_bottom))
    shoulder_bar = sub_vec(points["left_shoulder"], points["right_shoulder"])
    left_arm_front_angle = front_view_angle_deg(sub_vec(points["left_wrist"], points["left_shoulder"]))

    print("Static two-arm top-of-backswing pose")
    print("Club:", preset["label"])
    print("Chest rotation from address degrees:", CHEST_TOP_TURN_DEG)
    print("Chest rotation axis:", [round(v, 4) for v in chest_axis])
    print("Shoulder bar front-view angle degrees:", round(front_view_angle_deg(shoulder_bar), 3))
    print("Left arm front-view angle degrees:", round(left_arm_front_angle, 3))
    print("Left arm added counterclockwise bias degrees:", LEFT_ARM_FRONT_VIEW_COUNTERCLOCKWISE_DEG)
    print("Left elbow flex degrees:", round(elbow_flex_deg(points["left_shoulder"], points["left_elbow"], points["left_wrist"]), 3))
    print("Right elbow flex degrees:", round(elbow_flex_deg(points["right_shoulder"], points["right_elbow"], points["right_wrist"]), 3))
    print("Left arm reach:", round(length_vec(sub_vec(points["left_wrist"], points["left_shoulder"])), 4))
    print("Left full reach target:", round(points["left_full_reach"], 4))
    print("Shaft angle from x-axis degrees:", round(shaft_x_angle, 3))
    print("Shaft angle above floor degrees:", round(shaft_above_floor, 3))
    print("Plane distance left wrist:", round(point_plane_distance(points["left_wrist"], plane_point, plane_normal), 5))
    print("Plane distance grip:", round(point_plane_distance(grip, plane_point, plane_normal), 5))
    print("Plane distance shaft heel:", round(point_plane_distance(parts["heel"], plane_point, plane_normal), 5))
    print("Plane distance face center:", round(point_plane_distance(points["clubhead"], plane_point, plane_normal), 5))
    print("Left wrist:", [round(v, 4) for v in points["left_wrist"]])
    print("Right wrist:", [round(v, 4) for v in points["right_wrist"]])
    print("Clubhead:", [round(v, 4) for v in points["clubhead"]])
    print("Clubhead rigidly locked to shaft:", True)
    print("Clubface clockwise roll about shaft degrees:", TOP_CLUBFACE_CLOCKWISE_ROLL_DEG)
    print("Left wrist top definition:", LEFT_WRIST_TOP_LABEL)
    print("Right wrist top definition:", RIGHT_WRIST_TOP_LABEL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", default=DEFAULT_CLUB, choices=sorted(CLUB_PRESETS))
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["right", "left"])
    args = parser.parse_args()

    points = top_static_points(args.hand, args.club)
    print_top_report(points, args.club, args.hand)
    model = takeaway.make_takeaway_model(args.club, args.hand)
    data = mujoco.MjData(model)
    takeaway.apply_pose(model, data, points, args.club, args.hand)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
