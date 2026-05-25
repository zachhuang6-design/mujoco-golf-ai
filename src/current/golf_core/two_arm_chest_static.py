"""Static two-arm, chest-based golf setup for visual inspection.

This file intentionally does not train or swing yet. It builds a still address
pose so the shoulder/chest geometry can be checked in MuJoCo before we attach a
controller to it.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import (
    CLUB_PRESETS,
    INCH,
    ball_radius,
    ball_x,
    ball_z,
    club_len,
    forearm_len,
    get_club_preset,
    normalize_club_name,
    tee_half_height,
    tee_x,
    tee_z,
)
from golf_core.right_arm_static import (
    axis_visuals_xml,
    cross_vec,
    format_vec,
    matrix_columns_to_quat,
    normalize_vec,
)


DEFAULT_CLUB = "7iron"
DEFAULT_HAND = "right"

CHEST_WIDTH = 20.0 * INCH
RIGHT_SHOULDER_DROP = 2.5 * INCH
RIGHT_HAND_LOWER_ON_GRIP = 3.0 * INCH
STATIC_UPPER_ARM_LEN = 14.5 * INCH
STATIC_FOREARM_LEN = 11.5 * INCH

CLUB_MINOR_ANGLE_TO_POSITIVE_Y_DEG = 60.0
TARGET_CLUB_HEAD_OFFSET_X = -0.035
TARGET_CLUB_HEAD_OFFSET_Z = -0.006

# Center the shoulder span over the grip/clubhead x-position so the lead and
# trail arms use nearly the same reach proportion.
CHEST_CENTER_X = ball_x + TARGET_CLUB_HEAD_OFFSET_X
CHEST_CENTER_Y = 0.46
LEFT_SHOULDER_Z = 1.72
CHEST_AXIS_TILT_TOWARD_NEGATIVE_Y_DEG = 35.0

GRIP_TO_CLUBHEAD_DISTANCE_RIGHT = 0.72
ELBOW_TURN_TOWARD_BALL_DEG = 20.0
ELBOW_BEND_VIEWER_BLEND = 0.92
ELBOW_BEND_UP_BLEND = 0.12
ELBOW_VISUAL_BEND_DEG = 4.0


def deg(value):
    return math.radians(value)


def add_vec(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def sub_vec(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def scale_vec(values, scalar):
    return tuple(value * scalar for value in values)


def dot_vec(a, b):
    return sum(a[i] * b[i] for i in range(3))


def clamp(value, low, high):
    return max(low, min(high, value))


def length_vec(values):
    return math.sqrt(dot_vec(values, values))


def capsule_xml(name, start, end, size, rgba):
    return (
        f'<geom name="{name}" type="capsule" fromto="{format_vec(start)} '
        f'{format_vec(end)}" size="{size:.6f}" rgba="{rgba}" '
        'contype="0" conaffinity="0"/>'
    )


def site_xml(name, pos, size, rgba):
    return (
        f'<site name="{name}" pos="{format_vec(pos)}" size="{size:.6f}" '
        f'rgba="{rgba}"/>'
    )


def box_quat_from_x_axis(direction):
    x_axis = normalize_vec(direction)
    up_guess = (0.0, 0.0, 1.0)
    if abs(dot_vec(x_axis, up_guess)) > 0.92:
        up_guess = (0.0, 1.0, 0.0)
    y_axis = normalize_vec(cross_vec(up_guess, x_axis))
    z_axis = normalize_vec(cross_vec(x_axis, y_axis))
    return matrix_columns_to_quat(x_axis, y_axis, z_axis)


def club_direction_from_grip_to_head(hand):
    sign = -1.0 if hand == "right" else 1.0
    angle_from_vertical = deg(90.0 - CLUB_MINOR_ANGLE_TO_POSITIVE_Y_DEG)
    return normalize_vec(
        (
            0.0,
            sign * math.sin(angle_from_vertical),
            -math.cos(angle_from_vertical),
        )
    )


def setup_points(hand=DEFAULT_HAND):
    hand = hand.lower().strip()
    if hand not in {"right", "left"}:
        raise ValueError("hand must be 'right' or 'left'")

    trail_sign = -1.0 if hand == "right" else 1.0
    lead_sign = -trail_sign
    half_chest = CHEST_WIDTH / 2.0

    chest_center = (CHEST_CENTER_X, CHEST_CENTER_Y, LEFT_SHOULDER_Z - 0.5 * RIGHT_SHOULDER_DROP)
    left_shoulder = (
        CHEST_CENTER_X + lead_sign * half_chest,
        CHEST_CENTER_Y,
        LEFT_SHOULDER_Z,
    )
    right_shoulder = (
        CHEST_CENTER_X + trail_sign * half_chest,
        CHEST_CENTER_Y,
        LEFT_SHOULDER_Z - RIGHT_SHOULDER_DROP,
    )

    clubhead = (
        ball_x + TARGET_CLUB_HEAD_OFFSET_X,
        0.0,
        ball_z + TARGET_CLUB_HEAD_OFFSET_Z,
    )
    shaft_dir = club_direction_from_grip_to_head(hand)
    right_wrist = sub_vec(
        clubhead,
        scale_vec(shaft_dir, GRIP_TO_CLUBHEAD_DISTANCE_RIGHT),
    )
    left_wrist = sub_vec(
        clubhead,
        scale_vec(shaft_dir, GRIP_TO_CLUBHEAD_DISTANCE_RIGHT + RIGHT_HAND_LOWER_ON_GRIP),
    )

    if hand == "left":
        left_wrist, right_wrist = right_wrist, left_wrist

    sternum = chest_center
    right_elbow = elbow_from_shoulder_wrist("right", right_shoulder, right_wrist, sternum)
    left_elbow = elbow_from_shoulder_wrist("left", left_shoulder, left_wrist, sternum)
    right_elbow_turn_dir = elbow_turn_direction(right_shoulder)
    left_elbow_turn_dir = elbow_turn_direction(left_shoulder)

    return {
        "chest_center": chest_center,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "clubhead": clubhead,
        "shaft_dir": shaft_dir,
        "left_elbow_turn_dir": left_elbow_turn_dir,
        "right_elbow_turn_dir": right_elbow_turn_dir,
    }


def elbow_turn_direction(shoulder):
    neutral = (0.0, 1.0, 0.0)
    to_ball = normalize_vec((ball_x - shoulder[0], -shoulder[1], 0.0))
    dot_value = clamp(dot_vec(neutral, to_ball), -1.0, 1.0)
    total_angle = math.acos(dot_value)
    turn_angle = min(deg(ELBOW_TURN_TOWARD_BALL_DEG), total_angle)
    if total_angle < 1e-6:
        return neutral

    fraction = turn_angle / total_angle
    sin_total = math.sin(total_angle)
    neutral_weight = math.sin((1.0 - fraction) * total_angle) / sin_total
    ball_weight = math.sin(fraction * total_angle) / sin_total
    return normalize_vec(
        add_vec(scale_vec(neutral, neutral_weight), scale_vec(to_ball, ball_weight))
    )


def chest_axis_points(center):
    tilt = deg(CHEST_AXIS_TILT_TOWARD_NEGATIVE_Y_DEG)
    axis_up = normalize_vec((0.0, -math.sin(tilt), math.cos(tilt)))
    top = add_vec(center, scale_vec(axis_up, 0.20))
    bottom = sub_vec(center, scale_vec(axis_up, 0.36))
    return bottom, top


def elbow_from_shoulder_wrist(side, shoulder, wrist, sternum):
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = length_vec(shoulder_to_wrist)
    axis = normalize_vec(shoulder_to_wrist)
    upper_fraction = STATIC_UPPER_ARM_LEN / (STATIC_UPPER_ARM_LEN + STATIC_FOREARM_LEN)
    upper_visual_len = max(distance * upper_fraction, 1e-6)
    forearm_visual_len = max(distance * (1.0 - upper_fraction), 1e-6)
    base = add_vec(shoulder, scale_vec(axis, upper_visual_len))

    viewer = (0.0, 1.0, 0.0)
    upward = (0.0, 0.0, 1.0)
    raw_bend_direction = normalize_vec(
        add_vec(
            scale_vec(viewer, ELBOW_BEND_VIEWER_BLEND),
            scale_vec(upward, ELBOW_BEND_UP_BLEND),
        )
    )
    bend_direction = sub_vec(
        raw_bend_direction,
        scale_vec(axis, dot_vec(raw_bend_direction, axis)),
    )
    if length_vec(bend_direction) < 1e-6:
        bend_direction = cross_vec(axis, (1.0, 0.0, 0.0))
    bend_direction = normalize_vec(bend_direction)

    target_bend = deg(ELBOW_VISUAL_BEND_DEG)
    bend_height = target_bend / (1.0 / upper_visual_len + 1.0 / forearm_visual_len)
    return add_vec(base, scale_vec(bend_direction, bend_height))


def swing_plane_visual_xml(points):
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
    quat = matrix_columns_to_quat(target_axis, plane_depth_axis, plane_normal)
    center = (
        ball_x + 0.24,
        0.5 * shoulder_mid[1],
        0.5 * (ball_z + shoulder_mid[2]),
    )
    return f"""
    <geom name="two_arm_swing_plane_visual" type="box" pos="{format_vec(center)}" quat="{format_vec(quat)}" size="0.900 0.900 0.002" rgba="0.1 0.55 1 0.14" contype="0" conaffinity="0"/>
    <geom name="target_line_visual" type="capsule" fromto="{ball_x - 0.450:.6f} 0 {ball_z:.6f} {ball_x + 1.150:.6f} 0 {ball_z:.6f}" size="0.004" rgba="0.1 0.55 1 0.80" contype="0" conaffinity="0"/>
"""


def build_two_arm_static_xml(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    club_name = normalize_club_name(club_name)
    if club_name not in CLUB_PRESETS:
        raise ValueError(f"Unknown club {club_name}")
    preset = get_club_preset(club_name)
    points = setup_points(hand)

    chest = capsule_xml(
        "chest_shoulder_bar",
        points["left_shoulder"],
        points["right_shoulder"],
        0.045,
        "0.55 0.45 0.35 0.95",
    )
    chest_axis_bottom, chest_axis_top = chest_axis_points(points["chest_center"])
    spine = capsule_xml(
        "chest_rotation_axis_35deg",
        chest_axis_bottom,
        chest_axis_top,
        0.030,
        "0.55 0.45 0.35 0.55",
    )
    shaft_top = tuple(
        0.5 * (points["left_wrist"][i] + points["right_wrist"][i]) for i in range(3)
    )
    club_shaft = capsule_xml(
        "club_shaft_visual",
        shaft_top,
        points["clubhead"],
        0.011,
        "0.05 0.05 0.05 1",
    )
    left_elbow_turn_end = add_vec(
        points["left_elbow"],
        scale_vec(points["left_elbow_turn_dir"], 0.15),
    )
    right_elbow_turn_end = add_vec(
        points["right_elbow"],
        scale_vec(points["right_elbow_turn_dir"], 0.15),
    )
    club_head_quat = box_quat_from_x_axis((1.0, 0.0, 0.0))

    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>
    {axis_visuals_xml()}
    {swing_plane_visual_xml(points)}

    {chest}
    {spine}
    {site_xml("sternum_site", points["chest_center"], 0.035, "1 0.85 0.1 1")}

    {capsule_xml("left_upper_arm", points["left_shoulder"], points["left_elbow"], 0.034, "0.25 0.35 1.0 1")}
    {capsule_xml("left_forearm", points["left_elbow"], points["left_wrist"], 0.030, "0.2 0.9 0.35 1")}
    {capsule_xml("right_upper_arm", points["right_shoulder"], points["right_elbow"], 0.034, "0.25 0.35 1.0 1")}
    {capsule_xml("right_forearm", points["right_elbow"], points["right_wrist"], 0.030, "0.2 0.9 0.35 1")}
    {capsule_xml("left_elbow_turn_marker", points["left_elbow"], left_elbow_turn_end, 0.008, "1 0.45 0 1")}
    {capsule_xml("right_elbow_turn_marker", points["right_elbow"], right_elbow_turn_end, 0.008, "1 0.45 0 1")}

    {site_xml("left_shoulder_site", points["left_shoulder"], 0.032, "0.1 0.1 1 1")}
    {site_xml("right_shoulder_site", points["right_shoulder"], 0.032, "0.1 0.1 1 1")}
    {site_xml("left_elbow_site", points["left_elbow"], 0.027, "1 0.55 0 1")}
    {site_xml("right_elbow_site", points["right_elbow"], 0.027, "1 0.55 0 1")}
    {site_xml("left_wrist_site", points["left_wrist"], 0.025, "0 0.75 0.9 1")}
    {site_xml("right_wrist_site", points["right_wrist"], 0.025, "0 0.75 0.9 1")}

    {club_shaft}
    <geom name="club_head_visual" type="box" pos="{format_vec(points["clubhead"])}" quat="{format_vec(club_head_quat)}" size="0.018 0.050 0.032" mass="{preset["head_mass"]:.6f}" rgba="0.18 0.18 0.18 1" contype="0" conaffinity="0"/>
    {site_xml("clubhead_site", points["clubhead"], 0.018, "1 0.85 0 1")}

    <body name="tee" pos="{tee_x:.6f} 0 {tee_z:.6f}">
      <geom name="tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>
    <body name="ball" pos="{ball_x:.6f} 0 {ball_z:.6f}">
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_two_arm_static_model(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    return mujoco.MjModel.from_xml_string(build_two_arm_static_xml(club_name, hand))


def print_setup_report(points):
    print("Two-arm chest static setup")
    print("Chest width inches:", round(CHEST_WIDTH / INCH, 2))
    print("Chest center y:", round(CHEST_CENTER_Y, 4))
    print("Chest rotation axis tilt toward -y degrees:", round(CHEST_AXIS_TILT_TOWARD_NEGATIVE_Y_DEG, 2))
    print("Right shoulder lower inches:", round(RIGHT_SHOULDER_DROP / INCH, 2))
    print("Right hand lower on grip inches:", round(RIGHT_HAND_LOWER_ON_GRIP / INCH, 2))
    print("Upper arm visual inches:", round(STATIC_UPPER_ARM_LEN / INCH, 2))
    print("Forearm visual inches:", round(STATIC_FOREARM_LEN / INCH, 2))
    print("Elbow turn marker degrees toward ball:", round(ELBOW_TURN_TOWARD_BALL_DEG, 2))
    print("Elbow visual bend target degrees:", round(ELBOW_VISUAL_BEND_DEG, 2))
    print("Left shoulder:", [round(v, 4) for v in points["left_shoulder"]])
    print("Right shoulder:", [round(v, 4) for v in points["right_shoulder"]])
    print("Left wrist:", [round(v, 4) for v in points["left_wrist"]])
    print("Right wrist:", [round(v, 4) for v in points["right_wrist"]])
    print("Clubhead:", [round(v, 4) for v in points["clubhead"]])
    print(
        "Left arm reach:",
        round(length_vec(sub_vec(points["left_wrist"], points["left_shoulder"])), 4),
        "of max",
        round(STATIC_UPPER_ARM_LEN + STATIC_FOREARM_LEN, 4),
    )
    print(
        "Right arm reach:",
        round(length_vec(sub_vec(points["right_wrist"], points["right_shoulder"])), 4),
        "of max",
        round(STATIC_UPPER_ARM_LEN + STATIC_FOREARM_LEN, 4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", default=DEFAULT_CLUB, choices=sorted(CLUB_PRESETS))
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["right", "left"])
    args = parser.parse_args()

    points = setup_points(args.hand)
    print_setup_report(points)
    model = make_two_arm_static_model(args.club, args.hand)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
