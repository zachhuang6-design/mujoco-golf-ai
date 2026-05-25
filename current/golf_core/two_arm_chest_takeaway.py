"""Animated two-arm chest takeaway visual.

This is a kinematic sketch, not a trained physics controller. It starts from
the approved static address pose, rotates the chest, adds a small elbow flex,
adds complementary lead/trail wrist set, then pauses for visual feedback.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import (
    BALL_CONTACT_ATTRS,
    CLUB_PRESETS,
    ball_mass,
    ball_radius,
    ball_x,
    ball_z,
    get_club_preset,
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
from golf_core.two_arm_chest_static import (
    DEFAULT_CLUB,
    DEFAULT_HAND,
    ELBOW_TURN_TOWARD_BALL_DEG,
    ELBOW_VISUAL_BEND_DEG,
    RIGHT_SHOULDER_DROP,
    add_vec,
    box_quat_from_x_axis,
    chest_axis_points,
    deg,
    dot_vec,
    length_vec,
    scale_vec,
    setup_points,
    sub_vec,
    swing_plane_visual_xml,
)


CHEST_TAKEAWAY_TURN_DEG = 60.0
RIGHT_WRIST_EXTENSION_DEG = 15.0
RIGHT_WRIST_ULNAR_DEVIATION_DEG = 15.0
LEFT_WRIST_FLEXION_DEG = 15.0
LEFT_WRIST_RADIAL_DEVIATION_DEG = 15.0
LEFT_TAKEAWAY_ELBOW_BEND_DEG = 0.5
RIGHT_TAKEAWAY_ELBOW_BEND_DEG = 12.0
RIGHT_FOREARM_SUPINATION_DEG = 20.0
KEEP_WRISTS_ON_X_TRACK = False
ROTATE_CHEST_AROUND_VERTICAL_FOR_TAKEAWAY = True
TARGET_NEGATIVE_X_SHAFT_TAKEAWAY = True
LEFT_ARM_TAKEAWAY_VERTICAL_DROP = 0.42
LEFT_ARM_TAKEAWAY_Y_OFFSET = 0.04
CLUB_LIE_DEG = 60.0
CLUB_HEAD_HALF_THICKNESS = 0.018
CLUB_HEAD_HALF_TOE_WIDTH = 0.050
CLUB_HEAD_HALF_HEIGHT = 0.032
CLUB_HEAD_TOE_SIDE = -1.0
DISPLAY_BALL_OFFSET_X = 0.035
MATCH_SOLE_TILT_WITH_WRIST_SUPINATION = True

SETUP_HOLD_SECONDS = 3.0
TAKEAWAY_STEPS = 180
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


def segment_quat(start, end):
    return box_quat_from_x_axis(sub_vec(end, start))


def rotation_between(a, b):
    a = normalize_vec(a)
    b = normalize_vec(b)
    axis = cross_vec(a, b)
    axis_len = length_vec(axis)
    dot = max(-1.0, min(1.0, dot_vec(a, b)))
    if axis_len < 1e-8:
        if dot > 0.0:
            return (1.0, 0.0, 0.0), 0.0
        fallback = cross_vec(a, (0.0, 0.0, 1.0))
        if length_vec(fallback) < 1e-8:
            fallback = cross_vec(a, (0.0, 1.0, 0.0))
        return normalize_vec(fallback), math.pi
    return normalize_vec(axis), math.atan2(axis_len, dot)


def angle_between_deg(a, b):
    a = normalize_vec(a)
    b = normalize_vec(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_vec(a, b)))))


def signed_angle_about_axis(a, b, axis):
    a = normalize_vec(a)
    b = normalize_vec(b)
    axis = normalize_vec(axis)
    return math.atan2(dot_vec(axis, cross_vec(a, b)), dot_vec(a, b))


def loft_angle_in_xz_deg(top_axis):
    xz_axis = normalize_vec((top_axis[0], 0.0, top_axis[2]))
    angle = angle_between_deg(xz_axis, (0.0, 0.0, 1.0))
    return min(angle, 180.0 - angle)


def tilt_from_vertical_deg(axis):
    angle = angle_between_deg(axis, (0.0, 0.0, 1.0))
    return min(angle, 180.0 - angle)


def solve_shaft_roll_for_tilt(shaft_axis, toe_axis, target_tilt_deg):
    best_angle = 0.0
    best_score = float("inf")
    for index in range(721):
        angle = -math.pi + (2.0 * math.pi * index / 720.0)
        candidate_toe = rotate_vec(toe_axis, shaft_axis, angle)
        error = abs(tilt_from_vertical_deg(candidate_toe) - target_tilt_deg)
        score = error + 0.01 * abs(math.degrees(angle))
        if score < best_score:
            best_score = score
            best_angle = angle
    return best_angle


def takeaway_wrist_supination_roll_angle(club_name, hand=DEFAULT_HAND):
    takeaway = pose_points(hand, 1.0)
    parts = club_visual_parts(
        takeaway,
        club_name,
        hand,
        apply_wrist_supination=False,
    )
    spine_bottom, spine_top = chest_axis_points(takeaway["chest_center"])
    spine_axis = normalize_vec(sub_vec(spine_top, spine_bottom))
    target_tilt = tilt_from_vertical_deg(spine_axis)
    shaft_axis = normalize_vec(sub_vec(parts["heel"], parts["grip"]))
    return solve_shaft_roll_for_tilt(shaft_axis, parts["toe_axis"], target_tilt)


def apply_wrist_supination_roll(points, club_name, hand, shaft_axis, face_normal, toe_axis, top_axis):
    if "club_roll_angle" in points:
        roll_angle = points["club_roll_angle"]
        return (
            rotate_vec(face_normal, shaft_axis, roll_angle),
            rotate_vec(toe_axis, shaft_axis, roll_angle),
            rotate_vec(top_axis, shaft_axis, roll_angle),
        )
    if not MATCH_SOLE_TILT_WITH_WRIST_SUPINATION:
        return face_normal, toe_axis, top_axis
    roll_angle = smoothstep(points.get("progress", 0.0)) * takeaway_wrist_supination_roll_angle(
        club_name,
        hand,
    )
    return (
        rotate_vec(face_normal, shaft_axis, roll_angle),
        rotate_vec(toe_axis, shaft_axis, roll_angle),
        rotate_vec(top_axis, shaft_axis, roll_angle),
    )


def face_normal_for_loft(club_name):
    loft = deg(get_club_preset(club_name)["loft_deg"])
    return normalize_vec((math.cos(loft), 0.0, math.sin(loft)))


def sole_axis_for_lie(face_normal, shaft_axis, lie_deg):
    shaft_projection = sub_vec(
        shaft_axis,
        scale_vec(face_normal, dot_vec(shaft_axis, face_normal)),
    )
    projection_len = length_vec(shaft_projection)
    if projection_len < 1e-6:
        return (0.0, CLUB_HEAD_TOE_SIDE, 0.0)

    along_projection = normalize_vec(shaft_projection)
    across_face = normalize_vec(cross_vec(face_normal, along_projection))
    desired_dot = math.cos(deg(lie_deg))
    along_amount = max(-1.0, min(1.0, desired_dot / projection_len))
    across_amount = CLUB_HEAD_TOE_SIDE * math.sqrt(max(1.0 - along_amount * along_amount, 0.0))
    return normalize_vec(
        add_vec(
            scale_vec(along_projection, along_amount),
            scale_vec(across_face, across_amount),
        )
    )


def address_club_axes(club_name, hand=DEFAULT_HAND):
    address = setup_points(hand)
    address_shaft_axis = normalize_vec(sub_vec(address["clubhead"], grip_center(address)))
    address_face_normal = face_normal_for_loft(club_name)
    address_toe_axis = sole_axis_for_lie(
        address_face_normal,
        address_shaft_axis,
        CLUB_LIE_DEG,
    )
    address_top_axis = normalize_vec(cross_vec(address_face_normal, address_toe_axis))
    return address_shaft_axis, address_face_normal, address_toe_axis, address_top_axis


def club_axes(points, club_name, hand=DEFAULT_HAND):
    grip = grip_center(points)
    shaft_axis = normalize_vec(sub_vec(points["clubhead"], grip))
    address_shaft_axis, address_face_normal, address_toe_axis, address_top_axis = (
        address_club_axes(club_name, hand)
    )

    rotation_axis, rotation_angle = rotation_between(address_shaft_axis, shaft_axis)
    face_normal = rotate_vec(address_face_normal, rotation_axis, rotation_angle)
    toe_axis = rotate_vec(address_toe_axis, rotation_axis, rotation_angle)
    top_axis = rotate_vec(address_top_axis, rotation_axis, rotation_angle)

    return normalize_vec(face_normal), normalize_vec(toe_axis), normalize_vec(top_axis)


def club_body_axes(points, club_name, hand=DEFAULT_HAND):
    return club_visual_parts(points, club_name, hand)["body_axes"]


def to_local(vec, axes):
    return tuple(dot_vec(vec, axis) for axis in axes)


def address_face_center_shift(hand=DEFAULT_HAND):
    address = setup_points(hand)
    return (
        ball_x - address["clubhead"][0],
        0.0 - address["clubhead"][1],
        ball_z - address["clubhead"][2],
    )


def visual_face_center(points, hand=DEFAULT_HAND):
    return add_vec(points["clubhead"], address_face_center_shift(hand))


def club_visual_parts(points, club_name, hand=DEFAULT_HAND, apply_wrist_supination=True):
    grip = grip_center(points)
    face_center = visual_face_center(points, hand)
    face_normal, toe_axis, top_axis = club_axes(points, club_name, hand)
    head_center = sub_vec(face_center, scale_vec(face_normal, CLUB_HEAD_HALF_THICKNESS))

    for _ in range(8):
        heel = sub_vec(head_center, scale_vec(toe_axis, CLUB_HEAD_HALF_TOE_WIDTH))
        shaft_axis = normalize_vec(sub_vec(heel, grip))
        toe_axis = sole_axis_for_lie(face_normal, shaft_axis, CLUB_LIE_DEG)
        top_axis = normalize_vec(cross_vec(face_normal, toe_axis))

    heel = sub_vec(head_center, scale_vec(toe_axis, CLUB_HEAD_HALF_TOE_WIDTH))
    shaft_axis = normalize_vec(sub_vec(heel, grip))
    if apply_wrist_supination:
        face_normal, toe_axis, top_axis = apply_wrist_supination_roll(
            points,
            club_name,
            hand,
            shaft_axis,
            face_normal,
            toe_axis,
            top_axis,
        )
        head_center = sub_vec(face_center, scale_vec(face_normal, CLUB_HEAD_HALF_THICKNESS))
        heel = sub_vec(head_center, scale_vec(toe_axis, CLUB_HEAD_HALF_TOE_WIDTH))
        shaft_axis = normalize_vec(sub_vec(heel, grip))
    body_y = sub_vec(toe_axis, scale_vec(shaft_axis, dot_vec(toe_axis, shaft_axis)))
    if length_vec(body_y) < 1e-6:
        body_y = sub_vec(face_normal, scale_vec(shaft_axis, dot_vec(face_normal, shaft_axis)))
    if length_vec(body_y) < 1e-6:
        body_y = (0.0, 1.0, 0.0)
    body_y = normalize_vec(body_y)
    body_z = normalize_vec(cross_vec(shaft_axis, body_y))
    body_y = normalize_vec(cross_vec(body_z, shaft_axis))
    return {
        "grip": grip,
        "face_center": face_center,
        "head_center": head_center,
        "heel": heel,
        "face_normal": face_normal,
        "toe_axis": toe_axis,
        "top_axis": top_axis,
        "body_axes": (shaft_axis, body_y, body_z),
    }


def club_head_pose(points, club_name, hand=DEFAULT_HAND):
    parts = club_visual_parts(points, club_name, hand)
    center = parts["head_center"]
    quat = matrix_columns_to_quat(
        parts["face_normal"],
        parts["toe_axis"],
        parts["top_axis"],
    )
    return center, quat


def club_assembly_pose(points, club_name, hand=DEFAULT_HAND):
    parts = club_visual_parts(points, club_name, hand)
    body_x, body_y, body_z = parts["body_axes"]
    quat = matrix_columns_to_quat(body_x, body_y, body_z)
    return parts["grip"], quat


def rigid_club_body_xml(club_name, hand=DEFAULT_HAND):
    address = setup_points(hand)
    parts = club_visual_parts(address, club_name, hand)
    grip = parts["grip"]
    heel = parts["heel"]
    shaft_len = length_vec(sub_vec(heel, grip))
    body_axes = parts["body_axes"]
    face_normal = parts["face_normal"]
    toe_axis = parts["toe_axis"]
    top_axis = parts["top_axis"]
    head_center = parts["head_center"]
    head_pos = to_local(sub_vec(head_center, grip), body_axes)
    head_face = to_local(face_normal, body_axes)
    head_toe = to_local(toe_axis, body_axes)
    head_top = to_local(top_axis, body_axes)
    head_quat = matrix_columns_to_quat(
        normalize_vec(head_face),
        normalize_vec(head_toe),
        normalize_vec(head_top),
    )
    return f"""
    <body name="club_assembly_body" mocap="true">
      <body name="club_shaft_body" pos="0 0 0">
        <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 {shaft_len:.6f} 0 0" size="0.011" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
      </body>
      <body name="club_head_body" pos="{format_vec(head_pos)}" quat="{format_vec(head_quat)}">
        <geom name="club_head_geom" type="box" size="{CLUB_HEAD_HALF_THICKNESS:.6f} {CLUB_HEAD_HALF_TOE_WIDTH:.6f} {CLUB_HEAD_HALF_HEIGHT:.6f}" rgba="0.18 0.18 0.18 1" contype="0" conaffinity="0"/>
      </body>
    </body>
"""


def mocap_capsule_body(name, length, size, rgba):
    half = 0.5 * max(length, 1e-6)
    return f"""
    <body name="{name}" mocap="true">
      <geom name="{name}_geom" type="capsule" fromto="{-half:.6f} 0 0 {half:.6f} 0 0" size="{size:.6f}" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>
"""


def mocap_sphere_body(name, size, rgba):
    return f"""
    <body name="{name}" mocap="true">
      <geom name="{name}_geom" type="sphere" size="{size:.6f}" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>
"""


def mocap_box_body(name, size, rgba):
    return f"""
    <body name="{name}" mocap="true">
      <geom name="{name}_geom" type="box" size="{size}" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>
"""


def build_takeaway_xml(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    preset = get_club_preset(club_name)
    points = setup_points(hand)
    chest_axis_bottom, chest_axis_top = chest_axis_points(points["chest_center"])
    shaft_top = grip_center(points)

    lengths = segment_lengths(points, chest_axis_bottom, chest_axis_top, shaft_top)

    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>
    {axis_visuals_xml()}
    {swing_plane_visual_xml(points)}

    {mocap_capsule_body("chest_shoulder_bar_body", lengths["chest"], 0.045, "0.55 0.45 0.35 0.95")}
    {mocap_capsule_body("chest_rotation_axis_body", lengths["chest_axis"], 0.030, "0.55 0.45 0.35 0.55")}

    {mocap_capsule_body("left_upper_arm_body", lengths["left_upper"], 0.034, "0.25 0.35 1.0 1")}
    {mocap_capsule_body("left_forearm_body", lengths["left_forearm"], 0.030, "0.2 0.9 0.35 1")}
    {mocap_capsule_body("right_upper_arm_body", lengths["right_upper"], 0.034, "0.25 0.35 1.0 1")}
    {mocap_capsule_body("right_forearm_body", lengths["right_forearm"], 0.030, "0.2 0.9 0.35 1")}

    {mocap_capsule_body("left_elbow_turn_marker_body", 0.15, 0.008, "1 0.45 0 1")}
    {mocap_capsule_body("right_elbow_turn_marker_body", 0.15, 0.008, "1 0.45 0 1")}
    {mocap_capsule_body("right_forearm_supination_marker_body", 0.13, 0.007, "0.75 0.25 1 1")}
    {rigid_club_body_xml(club_name, hand)}

    {mocap_sphere_body("sternum_marker_body", 0.035, "1 0.85 0.1 1")}
    {mocap_sphere_body("left_shoulder_marker_body", 0.032, "0.1 0.1 1 1")}
    {mocap_sphere_body("right_shoulder_marker_body", 0.032, "0.1 0.1 1 1")}
    {mocap_sphere_body("left_elbow_marker_body", 0.027, "1 0.55 0 1")}
    {mocap_sphere_body("right_elbow_marker_body", 0.027, "1 0.55 0 1")}
    {mocap_sphere_body("left_wrist_marker_body", 0.025, "0 0.75 0.9 1")}
    {mocap_sphere_body("right_wrist_marker_body", 0.025, "0 0.75 0.9 1")}
    {mocap_sphere_body("clubhead_marker_body", 0.018, "1 0.85 0 1")}

    <body name="tee" pos="{tee_x + DISPLAY_BALL_OFFSET_X:.6f} 0 {tee_z:.6f}">
      <geom name="tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>
    <body name="ball" pos="{ball_x + DISPLAY_BALL_OFFSET_X:.6f} 0 {ball_z:.6f}">
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" {BALL_CONTACT_ATTRS} rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def grip_center(points):
    return tuple(0.5 * (points["left_wrist"][i] + points["right_wrist"][i]) for i in range(3))


def segment_lengths(points, chest_axis_bottom, chest_axis_top, shaft_top):
    return {
        "chest": length_vec(sub_vec(points["left_shoulder"], points["right_shoulder"])),
        "chest_axis": length_vec(sub_vec(chest_axis_top, chest_axis_bottom)),
        "left_upper": length_vec(sub_vec(points["left_elbow"], points["left_shoulder"])),
        "left_forearm": length_vec(sub_vec(points["left_wrist"], points["left_elbow"])),
        "right_upper": length_vec(sub_vec(points["right_elbow"], points["right_shoulder"])),
        "right_forearm": length_vec(sub_vec(points["right_wrist"], points["right_elbow"])),
        "club_shaft": length_vec(sub_vec(points["clubhead"], shaft_top)),
    }


def elbow_with_bend(shoulder, wrist, reference_elbow, bend_deg):
    shoulder_to_wrist = sub_vec(wrist, shoulder)
    distance = max(length_vec(shoulder_to_wrist), 1e-6)
    axis = normalize_vec(shoulder_to_wrist)
    shoulder_to_elbow = sub_vec(reference_elbow, shoulder)
    along = dot_vec(shoulder_to_elbow, axis)
    base = add_vec(shoulder, scale_vec(axis, along))
    bend_direction = sub_vec(reference_elbow, base)
    if length_vec(bend_direction) < 1e-6:
        bend_direction = (0.0, 1.0, 0.0)
    bend_direction = normalize_vec(bend_direction)

    upper_len = max(length_vec(sub_vec(reference_elbow, shoulder)), 1e-6)
    forearm_len = max(length_vec(sub_vec(wrist, reference_elbow)), 1e-6)
    bend_height = deg(bend_deg) / (1.0 / upper_len + 1.0 / forearm_len)
    return add_vec(base, scale_vec(bend_direction, bend_height))


def forearm_marker_direction(elbow, wrist, seed_direction, supination_angle):
    forearm_axis = normalize_vec(sub_vec(wrist, elbow))
    marker = sub_vec(seed_direction, scale_vec(forearm_axis, dot_vec(seed_direction, forearm_axis)))
    if length_vec(marker) < 1e-6:
        marker = cross_vec(forearm_axis, (0.0, 0.0, 1.0))
    if length_vec(marker) < 1e-6:
        marker = cross_vec(forearm_axis, (1.0, 0.0, 0.0))
    return normalize_vec(rotate_vec(marker, forearm_axis, supination_angle))


def shoulder_z_for_reach(rotated_shoulder, target_wrist, base_reach):
    dx = rotated_shoulder[0] - target_wrist[0]
    dy = rotated_shoulder[1] - target_wrist[1]
    horizontal_sq = dx * dx + dy * dy
    vertical = math.sqrt(max(base_reach * base_reach - horizontal_sq, 0.0))
    return target_wrist[2] + vertical


def apply_wrist_x_track(base, moved):
    target_wrists = {}
    required_lifts = []
    for side in ("left", "right"):
        wrist_key = f"{side}_wrist"
        shoulder_key = f"{side}_shoulder"
        target_wrist = (
            moved[wrist_key][0],
            base[wrist_key][1],
            base[wrist_key][2],
        )
        base_reach = length_vec(sub_vec(base[wrist_key], base[shoulder_key]))
        target_wrists[side] = target_wrist
        desired_z = shoulder_z_for_reach(
            moved[shoulder_key], target_wrist, base_reach
        )
        required_lifts.append(desired_z - moved[shoulder_key][2])

    moved["left_wrist"] = target_wrists["left"]
    moved["right_wrist"] = target_wrists["right"]
    chest_lift = max(0.0, max(required_lifts))

    for key in ("left_shoulder", "right_shoulder", "chest_center"):
        moved[key] = (
            moved[key][0],
            moved[key][1],
            moved[key][2] + chest_lift,
        )


def apply_negative_x_shaft_takeaway(base, moved, progress):
    left_reach = length_vec(sub_vec(base["left_wrist"], base["left_shoulder"]))
    hand_spacing = length_vec(sub_vec(base["left_wrist"], base["right_wrist"]))
    club_shaft_len = length_vec(sub_vec(base["clubhead"], grip_center(base)))

    left_drop = min(LEFT_ARM_TAKEAWAY_VERTICAL_DROP, left_reach * 0.88)
    y_offset = LEFT_ARM_TAKEAWAY_Y_OFFSET
    horizontal_sq = max(left_reach * left_reach - left_drop * left_drop - y_offset * y_offset, 0.0)
    x_reach = math.sqrt(horizontal_sq)

    target_left_wrist = (
        moved["left_shoulder"][0] - x_reach,
        moved["left_shoulder"][1] + y_offset,
        moved["left_shoulder"][2] - left_drop,
    )
    target_right_wrist = (
        target_left_wrist[0] - hand_spacing,
        target_left_wrist[1],
        target_left_wrist[2],
    )
    target_grip = tuple(
        0.5 * (target_left_wrist[i] + target_right_wrist[i]) for i in range(3)
    )
    target_clubhead = (
        target_grip[0] - club_shaft_len,
        target_grip[1],
        target_grip[2],
    )

    moved["left_wrist"] = lerp_vec(moved["left_wrist"], target_left_wrist, progress)
    moved["right_wrist"] = lerp_vec(moved["right_wrist"], target_right_wrist, progress)
    moved["clubhead"] = lerp_vec(moved["clubhead"], target_clubhead, progress)


def pose_points(hand=DEFAULT_HAND, progress=0.0):
    base = setup_points(hand)
    progress = smoothstep(progress)
    chest_axis_bottom, chest_axis_top = chest_axis_points(base["chest_center"])
    if ROTATE_CHEST_AROUND_VERTICAL_FOR_TAKEAWAY:
        chest_axis = (0.0, 0.0, 1.0)
    else:
        chest_axis = normalize_vec(sub_vec(chest_axis_top, chest_axis_bottom))
    turn_sign = -1.0 if hand == "right" else 1.0
    chest_angle = turn_sign * deg(CHEST_TAKEAWAY_TURN_DEG) * progress

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

    if KEEP_WRISTS_ON_X_TRACK:
        apply_wrist_x_track(base, moved)
    if TARGET_NEGATIVE_X_SHAFT_TAKEAWAY:
        apply_negative_x_shaft_takeaway(base, moved, progress)

    left_bend_deg = ELBOW_VISUAL_BEND_DEG + (
        LEFT_TAKEAWAY_ELBOW_BEND_DEG - ELBOW_VISUAL_BEND_DEG
    ) * progress
    right_bend_deg = ELBOW_VISUAL_BEND_DEG + (
        RIGHT_TAKEAWAY_ELBOW_BEND_DEG - ELBOW_VISUAL_BEND_DEG
    ) * progress
    moved["left_elbow"] = elbow_with_bend(
        moved["left_shoulder"],
        moved["left_wrist"],
        moved["left_elbow"],
        left_bend_deg,
    )
    moved["right_elbow"] = elbow_with_bend(
        moved["right_shoulder"],
        moved["right_wrist"],
        moved["right_elbow"],
        right_bend_deg,
    )
    supination_sign = -1.0 if hand == "right" else 1.0
    moved["right_forearm_supination_dir"] = forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        supination_sign * deg(RIGHT_FOREARM_SUPINATION_DEG) * progress,
    )

    if not TARGET_NEGATIVE_X_SHAFT_TAKEAWAY:
        shaft_top = grip_center(moved)
        shaft_vec = sub_vec(moved["clubhead"], shaft_top)
        grip_axis = normalize_vec(sub_vec(moved["left_wrist"], moved["right_wrist"]))
        if length_vec(grip_axis) < 1e-6:
            grip_axis = (1.0, 0.0, 0.0)
        extension_axis = grip_axis
        deviation_axis = normalize_vec(cross_vec(extension_axis, shaft_vec))
        if length_vec(deviation_axis) < 1e-6:
            deviation_axis = (0.0, 0.0, 1.0)

        wrist_extension_set = 0.5 * (RIGHT_WRIST_EXTENSION_DEG + LEFT_WRIST_FLEXION_DEG)
        wrist_deviation_set = 0.5 * (RIGHT_WRIST_ULNAR_DEVIATION_DEG + LEFT_WRIST_RADIAL_DEVIATION_DEG)
        extension_angle = turn_sign * deg(wrist_extension_set) * progress
        deviation_angle = deg(wrist_deviation_set) * progress
        shaft_vec = rotate_vec(shaft_vec, extension_axis, extension_angle)
        shaft_vec = rotate_vec(shaft_vec, deviation_axis, deviation_angle)
        moved["clubhead"] = add_vec(shaft_top, shaft_vec)

    moved["progress"] = progress
    return moved


def set_mocap_pose(model, data, body_name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body {body_name} is not a mocap body")
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = quat


def set_mocap_segment(model, data, body_name, start, end):
    midpoint = tuple(0.5 * (start[i] + end[i]) for i in range(3))
    set_mocap_pose(model, data, body_name, midpoint, segment_quat(start, end))


def apply_pose(model, data, points, club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    chest_axis_bottom, chest_axis_top = chest_axis_points(points["chest_center"])
    shaft_top = grip_center(points)

    set_mocap_segment(model, data, "chest_shoulder_bar_body", points["right_shoulder"], points["left_shoulder"])
    set_mocap_segment(model, data, "chest_rotation_axis_body", chest_axis_bottom, chest_axis_top)
    set_mocap_segment(model, data, "left_upper_arm_body", points["left_shoulder"], points["left_elbow"])
    set_mocap_segment(model, data, "left_forearm_body", points["left_elbow"], points["left_wrist"])
    set_mocap_segment(model, data, "right_upper_arm_body", points["right_shoulder"], points["right_elbow"])
    set_mocap_segment(model, data, "right_forearm_body", points["right_elbow"], points["right_wrist"])
    set_mocap_segment(
        model,
        data,
        "left_elbow_turn_marker_body",
        points["left_elbow"],
        add_vec(points["left_elbow"], scale_vec(points["left_elbow_turn_dir"], 0.15)),
    )
    set_mocap_segment(
        model,
        data,
        "right_elbow_turn_marker_body",
        points["right_elbow"],
        add_vec(points["right_elbow"], scale_vec(points["right_elbow_turn_dir"], 0.15)),
    )
    right_forearm_mid = tuple(
        0.5 * (points["right_elbow"][i] + points["right_wrist"][i]) for i in range(3)
    )
    set_mocap_segment(
        model,
        data,
        "right_forearm_supination_marker_body",
        right_forearm_mid,
        add_vec(right_forearm_mid, scale_vec(points["right_forearm_supination_dir"], 0.13)),
    )
    club_pos, club_quat = club_assembly_pose(points, club_name, hand)
    set_mocap_pose(model, data, "club_assembly_body", club_pos, club_quat)

    for body_name, point_name in [
        ("sternum_marker_body", "chest_center"),
        ("left_shoulder_marker_body", "left_shoulder"),
        ("right_shoulder_marker_body", "right_shoulder"),
        ("left_elbow_marker_body", "left_elbow"),
        ("right_elbow_marker_body", "right_elbow"),
        ("left_wrist_marker_body", "left_wrist"),
        ("right_wrist_marker_body", "right_wrist"),
    ]:
        set_mocap_pose(model, data, body_name, points[point_name])
    set_mocap_pose(model, data, "clubhead_marker_body", visual_face_center(points, hand))

    mujoco.mj_forward(model, data)


def make_takeaway_model(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    return mujoco.MjModel.from_xml_string(build_takeaway_xml(club_name, hand))


def print_takeaway_report(hand, club_name=DEFAULT_CLUB):
    print("Two-arm chest takeaway visual")
    preset = get_club_preset(club_name)
    address = pose_points(hand, 0.0)
    parts = club_visual_parts(address, club_name, hand)
    shaft_axis = normalize_vec(sub_vec(parts["heel"], parts["grip"]))
    sole_lie = angle_between_deg(shaft_axis, parts["toe_axis"])
    loft_angle = loft_angle_in_xz_deg(parts["top_axis"])
    wrist_supination_roll = math.degrees(takeaway_wrist_supination_roll_angle(club_name, hand))
    takeaway = pose_points(hand, 1.0)
    takeaway_parts = club_visual_parts(takeaway, club_name, hand)
    spine_bottom, spine_top = chest_axis_points(takeaway["chest_center"])
    spine_axis = normalize_vec(sub_vec(spine_top, spine_bottom))
    spine_tilt = tilt_from_vertical_deg(spine_axis)
    sole_tilt = tilt_from_vertical_deg(takeaway_parts["toe_axis"])
    print("Club:", preset["label"])
    print("Club loft degrees:", preset["loft_deg"])
    print("Club lie degrees:", CLUB_LIE_DEG)
    print("Measured setup lie degrees:", round(sole_lie, 3))
    print("Measured setup loft degrees:", round(loft_angle, 3))
    print("Displayed ball x offset:", DISPLAY_BALL_OFFSET_X)
    print("Clubhead rigidly locked to shaft:", True)
    print("Wrist supination matches sole tilt:", MATCH_SOLE_TILT_WITH_WRIST_SUPINATION)
    print("Calculated wrist supination roll degrees:", round(wrist_supination_roll, 3))
    print("Takeaway spine tilt degrees:", round(spine_tilt, 3))
    print("Takeaway sole tilt degrees:", round(sole_tilt, 3))
    print("Clubhead attachment:", "separate shaft/head bodies bound under one club assembly")
    print("Address pause seconds:", SETUP_HOLD_SECONDS)
    print("Chest turn degrees:", CHEST_TAKEAWAY_TURN_DEG)
    print("Right wrist extension degrees:", RIGHT_WRIST_EXTENSION_DEG)
    print("Right wrist ulnar deviation degrees:", RIGHT_WRIST_ULNAR_DEVIATION_DEG)
    print("Left wrist flexion degrees:", LEFT_WRIST_FLEXION_DEG)
    print("Left wrist radial deviation degrees:", LEFT_WRIST_RADIAL_DEVIATION_DEG)
    print("Left elbow flex degrees at pause:", LEFT_TAKEAWAY_ELBOW_BEND_DEG)
    print("Right elbow flex degrees at pause:", RIGHT_TAKEAWAY_ELBOW_BEND_DEG)
    print("Right forearm supination degrees:", RIGHT_FOREARM_SUPINATION_DEG)
    print("Elbow turn marker degrees:", ELBOW_TURN_TOWARD_BALL_DEG)
    print("Wrists stay on x-track:", KEEP_WRISTS_ON_X_TRACK)
    print("Chest takeaway rotates around vertical:", ROTATE_CHEST_AROUND_VERTICAL_FOR_TAKEAWAY)
    print("Target shaft along negative x:", TARGET_NEGATIVE_X_SHAFT_TAKEAWAY)
    print("Address face center:", [round(v, 4) for v in visual_face_center(address, hand)])
    print("Displayed ball center:", [round(ball_x + DISPLAY_BALL_OFFSET_X, 4), 0.0, round(ball_z, 4)])
    print("Takeaway face center:", [round(v, 4) for v in visual_face_center(takeaway, hand)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", default=DEFAULT_CLUB, choices=sorted(CLUB_PRESETS))
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["right", "left"])
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    print_takeaway_report(args.hand, args.club)
    model = make_takeaway_model(args.club, args.hand)
    data = mujoco.MjData(model)
    apply_pose(model, data, pose_points(args.hand, 0.0), args.club, args.hand)

    sleep_time = max(0.001, 0.01 / max(args.speed, 0.1))
    setup_hold_steps = max(1, int(SETUP_HOLD_SECONDS / sleep_time))
    step = 0
    total_steps = setup_hold_steps + TAKEAWAY_STEPS + PAUSE_STEPS
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if step < setup_hold_steps:
                progress = 0.0
            elif step < setup_hold_steps + TAKEAWAY_STEPS:
                progress = (step - setup_hold_steps) / TAKEAWAY_STEPS
            else:
                progress = 1.0

            apply_pose(model, data, pose_points(args.hand, progress), args.club, args.hand)
            viewer.sync()
            step = (step + 1) % total_steps
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
