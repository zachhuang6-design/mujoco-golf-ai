"""Animated two-arm chest full-swing visual.

This is a first-pass kinematic sequencing sketch. It reuses the accepted
backswing, then adds a hardcoded transition, downswing, impact, extension, and
finish. It is not a physics controller yet.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_core.common import CLUB_PRESETS, get_club_preset
from golf_core import two_arm_chest_backswing as backswing
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


BACKSWING_END = 0.50
TRANSITION_END = 0.62
DELIVERY_END = 0.76
IMPACT_END = 0.84
EXTENSION_END = 0.91

SETUP_HOLD_SECONDS = 3.0
TAKEAWAY_PAUSE_SECONDS = 7.0
IMPACT_PAUSE_SECONDS = 7.0
SWING_STEPS = 620
PAUSE_STEPS = 10_000

JOINT_CLASSIFICATION = [
    (
        "chest_spine_axis",
        "hinge",
        "spine/chest turn around the tilted address axis",
        "future MJCF hinge joint on the torso body",
    ),
    (
        "left_shoulder",
        "ball",
        "lead shoulder socket with broad rotational freedom",
        "future MJCF ball joint on the lead upper-arm body",
    ),
    (
        "right_shoulder",
        "ball",
        "trail shoulder socket with broad rotational freedom",
        "future MJCF ball joint on the trail upper-arm body",
    ),
    (
        "left_elbow_flexion",
        "hinge",
        "lead elbow stays almost straight during this swing",
        "future MJCF hinge limited near 0 degrees of flexion",
    ),
    (
        "right_elbow_flexion",
        "hinge",
        "trail elbow folds through the backswing and releases through impact",
        "future MJCF hinge limited from straight to human flexion range",
    ),
    (
        "forearm_rotation",
        "hinge",
        "pronation/supination about the forearm long axis",
        "future MJCF axial hinge between elbow and wrist frames",
    ),
    (
        "wrist_set",
        "hinge pair",
        "wrist flex/extension plus radial/ulnar deviation",
        "future MJCF two hinge joints between forearm and hand/grip",
    ),
    (
        "club_grip",
        "equality/tendon-like constraint",
        "both wrists remain attached to the rigid club shaft",
        "future MJCF equality constraints or tendon/grip body attachment",
    ),
]

LEGAL_LIMITS = {
    "lead_elbow_flex_max_deg": 8.0,
    "trail_elbow_flex_max_deg": 135.0,
    "preferred_trail_elbow_flex_max_deg": 100.0,
    "wrist_to_shaft_max_m": 0.01,
    "segment_length_warning_m": 0.03,
    "visual_shaft_length_warning_m": 0.03,
    "frame_jump_warning_m": 0.09,
}


def smootherstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp_vec(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


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


def custom_chest_axis(chest_center, spine_lean_deg=0.0, hip_bump_x=0.0):
    axis_bottom, axis_top = chest_axis_points(chest_center)
    base_axis = sub_vec(axis_top, axis_bottom)
    leaned_axis = rotate_vec(base_axis, (0.0, 1.0, 0.0), -deg(spine_lean_deg))
    half_axis = scale_vec(leaned_axis, 0.5)
    return (
        add_vec(sub_vec(chest_center, half_axis), (hip_bump_x, 0.0, 0.0)),
        add_vec(add_vec(chest_center, half_axis), (-0.45 * hip_bump_x, 0.0, 0.0)),
    )


def pose_chest_axis(points):
    if "chest_axis_bottom" in points and "chest_axis_top" in points:
        return points["chest_axis_bottom"], points["chest_axis_top"]
    return chest_axis_points(points["chest_center"])


def grip_center(points):
    return takeaway.grip_center(points)


def elbow_angle_deg(shoulder, elbow, wrist):
    upper = normalize_vec(sub_vec(shoulder, elbow))
    forearm = normalize_vec(sub_vec(wrist, elbow))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot_vec(upper, forearm)))))


def elbow_flex_deg(shoulder, elbow, wrist):
    return 180.0 - elbow_angle_deg(shoulder, elbow, wrist)


def arm_lengths(base, side):
    return (
        length_vec(sub_vec(base[f"{side}_elbow"], base[f"{side}_shoulder"])),
        length_vec(sub_vec(base[f"{side}_wrist"], base[f"{side}_elbow"])),
    )


def shaft_length(base):
    return length_vec(sub_vec(base["clubhead"], grip_center(base)))


def point_to_line_distance(point, line_point, line_axis):
    line_axis = normalize_vec(line_axis)
    offset = sub_vec(point, line_point)
    projection = scale_vec(line_axis, dot_vec(offset, line_axis))
    return length_vec(sub_vec(offset, projection))


def visual_shaft_length(points, club_name, hand):
    parts = takeaway.club_visual_parts(points, club_name, hand)
    return length_vec(sub_vec(parts["heel"], parts["grip"]))


def wrist_to_visual_shaft_distance(points, side, club_name, hand):
    parts = takeaway.club_visual_parts(points, club_name, hand)
    shaft_axis = sub_vec(parts["heel"], parts["grip"])
    return point_to_line_distance(points[f"{side}_wrist"], parts["grip"], shaft_axis)


def pose_legality_metrics(points, base, club_name, hand, previous=None):
    metrics = {
        "left_elbow_flex_deg": elbow_flex_deg(
            points["left_shoulder"],
            points["left_elbow"],
            points["left_wrist"],
        ),
        "right_elbow_flex_deg": elbow_flex_deg(
            points["right_shoulder"],
            points["right_elbow"],
            points["right_wrist"],
        ),
        "left_wrist_to_shaft_m": wrist_to_visual_shaft_distance(points, "left", club_name, hand),
        "right_wrist_to_shaft_m": wrist_to_visual_shaft_distance(points, "right", club_name, hand),
        "visual_shaft_delta_m": abs(
            visual_shaft_length(points, club_name, hand)
            - visual_shaft_length(base, club_name, hand)
        ),
    }

    for side in ("left", "right"):
        upper_len, forearm_len = arm_lengths(base, side)
        metrics[f"{side}_upper_delta_m"] = abs(
            length_vec(sub_vec(points[f"{side}_elbow"], points[f"{side}_shoulder"]))
            - upper_len
        )
        metrics[f"{side}_forearm_delta_m"] = abs(
            length_vec(sub_vec(points[f"{side}_wrist"], points[f"{side}_elbow"]))
            - forearm_len
        )

    if previous is None:
        metrics["max_point_jump_m"] = 0.0
    else:
        keys = (
            "chest_center",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "clubhead",
        )
        metrics["max_point_jump_m"] = max(
            length_vec(sub_vec(points[key], previous[key]))
            for key in keys
        )

    return metrics


def validate_full_swing_legality(hand=DEFAULT_HAND, club_name=DEFAULT_CLUB, samples=801):
    base = setup_points(hand)
    max_metrics = {}
    previous = None

    for index in range(samples):
        progress = index / max(samples - 1, 1)
        points = pose_points(hand, progress, club_name)
        metrics = pose_legality_metrics(points, base, club_name, hand, previous)
        for key, value in metrics.items():
            max_metrics[key] = max(max_metrics.get(key, 0.0), value)
        previous = points

    warnings = []
    if max_metrics["left_elbow_flex_deg"] > LEGAL_LIMITS["lead_elbow_flex_max_deg"]:
        warnings.append("lead elbow bends beyond the current straight-lead-arm limit")
    if max_metrics["right_elbow_flex_deg"] > LEGAL_LIMITS["trail_elbow_flex_max_deg"]:
        warnings.append("trail elbow exceeds the human flexion limit")
    if max_metrics["right_elbow_flex_deg"] > LEGAL_LIMITS["preferred_trail_elbow_flex_max_deg"]:
        warnings.append("trail elbow is legal but above the preferred golf-swing range")
    if max(max_metrics["left_wrist_to_shaft_m"], max_metrics["right_wrist_to_shaft_m"]) > LEGAL_LIMITS["wrist_to_shaft_max_m"]:
        warnings.append("one or both wrists drift away from the visual shaft line")
    if max_metrics["visual_shaft_delta_m"] > LEGAL_LIMITS["visual_shaft_length_warning_m"]:
        warnings.append("visual shaft/heel length changes because this is still mocap-driven")
    for key in (
        "left_upper_delta_m",
        "left_forearm_delta_m",
        "right_upper_delta_m",
        "right_forearm_delta_m",
    ):
        if max_metrics[key] > LEGAL_LIMITS["segment_length_warning_m"]:
            warnings.append(f"{key} changes, so the kinematic marker path is not yet a true rigid limb")
    if max_metrics["max_point_jump_m"] > LEGAL_LIMITS["frame_jump_warning_m"]:
        warnings.append("frame-to-frame motion has a large discontinuity")

    return {
        "legal_enough_for_visual": len(warnings) == 0,
        "warnings": warnings,
        "metrics": max_metrics,
    }


def hand_offsets_on_shaft(base):
    grip = grip_center(base)
    shaft_axis = normalize_vec(sub_vec(base["clubhead"], grip))
    return {
        side: dot_vec(sub_vec(base[f"{side}_wrist"], grip), shaft_axis)
        for side in ("left", "right")
    }


def freeze_club_frame(points, club_name, hand):
    parts = takeaway.club_visual_parts(points, club_name, hand)
    points["club_face_normal"] = parts["face_normal"]
    points["club_toe_axis"] = parts["toe_axis"]
    points["club_top_axis"] = parts["top_axis"]
    points["club_roll_angle"] = 0.0
    return points


def square_clubface_to_target(points, club_name, hand):
    parts = takeaway.club_visual_parts(points, club_name, hand)
    shaft_axis = normalize_vec(sub_vec(parts["heel"], parts["grip"]))
    target_face = takeaway.face_normal_for_loft(club_name)

    best_angle = 0.0
    best_score = float("inf")
    for index in range(721):
        angle = -math.pi + (2.0 * math.pi * index / 720.0)
        face = rotate_vec(points["club_face_normal"], shaft_axis, angle)
        horizontal_angle = math.atan2(face[1], face[0])
        loft_error = face[2] - target_face[2]
        wrong_way_penalty = max(0.0, -face[0]) * 10.0
        score = 4.0 * abs(horizontal_angle) + abs(loft_error) + wrong_way_penalty
        if score < best_score:
            best_score = score
            best_angle = angle

    points["club_face_normal"] = rotate_vec(points["club_face_normal"], shaft_axis, best_angle)
    points["club_toe_axis"] = rotate_vec(points["club_toe_axis"], shaft_axis, best_angle)
    points["club_top_axis"] = rotate_vec(points["club_top_axis"], shaft_axis, best_angle)
    points["club_roll_angle"] = 0.0
    return points


def body_from_address(
    base,
    hand,
    chest_turn_deg,
    shift=(0.0, 0.0, 0.0),
    spine_lean_deg=0.0,
    hip_bump_x=0.0,
):
    chest_axis_bottom, chest_axis_top = chest_axis_points(base["chest_center"])
    chest_axis = normalize_vec(sub_vec(chest_axis_top, chest_axis_bottom))
    turn = deg(chest_turn_deg if hand == "right" else -chest_turn_deg)
    moved = {"chest_center": add_vec(base["chest_center"], shift)}
    shoulder_counter = (-0.35 * hip_bump_x - 0.0025 * spine_lean_deg, 0.0, 0.0)

    for key in ("left_shoulder", "right_shoulder"):
        moved[key] = add_vec(
            add_vec(rotate_point(base[key], base["chest_center"], chest_axis, turn), shoulder_counter),
            shift,
        )

    moved["left_elbow_turn_dir"] = rotate_vec(base["left_elbow_turn_dir"], chest_axis, turn)
    moved["right_elbow_turn_dir"] = rotate_vec(base["right_elbow_turn_dir"], chest_axis, turn)
    if spine_lean_deg or hip_bump_x:
        moved["chest_axis_bottom"], moved["chest_axis_top"] = custom_chest_axis(
            moved["chest_center"],
            spine_lean_deg,
            hip_bump_x,
        )
    return moved


def apply_grip_and_club(base, moved, grip, clubhead):
    shaft_axis = normalize_vec(sub_vec(clubhead, grip))
    offsets = hand_offsets_on_shaft(base)
    moved["left_wrist"] = add_vec(grip, scale_vec(shaft_axis, offsets["left"]))
    moved["right_wrist"] = add_vec(grip, scale_vec(shaft_axis, offsets["right"]))
    moved["clubhead"] = clubhead
    return moved


def legalize_lead_arm_and_club(base, moved, shaft_axis):
    shaft_axis = normalize_vec(shaft_axis)
    offsets = hand_offsets_on_shaft(base)
    left_upper, left_forearm = arm_lengths(base, "left")
    left_reach = left_upper + left_forearm - 1e-6
    shoulder_to_wrist = sub_vec(moved["left_wrist"], moved["left_shoulder"])
    if length_vec(shoulder_to_wrist) < 1e-6:
        shoulder_to_wrist = sub_vec(base["left_wrist"], base["left_shoulder"])

    legal_left_wrist = add_vec(
        moved["left_shoulder"],
        scale_vec(normalize_vec(shoulder_to_wrist), left_reach),
    )
    legal_grip = sub_vec(legal_left_wrist, scale_vec(shaft_axis, offsets["left"]))

    moved["left_wrist"] = add_vec(legal_grip, scale_vec(shaft_axis, offsets["left"]))
    moved["right_wrist"] = add_vec(legal_grip, scale_vec(shaft_axis, offsets["right"]))
    moved["clubhead"] = add_vec(legal_grip, scale_vec(shaft_axis, shaft_length(base)))
    return moved


def solve_elbows(base, moved, left_straight=True, right_reference=None):
    left_upper, left_forearm = arm_lengths(base, "left")
    right_upper, right_forearm = arm_lengths(base, "right")

    if left_straight:
        left_axis = normalize_vec(sub_vec(moved["left_wrist"], moved["left_shoulder"]))
        moved["left_elbow"] = add_vec(moved["left_shoulder"], scale_vec(left_axis, left_upper))
    else:
        moved["left_elbow"] = backswing.elbow_from_lengths(
            moved["left_shoulder"],
            moved["left_wrist"],
            moved.get("left_elbow", base["left_elbow"]),
            left_upper,
            left_forearm,
        )

    moved["right_elbow"] = backswing.elbow_from_lengths(
        moved["right_shoulder"],
        moved["right_wrist"],
        right_reference or moved.get("right_elbow", base["right_elbow"]),
        right_upper,
        right_forearm,
    )
    return moved


def clubhead_from_grip_axis(base, grip, axis):
    return add_vec(grip, scale_vec(normalize_vec(axis), shaft_length(base)))


def sequenced_keyframe(
    base,
    hand,
    club_name,
    chest_turn_deg,
    grip,
    shaft_axis,
    shift=(0.0, 0.0, 0.0),
    spine_lean_deg=0.0,
    hip_bump_x=0.0,
    right_supination_deg=20.0,
):
    moved = body_from_address(base, hand, chest_turn_deg, shift, spine_lean_deg, hip_bump_x)
    clubhead = clubhead_from_grip_axis(base, grip, shaft_axis)
    apply_grip_and_club(base, moved, grip, clubhead)
    legalize_lead_arm_and_club(base, moved, shaft_axis)
    solve_elbows(base, moved, left_straight=True)
    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        -deg(right_supination_deg) if hand == "right" else deg(right_supination_deg),
    )
    moved["progress"] = 0.0
    return freeze_club_frame(moved, club_name, hand)


def mirrored_backswing_keyframe(
    base,
    hand,
    club_name,
    backswing_progress,
    spine_lean_deg=0.0,
    hip_bump_x=0.0,
):
    moved = dict(backswing.pose_points(hand, backswing_progress, club_name))
    if spine_lean_deg or hip_bump_x:
        moved["chest_axis_bottom"], moved["chest_axis_top"] = custom_chest_axis(
            moved["chest_center"],
            spine_lean_deg,
            hip_bump_x,
        )
    shaft_axis = normalize_vec(sub_vec(moved["clubhead"], grip_center(moved)))
    legalize_lead_arm_and_club(base, moved, shaft_axis)
    solve_elbows(base, moved, left_straight=True)
    return freeze_club_frame(moved, club_name, hand)


def impact_keyframe(base, hand, club_name):
    moved = body_from_address(
        base,
        hand,
        34.0,
        shift=(0.065, 0.0, 0.0),
        spine_lean_deg=11.0,
        hip_bump_x=0.06,
    )
    clubhead = base["clubhead"]
    rough_grip = (clubhead[0] + 0.20, 0.28, 1.04)
    shaft_axis = normalize_vec(sub_vec(clubhead, rough_grip))
    grip = sub_vec(clubhead, scale_vec(shaft_axis, shaft_length(base)))
    apply_grip_and_club(base, moved, grip, clubhead)
    legalize_lead_arm_and_club(base, moved, shaft_axis)
    solve_elbows(base, moved, left_straight=True)
    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        -deg(5.0) if hand == "right" else deg(5.0),
    )
    moved["progress"] = 0.0
    freeze_club_frame(moved, club_name, hand)
    return square_clubface_to_target(moved, club_name, hand)


def make_keyframes(hand=DEFAULT_HAND, club_name=DEFAULT_CLUB):
    base = setup_points(hand)
    top = dict(backswing.pose_points(hand, 1.0, club_name))
    top_shaft_axis = normalize_vec(sub_vec(top["clubhead"], grip_center(top)))
    legalize_lead_arm_and_club(base, top, top_shaft_axis)
    solve_elbows(base, top, left_straight=True)
    top = freeze_club_frame(top, club_name, hand)

    transition = mirrored_backswing_keyframe(
        base,
        hand,
        club_name,
        backswing_progress=0.78,
        spine_lean_deg=5.0,
        hip_bump_x=0.035,
    )
    delivery = mirrored_backswing_keyframe(
        base,
        hand,
        club_name,
        backswing_progress=backswing.TAKEAWAY_PHASE,
        spine_lean_deg=8.0,
        hip_bump_x=0.045,
    )
    impact = impact_keyframe(base, hand, club_name)
    extension = sequenced_keyframe(
        base,
        hand,
        club_name,
        chest_turn_deg=62.0,
        grip=(0.42, 0.38, 1.28),
        shaft_axis=(0.82, -0.10, 0.56),
        shift=(0.055, 0.0, 0.0),
    )
    finish = sequenced_keyframe(
        base,
        hand,
        club_name,
        chest_turn_deg=118.0,
        grip=(0.45, 0.75, 1.75),
        shaft_axis=(-0.72, 0.16, 0.68),
        shift=(0.06, 0.0, 0.0),
    )
    return [top, transition, delivery, impact, extension, finish]


def transport_club_frame(start, moved, club_name, hand):
    start_parts = takeaway.club_visual_parts(start, club_name, hand)
    moved_grip = grip_center(moved)
    moved_shaft = normalize_vec(sub_vec(moved["clubhead"], moved_grip))
    start_shaft = normalize_vec(sub_vec(start_parts["heel"], start_parts["grip"]))
    axis, angle = takeaway.rotation_between(start_shaft, moved_shaft)
    moved["club_face_normal"] = rotate_vec(start_parts["face_normal"], axis, angle)
    moved["club_toe_axis"] = rotate_vec(start_parts["toe_axis"], axis, angle)
    moved["club_top_axis"] = rotate_vec(start_parts["top_axis"], axis, angle)
    moved["club_roll_angle"] = 0.0


def orthonormalize_club_axes(face_normal, toe_axis):
    face_normal = normalize_vec(face_normal)
    toe_axis = sub_vec(toe_axis, scale_vec(face_normal, dot_vec(toe_axis, face_normal)))
    if length_vec(toe_axis) < 1e-6:
        toe_axis = cross_vec(face_normal, (0.0, 0.0, 1.0))
    if length_vec(toe_axis) < 1e-6:
        toe_axis = cross_vec(face_normal, (0.0, 1.0, 0.0))
    toe_axis = normalize_vec(toe_axis)
    top_axis = normalize_vec(cross_vec(face_normal, toe_axis))
    return face_normal, toe_axis, top_axis


def blend_club_frame(start, target, moved, t, club_name, hand):
    start_parts = takeaway.club_visual_parts(start, club_name, hand)
    target_parts = takeaway.club_visual_parts(target, club_name, hand)
    face = backswing.rotate_direction_toward(
        start_parts["face_normal"],
        target_parts["face_normal"],
        t,
    )
    toe = backswing.rotate_direction_toward(
        start_parts["toe_axis"],
        target_parts["toe_axis"],
        t,
    )
    moved["club_face_normal"], moved["club_toe_axis"], moved["club_top_axis"] = (
        orthonormalize_club_axes(face, toe)
    )
    moved["club_roll_angle"] = 0.0


def blend_pose(start, target, t, base, club_name, hand):
    t = smootherstep(t)
    moved = {
        "chest_center": lerp_vec(start["chest_center"], target["chest_center"], t),
        "left_shoulder": lerp_vec(start["left_shoulder"], target["left_shoulder"], t),
        "right_shoulder": lerp_vec(start["right_shoulder"], target["right_shoulder"], t),
    }
    start_axis_bottom, start_axis_top = pose_chest_axis(start)
    target_axis_bottom, target_axis_top = pose_chest_axis(target)
    moved["chest_axis_bottom"] = lerp_vec(start_axis_bottom, target_axis_bottom, t)
    moved["chest_axis_top"] = lerp_vec(start_axis_top, target_axis_top, t)

    for side in ("left", "right"):
        shoulder_key = f"{side}_shoulder"
        wrist_key = f"{side}_wrist"
        start_arm = sub_vec(start[wrist_key], start[shoulder_key])
        target_arm = sub_vec(target[wrist_key], target[shoulder_key])
        moved[wrist_key] = add_vec(
            moved[shoulder_key],
            backswing.rotate_direction_toward(start_arm, target_arm, t),
        )

    start_grip = grip_center(start)
    target_grip = grip_center(target)
    moved_grip = grip_center(moved)
    start_shaft = normalize_vec(sub_vec(start["clubhead"], start_grip))
    target_shaft = normalize_vec(sub_vec(target["clubhead"], target_grip))
    shaft_axis = normalize_vec(backswing.rotate_direction_toward(start_shaft, target_shaft, t))
    moved["clubhead"] = add_vec(moved_grip, scale_vec(shaft_axis, shaft_length(base)))

    takeaway.lock_wrists_to_shaft(base, moved)
    moved_grip = grip_center(moved)
    moved["clubhead"] = add_vec(moved_grip, scale_vec(shaft_axis, shaft_length(base)))
    legalize_lead_arm_and_club(base, moved, shaft_axis)

    left_upper, _ = arm_lengths(base, "left")
    left_axis = normalize_vec(sub_vec(moved["left_wrist"], moved["left_shoulder"]))
    moved["left_elbow"] = add_vec(moved["left_shoulder"], scale_vec(left_axis, left_upper))
    moved["right_elbow"] = backswing.elbow_from_lengths(
        moved["right_shoulder"],
        moved["right_wrist"],
        lerp_vec(start["right_elbow"], target["right_elbow"], t),
        *arm_lengths(base, "right"),
    )
    moved["left_elbow_turn_dir"] = normalize_vec(
        backswing.rotate_direction_toward(start["left_elbow_turn_dir"], target["left_elbow_turn_dir"], t)
    )
    moved["right_elbow_turn_dir"] = normalize_vec(
        backswing.rotate_direction_toward(start["right_elbow_turn_dir"], target["right_elbow_turn_dir"], t)
    )
    moved["right_forearm_supination_dir"] = takeaway.forearm_marker_direction(
        moved["right_elbow"],
        moved["right_wrist"],
        moved["right_elbow_turn_dir"],
        -deg(12.0) if hand == "right" else deg(12.0),
    )
    moved["progress"] = 0.0
    blend_club_frame(start, target, moved, t, club_name, hand)
    return moved


def phase_blend(progress, start, end):
    return (progress - start) / (end - start)


def takeaway_progress():
    return BACKSWING_END * backswing.TAKEAWAY_PHASE


def legalize_full_swing_pose(points, base, club_name, hand):
    points = dict(points)
    shaft_axis = normalize_vec(sub_vec(points["clubhead"], grip_center(points)))
    legalize_lead_arm_and_club(base, points, shaft_axis)
    solve_elbows(base, points, left_straight=True)
    return freeze_club_frame(points, club_name, hand)


def pose_points(hand=DEFAULT_HAND, progress=0.0, club_name=DEFAULT_CLUB):
    progress = max(0.0, min(1.0, progress))
    base = setup_points(hand)
    if progress <= BACKSWING_END:
        return legalize_full_swing_pose(
            backswing.pose_points(hand, progress / BACKSWING_END, club_name),
            base,
            club_name,
            hand,
        )

    top, transition, delivery, impact, extension, finish = make_keyframes(hand, club_name)

    if progress <= TRANSITION_END:
        return blend_pose(top, transition, phase_blend(progress, BACKSWING_END, TRANSITION_END), base, club_name, hand)
    if progress <= DELIVERY_END:
        return blend_pose(transition, delivery, phase_blend(progress, TRANSITION_END, DELIVERY_END), base, club_name, hand)
    if progress <= IMPACT_END:
        return blend_pose(delivery, impact, phase_blend(progress, DELIVERY_END, IMPACT_END), base, club_name, hand)
    if progress <= EXTENSION_END:
        return blend_pose(impact, extension, phase_blend(progress, IMPACT_END, EXTENSION_END), base, club_name, hand)
    return blend_pose(extension, finish, phase_blend(progress, EXTENSION_END, 1.0), base, club_name, hand)


def print_full_swing_report(hand, club_name=DEFAULT_CLUB):
    preset = get_club_preset(club_name)
    base = setup_points(hand)
    keyframes = make_keyframes(hand, club_name)
    labels = ["top", "transition", "delivery", "impact", "extension", "finish"]
    legality = validate_full_swing_legality(hand, club_name)
    metrics = legality["metrics"]

    print("Two-arm chest full-swing visual")
    print("Club:", preset["label"])
    print("Current enforcement mode:", "kinematic mocap visual with MuJoCo-style legality checks")
    print("Physical joint enforcement:", "not active yet; requires a nested MJCF jointed arm conversion")
    print("MuJoCo-style joint classification:")
    for name, joint_type, purpose, future_use in JOINT_CLASSIFICATION:
        print(" ", name, "|", joint_type, "|", purpose, "|", future_use)
    print("Backswing end share:", BACKSWING_END)
    print("Transition end share:", TRANSITION_END)
    print("Delivery end share:", DELIVERY_END)
    print("Impact end share:", IMPACT_END)
    print("Extension end share:", EXTENSION_END)
    print("Swing steps:", SWING_STEPS)
    for label, points in zip(labels, keyframes):
        grip = grip_center(points)
        print(
            label,
            "left_flex",
            round(elbow_flex_deg(points["left_shoulder"], points["left_elbow"], points["left_wrist"]), 2),
            "right_flex",
            round(elbow_flex_deg(points["right_shoulder"], points["right_elbow"], points["right_wrist"]), 2),
            "grip",
            [round(v, 3) for v in grip],
            "clubhead",
            [round(v, 3) for v in points["clubhead"]],
        )
    print("Shaft length:", round(shaft_length(base), 4))
    print("Wrists locked onto shaft line:", True)
    print("Address pause seconds:", SETUP_HOLD_SECONDS)
    print("Takeaway pause seconds:", TAKEAWAY_PAUSE_SECONDS)
    print("Impact pause seconds:", IMPACT_PAUSE_SECONDS)
    print("Legality check:", "PASS" if legality["legal_enough_for_visual"] else "WARNING")
    print(" Max left elbow flex deg:", round(metrics["left_elbow_flex_deg"], 3))
    print(" Max right elbow flex deg:", round(metrics["right_elbow_flex_deg"], 3))
    print(" Max wrist-to-visual-shaft m:", round(max(metrics["left_wrist_to_shaft_m"], metrics["right_wrist_to_shaft_m"]), 5))
    print(" Max visual shaft-length delta m:", round(metrics["visual_shaft_delta_m"], 5))
    print(" Max left upper/forearm delta m:", round(metrics["left_upper_delta_m"], 5), round(metrics["left_forearm_delta_m"], 5))
    print(" Max right upper/forearm delta m:", round(metrics["right_upper_delta_m"], 5), round(metrics["right_forearm_delta_m"], 5))
    print(" Max frame jump m:", round(metrics["max_point_jump_m"], 5))
    if legality["warnings"]:
        print(" Legality warnings:")
        for warning in legality["warnings"]:
            print("  -", warning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", default=DEFAULT_CLUB, choices=sorted(CLUB_PRESETS))
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=["right", "left"])
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    print_full_swing_report(args.hand, args.club)
    model = takeaway.make_takeaway_model(args.club, args.hand)
    data = mujoco.MjData(model)
    takeaway.apply_pose(model, data, pose_points(args.hand, 0.0, args.club), args.club, args.hand)

    sleep_time = max(0.001, 0.01 / max(args.speed, 0.1))
    setup_hold_steps = max(1, int(SETUP_HOLD_SECONDS / sleep_time))
    takeaway_pause_steps = max(1, int(TAKEAWAY_PAUSE_SECONDS / sleep_time))
    impact_pause_steps = max(1, int(IMPACT_PAUSE_SECONDS / sleep_time))
    takeaway_steps = max(1, int(SWING_STEPS * takeaway_progress()))
    impact_steps = max(takeaway_steps + 1, int(SWING_STEPS * IMPACT_END))
    finish_steps = max(1, SWING_STEPS - impact_steps)
    step = 0
    total_steps = (
        setup_hold_steps
        + takeaway_steps
        + takeaway_pause_steps
        + (impact_steps - takeaway_steps)
        + impact_pause_steps
        + finish_steps
        + PAUSE_STEPS
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if step < setup_hold_steps:
                progress = 0.0
            elif step < setup_hold_steps + takeaway_steps:
                progress = (step - setup_hold_steps) / SWING_STEPS
            elif step < setup_hold_steps + takeaway_steps + takeaway_pause_steps:
                progress = takeaway_progress()
            elif step < setup_hold_steps + takeaway_steps + takeaway_pause_steps + (impact_steps - takeaway_steps):
                motion_step = step - setup_hold_steps - takeaway_pause_steps
                progress = motion_step / SWING_STEPS
            elif step < setup_hold_steps + takeaway_steps + takeaway_pause_steps + (impact_steps - takeaway_steps) + impact_pause_steps:
                progress = IMPACT_END
            elif step < setup_hold_steps + takeaway_steps + takeaway_pause_steps + (impact_steps - takeaway_steps) + impact_pause_steps + finish_steps:
                motion_step = step - setup_hold_steps - takeaway_pause_steps - impact_pause_steps
                progress = motion_step / SWING_STEPS
            else:
                progress = 1.0

            takeaway.apply_pose(model, data, pose_points(args.hand, progress, args.club), args.club, args.hand)
            viewer.sync()
            step = (step + 1) % total_steps
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
