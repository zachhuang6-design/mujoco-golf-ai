import argparse
import math
import random
import time

import mujoco

from golf_3joint_common import CLUB_PRESETS, get_club_launch_profile, normalize_club_name
from human_right_arm_biomech_static import (
    CTRL_LIMITS,
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    deg,
    hand_sign,
    make_static_model,
    normalize_hand,
    setup_angles,
)


EPISODE_STEPS = 1200
INVALID_SWING_REWARD = -100000.0
MIN_FORWARD_CLUB_SPEED = 0.50
MIN_FORWARD_BALL_SPEED = 0.20
MIN_POST_IMPACT_DISTANCE = 0.03
POST_IMPACT_WINDOW_STEPS = 20

JOINT_NAMES = (
    "shoulder_turn",
    "shoulder_lift",
    "shoulder_long_axis_twist",
    "elbow_flex",
    "wrist_cock",
    "wrist_deviation",
    "wrist_roll",
)

KP = (22.0, 28.0, 16.0, 18.0, 12.0, 8.0, 7.0)
KD = (3.1, 3.8, 2.0, 2.4, 1.7, 1.2, 1.0)
CTRL_BY_JOINT = tuple(CTRL_LIMITS[name] for name in JOINT_NAMES)

IMPACT_ELBOW_TARGET = deg(15.0)
IMPACT_ELBOW_TOLERANCE = deg(12.0)
TOP_ELBOW_TARGET = deg(43.0)
TOP_ELBOW_TOLERANCE = deg(12.0)
TOP_WRIST_COCK_TARGET_DEG = 74.0
TOP_WRIST_COCK_TOLERANCE = deg(16.0)
EARLY_SET_FRACTION = 0.58
EARLY_ELBOW_TARGET = deg(24.0)
EARLY_ELBOW_TOLERANCE = deg(15.0)
EARLY_WRIST_COCK_TARGET_DEG = 48.0
EARLY_WRIST_COCK_TOLERANCE = deg(18.0)
EARLY_ROTATION_FRACTION = 0.50
EARLY_SHOULDER_TURN_TARGET_DEG = 42.0
EARLY_SHOULDER_LIFT_TARGET_DEG = 45.0
EARLY_SHOULDER_TWIST_TARGET_DEG = 20.0
EARLY_SHOULDER_TURN_TOLERANCE = deg(22.0)
EARLY_SHOULDER_LIFT_TOLERANCE = deg(24.0)
EARLY_SHOULDER_TWIST_TOLERANCE = deg(22.0)
BACKSWING_SHOULDER_TURN_SET_FRACTION = 0.82
BACKSWING_SHOULDER_LIFT_SET_FRACTION = 0.88
BACKSWING_SHOULDER_TWIST_SET_FRACTION = 0.70
BACKSWING_ELBOW_SET_FRACTION = 0.72
BACKSWING_WRIST_SET_FRACTION = 0.62
OFFLINE_Y_PENALTY = 850.0


def clip(value, low, high):
    return max(low, min(high, value))


def smoothstep(x):
    x = clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def interpolate_pose(start_pose, end_pose, t):
    t = smoothstep(t)
    return tuple(lerp(start_pose[i], end_pose[i], t) for i in range(len(start_pose)))


def angle_score(angle, target, tolerance):
    return max(0.0, 1.0 - abs(angle - target) / tolerance)


def value_score(value, target, tolerance):
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def address_pose(hand=DEFAULT_HAND):
    sign = hand_sign(hand)
    _, _, wrist_cock = setup_angles()
    return (
        0.0,
        sign * deg(7.5),
        0.0,
        0.0,
        sign * deg(wrist_cock),
        0.0,
        0.0,
    )


def default_top_pose(hand=DEFAULT_HAND):
    sign = hand_sign(hand)
    return (
        sign * deg(70.0),
        sign * deg(85.0),
        sign * deg(35.0),
        TOP_ELBOW_TARGET,
        sign * deg(TOP_WRIST_COCK_TARGET_DEG),
        sign * deg(12.0),
        sign * deg(10.0),
    )


def default_impact_pose(hand=DEFAULT_HAND):
    sign = hand_sign(hand)
    return (
        sign * deg(-6.0),
        sign * deg(4.0),
        sign * deg(8.0),
        IMPACT_ELBOW_TARGET,
        sign * deg(-8.0),
        sign * deg(4.0),
        0.0,
    )


def default_finish_pose(hand=DEFAULT_HAND):
    sign = hand_sign(hand)
    return (
        -sign * deg(55.0),
        -sign * deg(55.0),
        -sign * deg(35.0),
        deg(35.0),
        -sign * deg(45.0),
        -sign * deg(10.0),
        -sign * deg(18.0),
    )


def signed_range(hand, low_abs_deg, high_abs_deg):
    sign = hand_sign(hand)
    values = (sign * deg(low_abs_deg), sign * deg(high_abs_deg))
    return min(values), max(values)


def parameter_specs(hand=DEFAULT_HAND):
    return [
        ("top_step", 220.0, 460.0),
        ("top_hold", 45.0, 210.0),
        ("impact_gap", 150.0, 390.0),
        ("finish_gap", 60.0, 280.0),
        ("elbow_lag", 15.0, 145.0),
        ("wrist_lag", 35.0, 190.0),
        ("top_shoulder_turn", *signed_range(hand, 40.0, 105.0)),
        ("top_shoulder_lift", *signed_range(hand, 55.0, 120.0)),
        ("top_shoulder_twist", *signed_range(hand, 5.0, 70.0)),
        ("top_elbow_flex", deg(30.0), deg(45.0)),
        ("top_wrist_cock", *signed_range(hand, 55.0, 80.0)),
        ("top_wrist_deviation", *signed_range(hand, -25.0, 35.0)),
        ("top_wrist_roll", *signed_range(hand, -35.0, 45.0)),
        ("impact_shoulder_turn", *signed_range(hand, -25.0, 20.0)),
        ("impact_shoulder_lift", *signed_range(hand, -18.0, 20.0)),
        ("impact_shoulder_twist", *signed_range(hand, -25.0, 35.0)),
        ("impact_elbow_flex", deg(5.0), deg(30.0)),
        ("impact_wrist_cock", *signed_range(hand, -35.0, 20.0)),
        ("impact_wrist_deviation", *signed_range(hand, -20.0, 25.0)),
        ("impact_wrist_roll", *signed_range(hand, -30.0, 35.0)),
        ("finish_shoulder_turn", *signed_range(hand, -95.0, 20.0)),
        ("finish_shoulder_lift", *signed_range(hand, -95.0, 35.0)),
        ("finish_shoulder_twist", *signed_range(hand, -80.0, 25.0)),
        ("finish_elbow_flex", deg(0.0), deg(45.0)),
        ("finish_wrist_cock", *signed_range(hand, -80.0, 45.0)),
        ("finish_wrist_deviation", *signed_range(hand, -40.0, 35.0)),
        ("finish_wrist_roll", *signed_range(hand, -60.0, 55.0)),
    ]


def initial_mean(hand=DEFAULT_HAND):
    top = default_top_pose(hand)
    impact = default_impact_pose(hand)
    finish = default_finish_pose(hand)
    return [
        320.0,
        115.0,
        230.0,
        130.0,
        50.0,
        85.0,
        *top,
        *impact,
        *finish,
    ]


def initial_std():
    return [
        55.0,
        38.0,
        62.0,
        52.0,
        24.0,
        32.0,
        deg(20.0),
        deg(18.0),
        deg(16.0),
        deg(5.0),
        deg(7.0),
        deg(14.0),
        deg(18.0),
        deg(14.0),
        deg(12.0),
        deg(16.0),
        deg(7.0),
        deg(14.0),
        deg(12.0),
        deg(14.0),
        deg(22.0),
        deg(25.0),
        deg(12.0),
        deg(22.0),
        deg(22.0),
        deg(18.0),
        deg(22.0),
    ]


def min_std():
    return [
        5.0,
        6.0,
        6.0,
        6.0,
        4.0,
        4.0,
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(1.5),
        deg(2.0),
        deg(1.0),
        deg(2.0),
        deg(2.0),
        deg(2.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
        deg(3.0),
    ]


def sample_vector(rng, mean, std, specs):
    return [
        clip(rng.gauss(mean[i], std[i]), specs[i][1], specs[i][2])
        for i in range(len(specs))
    ]


def vector_to_candidate(vector, hand=DEFAULT_HAND):
    top_step = int(round(vector[0]))
    top_hold = int(round(vector[1]))
    down_start_step = top_step + top_hold
    impact_step = int(round(down_start_step + vector[2]))
    finish_step = int(round(impact_step + vector[3]))
    elbow_lag = int(round(vector[4]))
    wrist_lag = int(round(max(vector[5], elbow_lag + 5)))
    wrist_lag = min(wrist_lag, max(5, impact_step - down_start_step - 10))
    elbow_lag = min(elbow_lag, max(0, wrist_lag - 5))

    return {
        "hand": normalize_hand(hand),
        "top_step": top_step,
        "top_hold": top_hold,
        "down_start_step": down_start_step,
        "impact_step": impact_step,
        "finish_step": finish_step,
        "elbow_lag": elbow_lag,
        "wrist_lag": wrist_lag,
        "address_pose": address_pose(hand),
        "top_pose": tuple(vector[6:13]),
        "impact_pose": tuple(vector[13:20]),
        "finish_pose": tuple(vector[20:27]),
    }


def target_angles(candidate, step):
    address = candidate.get("address_pose", address_pose(candidate.get("hand", DEFAULT_HAND)))
    top = candidate["top_pose"]
    impact = candidate["impact_pose"]
    finish = candidate["finish_pose"]
    top_step = candidate["top_step"]
    down_start_step = candidate.get("down_start_step", top_step + candidate.get("top_hold", 0))
    impact_step = candidate["impact_step"]
    finish_step = candidate["finish_step"]

    if step < top_step:
        target = []
        for joint in range(len(JOINT_NAMES)):
            if joint == 0:
                t = step / max(1, top_step * BACKSWING_SHOULDER_TURN_SET_FRACTION)
            elif joint == 1:
                t = step / max(1, top_step * BACKSWING_SHOULDER_LIFT_SET_FRACTION)
            elif joint == 2:
                t = step / max(1, top_step * BACKSWING_SHOULDER_TWIST_SET_FRACTION)
            elif joint == 3:
                t = step / max(1, top_step * BACKSWING_ELBOW_SET_FRACTION)
            elif joint in (4, 5, 6):
                t = step / max(1, top_step * BACKSWING_WRIST_SET_FRACTION)
            else:
                t = step / max(1, top_step)
            target.append(lerp(address[joint], top[joint], smoothstep(t)))
        return tuple(target)

    if step < down_start_step:
        return top

    if step < impact_step:
        target = []
        lags = (
            0,
            0,
            12,
            candidate["elbow_lag"],
            candidate["wrist_lag"],
            candidate["wrist_lag"],
            candidate["wrist_lag"],
        )
        for joint in range(len(JOINT_NAMES)):
            joint_start = down_start_step + lags[joint]
            if step <= joint_start:
                target.append(top[joint])
            else:
                t = (step - joint_start) / max(1, impact_step - joint_start)
                target.append(lerp(top[joint], impact[joint], smoothstep(t)))
        return tuple(target)

    if step < finish_step:
        return interpolate_pose(impact, finish, (step - impact_step) / max(1, finish_step - impact_step))

    return finish


def apply_pd_controls(data, candidate, step):
    target = target_angles(candidate, step)
    ctrl_energy = 0.0
    for i in range(len(JOINT_NAMES)):
        command = KP[i] * (target[i] - data.qpos[i]) - KD[i] * data.qvel[i]
        command = clip(command, -CTRL_BY_JOINT[i], CTRL_BY_JOINT[i])
        data.ctrl[i] = command
        ctrl_energy += command * command
    return ctrl_energy


def contact_includes(data, geom_a, geom_b):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
            return True
    return False


def pose_score(values, target, tolerances):
    score = 0.0
    for i in range(len(values)):
        score += angle_score(values[i], target[i], tolerances[i])
    return score / len(values)


def address_score(data, candidate):
    return pose_score(data.qpos[:7], candidate["address_pose"], (deg(10.0), deg(10.0), deg(12.0), deg(8.0), deg(12.0), deg(12.0), deg(12.0)))


def top_score(data, candidate):
    return pose_score(data.qpos[:7], candidate["top_pose"], (deg(24.0), deg(24.0), deg(28.0), deg(16.0), deg(22.0), deg(24.0), deg(28.0)))


def impact_pose_score(data, candidate):
    return pose_score(data.qpos[:7], candidate["impact_pose"], (deg(18.0), deg(18.0), deg(22.0), deg(12.0), deg(20.0), deg(22.0), deg(24.0)))


def elbow_score(elbow_angle):
    return angle_score(elbow_angle, IMPACT_ELBOW_TARGET, IMPACT_ELBOW_TOLERANCE)


def top_set_score(data, candidate):
    hand = candidate.get("hand", DEFAULT_HAND)
    sign = hand_sign(hand)
    elbow = data.qpos[3]
    wrist_cock = data.qpos[4]
    elbow_part = angle_score(elbow, TOP_ELBOW_TARGET, TOP_ELBOW_TOLERANCE)
    wrist_part = angle_score(
        wrist_cock,
        sign * deg(TOP_WRIST_COCK_TARGET_DEG),
        TOP_WRIST_COCK_TOLERANCE,
    )
    return 0.5 * elbow_part + 0.5 * wrist_part


def early_set_score(data, candidate):
    hand = candidate.get("hand", DEFAULT_HAND)
    sign = hand_sign(hand)
    elbow = data.qpos[3]
    wrist_cock = data.qpos[4]
    elbow_part = angle_score(elbow, EARLY_ELBOW_TARGET, EARLY_ELBOW_TOLERANCE)
    wrist_part = angle_score(
        wrist_cock,
        sign * deg(EARLY_WRIST_COCK_TARGET_DEG),
        EARLY_WRIST_COCK_TOLERANCE,
    )
    return 0.5 * elbow_part + 0.5 * wrist_part


def early_rotation_score(data, candidate):
    hand = candidate.get("hand", DEFAULT_HAND)
    sign = hand_sign(hand)
    shoulder_turn = data.qpos[0]
    shoulder_lift = data.qpos[1]
    shoulder_twist = data.qpos[2]
    turn_part = angle_score(
        shoulder_turn,
        sign * deg(EARLY_SHOULDER_TURN_TARGET_DEG),
        EARLY_SHOULDER_TURN_TOLERANCE,
    )
    lift_part = angle_score(
        shoulder_lift,
        sign * deg(EARLY_SHOULDER_LIFT_TARGET_DEG),
        EARLY_SHOULDER_LIFT_TOLERANCE,
    )
    twist_part = angle_score(
        shoulder_twist,
        sign * deg(EARLY_SHOULDER_TWIST_TARGET_DEG),
        EARLY_SHOULDER_TWIST_TOLERANCE,
    )
    return (turn_part + lift_part + twist_part) / 3.0


def simulate_swing(model, candidate, launch_profile, require_hit=True):
    data = mujoco.MjData(model)
    apply_setup_pose(model, data, candidate.get("hand", DEFAULT_HAND))

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

    initial_ball_x = data.xpos[ball_id][0]
    initial_ball_y = data.xpos[ball_id][1]
    initial_ball_z = data.xpos[ball_id][2]
    max_ball_x = initial_ball_x
    max_ball_z = initial_ball_z
    min_tip_to_ball = math.inf
    active_min_tip_to_ball = math.inf
    active_max_club_vx = -math.inf
    first_hit_step = None
    hit_ball = False
    impact_ball_x = None
    impact_club_vx = 0.0
    impact_elbow = 0.0
    post_impact_max_ball_vx = 0.0
    post_impact_max_ball_vz = 0.0
    post_impact_distance = 0.0
    top_sample = 0.0
    early_set_sample = 0.0
    early_rotation_sample = 0.0
    top_set_sample = 0.0
    impact_sample = 0.0
    elbow_sample = 0.0
    address_sample = address_score(data, candidate)
    total_ctrl_energy = 0.0
    max_offline_y = 0.0
    impact_was_active_downswing = False
    impact_club_was_forward = False

    previous_tip_pos = data.site_xpos[club_tip_id].copy()
    previous_ball_pos = data.xpos[ball_id].copy()

    for step in range(EPISODE_STEPS):
        total_ctrl_energy += apply_pd_controls(data, candidate, step)
        mujoco.mj_step(model, data)

        ball_pos = data.xpos[ball_id]
        tip_pos = data.site_xpos[club_tip_id]
        club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
        ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep
        tip_to_ball = math.dist(tip_pos, ball_pos)
        down_start_step = candidate.get("down_start_step", candidate["top_step"] + candidate.get("top_hold", 0))
        active_downswing = down_start_step <= step <= candidate["impact_step"] + 55

        min_tip_to_ball = min(min_tip_to_ball, tip_to_ball)
        max_ball_x = max(max_ball_x, ball_pos[0])
        max_ball_z = max(max_ball_z, ball_pos[2])
        max_offline_y = max(max_offline_y, abs(ball_pos[1] - initial_ball_y))
        if active_downswing:
            active_min_tip_to_ball = min(active_min_tip_to_ball, tip_to_ball)
            active_max_club_vx = max(active_max_club_vx, club_velocity[0])

        if step == down_start_step:
            top_sample = top_score(data, candidate)
            top_set_sample = top_set_score(data, candidate)

        if step == int(candidate["top_step"] * EARLY_SET_FRACTION):
            early_set_sample = early_set_score(data, candidate)

        if step == int(candidate["top_step"] * EARLY_ROTATION_FRACTION):
            early_rotation_sample = early_rotation_score(data, candidate)

        if contact_includes(data, club_head_geom_id, ball_geom_id):
            hit_ball = True
            if first_hit_step is None:
                first_hit_step = step
                impact_ball_x = ball_pos[0]
                impact_club_vx = club_velocity[0]
                impact_elbow = data.qpos[3]
                impact_was_active_downswing = active_downswing
                impact_club_was_forward = impact_club_vx >= MIN_FORWARD_CLUB_SPEED
                impact_sample = impact_pose_score(data, candidate)
                elbow_sample = elbow_score(impact_elbow)

        if first_hit_step is not None and step <= first_hit_step + POST_IMPACT_WINDOW_STEPS:
            post_impact_max_ball_vx = max(post_impact_max_ball_vx, ball_velocity[0])
            post_impact_max_ball_vz = max(post_impact_max_ball_vz, ball_velocity[2])
            if impact_ball_x is not None:
                post_impact_distance = max(post_impact_distance, ball_pos[0] - impact_ball_x)

        previous_tip_pos = tip_pos.copy()
        previous_ball_pos = ball_pos.copy()

    if active_min_tip_to_ball == math.inf:
        active_min_tip_to_ball = min_tip_to_ball
    if active_max_club_vx == -math.inf:
        active_max_club_vx = 0.0

    ball_launched_forward = (
        post_impact_max_ball_vx >= MIN_FORWARD_BALL_SPEED
        or post_impact_distance >= MIN_POST_IMPACT_DISTANCE
    )
    valid_impact = impact_was_active_downswing and impact_club_was_forward and ball_launched_forward

    distance = max_ball_x - initial_ball_x
    height_gain = max(0.0, max_ball_z - initial_ball_z)
    launch_score = value_score(
        post_impact_max_ball_vz,
        launch_profile["target_vertical_speed"],
        launch_profile["vertical_speed_tolerance"],
    )
    excess_vertical_speed = max(
        0.0,
        post_impact_max_ball_vz
        - launch_profile["target_vertical_speed"]
        - launch_profile["vertical_speed_tolerance"],
    )
    sequence_score = 1.0 if candidate["wrist_lag"] > candidate["elbow_lag"] else 0.0
    sequence_score += (
        address_sample
        + top_sample
        + top_set_sample
        + early_set_sample
        + early_rotation_sample
        + impact_sample
        + elbow_sample
    )

    reward = (
        distance * launch_profile["distance_weight"]
        + height_gain * launch_profile["height_weight"]
        + max(0.0, post_impact_max_ball_vx) * launch_profile["forward_speed_weight"]
        + max(0.0, post_impact_max_ball_vz) * launch_profile["vertical_speed_weight"]
        + launch_score * launch_profile["launch_score_weight"]
        + address_sample * 120.0
        + top_sample * 420.0
        + top_set_sample * 380.0
        + early_set_sample * 260.0
        + early_rotation_sample * 320.0
        + impact_sample * 180.0
        + elbow_sample * 200.0
        + sequence_score * 45.0
        - max_offline_y * OFFLINE_Y_PENALTY
        - excess_vertical_speed * launch_profile["excess_vertical_speed_penalty"]
        - total_ctrl_energy * 0.00002
    )
    if require_hit and not valid_impact:
        reward = (
            INVALID_SWING_REWARD
            - active_min_tip_to_ball * 260.0
            + max(0.0, active_max_club_vx) * 8.0
            + address_sample * 120.0
            + top_sample * 420.0
            + top_set_sample * 380.0
            + early_set_sample * 260.0
            + early_rotation_sample * 320.0
            + elbow_sample * 200.0
            + sequence_score * 45.0
            - max_offline_y * OFFLINE_Y_PENALTY
        )

    return {
        "reward": reward,
        "distance": distance,
        "height_gain": height_gain,
        "hit_ball": hit_ball,
        "valid_impact": valid_impact,
        "first_hit_step": first_hit_step,
        "impact_club_vx": impact_club_vx,
        "impact_elbow": impact_elbow,
        "post_impact_max_ball_vx": post_impact_max_ball_vx,
        "post_impact_max_ball_vz": post_impact_max_ball_vz,
        "post_impact_distance": post_impact_distance,
        "address_score": address_sample,
        "top_score": top_sample,
        "top_set_score": top_set_sample,
        "early_set_score": early_set_sample,
        "early_rotation_score": early_rotation_sample,
        "impact_pose_score": impact_sample,
        "impact_elbow_score": elbow_sample,
        "launch_score": launch_score,
        "sequence_score": sequence_score,
        "active_min_tip_to_ball": active_min_tip_to_ball,
        "active_max_club_vx": active_max_club_vx,
        "offline_y": max_offline_y,
    }


def mean(values):
    return sum(values) / len(values)


def stddev(values, center):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def update_distribution(elite_vectors, old_mean, old_std, specs, smoothing):
    new_mean = []
    new_std = []
    mins = min_std()
    for i, (_, low, high) in enumerate(specs):
        values = [vector[i] for vector in elite_vectors]
        elite_mean = clip(mean(values), low, high)
        elite_std = max(stddev(values, elite_mean), mins[i])
        new_mean.append((1.0 - smoothing) * old_mean[i] + smoothing * elite_mean)
        new_std.append((1.0 - smoothing) * old_std[i] + smoothing * elite_std)
    return new_mean, new_std


def print_result(prefix, result):
    print(
        prefix,
        "reward",
        round(result["reward"], 3),
        "distance",
        round(result["distance"], 4),
        "height",
        round(result["height_gain"], 4),
        "hit",
        result["hit_ball"],
        "valid",
        result["valid_impact"],
        "first_hit_step",
        result["first_hit_step"],
        "post_vx",
        round(result["post_impact_max_ball_vx"], 4),
        "post_vz",
        round(result["post_impact_max_ball_vz"], 4),
        "top",
        round(result["top_score"], 3),
        "top_set",
        round(result["top_set_score"], 3),
        "early_set",
        round(result["early_set_score"], 3),
        "early_rot",
        round(result["early_rotation_score"], 3),
        "offline_y",
        round(result["offline_y"], 4),
        "impact",
        round(result["impact_pose_score"], 3),
        "elbow_deg",
        round(math.degrees(result["impact_elbow"]), 2),
        "elbow",
        round(result["impact_elbow_score"], 3),
        "launch",
        round(result["launch_score"], 3),
    )


def print_candidate(candidate):
    print("\n# Paste this dictionary into human_right_arm_biomech_swing.py as BIOMECH_SWING_CANDIDATE:\n")
    print("BIOMECH_SWING_CANDIDATE = {")
    print(f'    "hand": "{candidate["hand"]}",')
    for key in ("top_step", "top_hold", "down_start_step", "impact_step", "finish_step", "elbow_lag", "wrist_lag"):
        print(f'    "{key}": {candidate[key]},')
    for key in ("address_pose", "top_pose", "impact_pose", "finish_pose"):
        values = ", ".join(f"{value:.4f}" for value in candidate[key])
        print(f'    "{key}": ({values}),')
    print("}")


def train(generations, population, elite_count, seed=None, smoothing=0.7, club_name="7iron", hand=DEFAULT_HAND):
    rng = random.Random(seed)
    club_name = normalize_club_name(club_name)
    hand = normalize_hand(hand)
    specs = parameter_specs(hand)
    mean_vector = initial_mean(hand)
    std_vector = initial_std()
    model = make_static_model(club_name, hand, club_contact=True, include_actuators=True)
    launch_profile = get_club_launch_profile(club_name)

    print("Biomech dynamic model:", club_name, hand)
    print("Joint names:", JOINT_NAMES)
    print("Address pose deg:", [round(math.degrees(v), 2) for v in address_pose(hand)])
    print("Default top pose deg:", [round(math.degrees(v), 2) for v in default_top_pose(hand)])

    best_result = None
    best_candidate = None
    best_valid_result = None
    best_valid_candidate = None
    start_time = time.time()

    for generation in range(1, generations + 1):
        scored = []
        for _ in range(population):
            vector = sample_vector(rng, mean_vector, std_vector, specs)
            candidate = vector_to_candidate(vector, hand)
            result = simulate_swing(model, candidate, launch_profile, require_hit=True)
            scored.append((result["reward"], vector, candidate, result))

            if best_result is None or result["reward"] > best_result["reward"]:
                best_result = result
                best_candidate = candidate
            if result["valid_impact"] and (
                best_valid_result is None or result["reward"] > best_valid_result["reward"]
            ):
                best_valid_result = result
                best_valid_candidate = candidate

        scored.sort(key=lambda item: item[0], reverse=True)
        if best_valid_result is not None:
            valid_items = [item for item in scored if item[3]["valid_impact"]]
            elite_source = valid_items if len(valid_items) >= max(2, elite_count // 2) else scored
        else:
            elite_source = scored
        elite_vectors = [item[1] for item in elite_source[:elite_count]]
        mean_vector, std_vector = update_distribution(elite_vectors, mean_vector, std_vector, specs, smoothing)

        elapsed = time.time() - start_time
        print_result(f"Generation {generation}/{generations}", scored[0][3])
        print("valid_found", best_valid_result is not None, "elapsed_sec", round(elapsed, 1))

    final_result = best_valid_result if best_valid_result is not None else best_result
    final_candidate = best_valid_candidate if best_valid_candidate is not None else best_candidate
    print("Training complete")
    if best_valid_result is None:
        print("No valid biomech impact found yet.")
    print_result("Final best", final_result)
    print("Impact club vx:", round(final_result["impact_club_vx"], 4))
    print("Impact elbow bend deg:", round(math.degrees(final_result["impact_elbow"]), 4))
    print("Post-impact distance:", round(final_result["post_impact_distance"], 4))
    print("Address score:", round(final_result["address_score"], 4))
    print("Top score:", round(final_result["top_score"], 4))
    print("Top set score:", round(final_result["top_set_score"], 4))
    print("Early hinge set score:", round(final_result["early_set_score"], 4))
    print("Early shoulder/elbow-plane rotation score:", round(final_result["early_rotation_score"], 4))
    print("Offline y:", round(final_result["offline_y"], 4))
    print("Impact pose score:", round(final_result["impact_pose_score"], 4))
    print("Impact elbow score:", round(final_result["impact_elbow_score"], 4))
    print("Launch score:", round(final_result["launch_score"], 4))
    print_candidate(final_candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="CEM trainer for the 6-joint right-arm biomechanics model.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        generations=args.generations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        smoothing=args.smoothing,
        club_name=args.club,
        hand=args.hand,
    )
