"""Inverse-kinematics bridge from the kinematic swing to the joint model."""

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from golf_core import two_arm_chest_followthrough as kinematic
from golf_core import two_arm_chest_takeaway as kinematic_club
from golf_core.common import get_club_preset
from golf_core.two_arm_joint_physics import (
    CONTROL_JOINTS,
    FACE_NORMAL_MARKER_LEN,
    PD_KD,
    PD_KP,
    DEFAULT_CLUB,
    DEFAULT_HAND,
    SWING_STEPS,
    TRAIL_GRIP_DOWN_SHAFT,
    TORQUE_LIMITS,
    baseline_pd_control,
    joint_qpos_indices,
    joint_qvel_indices,
    make_joint_model,
    reset_to_baseline,
)


IK_SITE_TARGETS = (
    ("left_shoulder_site", "left_shoulder", 3.0),
    ("right_shoulder_site", "right_shoulder", 3.0),
    ("left_elbow_site", "left_elbow", 2.4),
    ("right_elbow_site", "right_elbow", 2.4),
    ("left_grip_site", "left_wrist", 6.0),
    ("right_grip_site", "right_wrist", 6.0),
    ("club_trail_grip_site", "right_wrist", 4.0),
    ("clubhead_site", "clubhead", 3.0),
)

IK_SITE_LOCKS = (
    ("right_grip_site", "club_trail_grip_site", 18.0),
)


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def smooth_window(progress, start, peak_start, peak_end, end):
    if progress <= start or progress >= end:
        return 0.0
    if peak_start <= progress <= peak_end:
        return 1.0
    if progress < peak_start:
        t = (progress - start) / max(peak_start - start, 1e-9)
    else:
        t = (end - progress) / max(end - peak_end, 1e-9)
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def normalized(values):
    vector = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def square_face_normal(club_name):
    """World-space face normal for a square, lofted clubface at impact."""
    loft = math.radians(get_club_preset(club_name)["loft_deg"])
    return normalized((math.cos(loft), 0.0, math.sin(loft)))


def target_site_weight(site_name, base_weight, progress):
    """Relax old visual-target matching after impact so grip integrity wins."""
    if progress < 0.89:
        return base_weight
    if site_name in {"left_grip_site", "right_grip_site", "club_trail_grip_site"}:
        if progress >= 0.92:
            return base_weight * 0.04
        return base_weight * 0.25
    if site_name == "clubhead_site":
        if progress >= 0.92:
            return base_weight * 0.18
        return base_weight * 0.45
    if progress >= 0.92 and site_name in {"right_elbow_site", "left_elbow_site"}:
        return base_weight * 0.55
    return base_weight


def physical_site_target(site_name, targets, hand, club_name):
    if site_name in {"right_grip_site", "club_trail_grip_site"}:
        parts = kinematic_club.club_visual_parts(targets, club_name, hand)
        left_wrist = np.asarray(targets["left_wrist"], dtype=np.float64)
        heel = np.asarray(parts["heel"], dtype=np.float64)
        shaft_axis = normalized(heel - left_wrist)
        return left_wrist + shaft_axis * TRAIL_GRIP_DOWN_SHAFT
    return None


def grip_lock_weight(base_weight, progress):
    if progress < 0.22:
        return base_weight * 15.0
    if progress < 0.84:
        return base_weight * 7.0
    if progress < 0.92:
        return base_weight * 10.0
    return base_weight * 15.0


def face_square_weight(progress):
    setup_weight = 12.0 * (1.0 - clamp01(progress / 0.16))
    impact_weight = 13.0 * smooth_window(progress, 0.68, 0.75, 0.84, 0.90)
    return max(setup_weight, impact_weight)


def raw_kinematic_targets(progress, hand=DEFAULT_HAND, club_name=DEFAULT_CLUB):
    return kinematic.pose_points(hand, progress, club_name)


def kinematic_targets(progress, hand=DEFAULT_HAND, club_name=DEFAULT_CLUB):
    """IK targets with a small chest-lead transition bias."""
    if not 0.50 < progress < 0.62:
        return raw_kinematic_targets(progress, hand, club_name)

    phase = clamp01((progress - 0.50) / 0.12)
    chest_phase = phase * phase * (3.0 - 2.0 * phase)
    arm_phase = clamp01((phase - 0.34) / 0.66)
    arm_phase = arm_phase * arm_phase * (3.0 - 2.0 * arm_phase)
    chest_progress = 0.50 + 0.12 * chest_phase
    arm_progress = 0.50 + 0.12 * arm_phase

    arm_targets = raw_kinematic_targets(arm_progress, hand, club_name)
    chest_targets = raw_kinematic_targets(chest_progress, hand, club_name)
    targets = dict(arm_targets)
    for key in ("chest_center", "left_shoulder", "right_shoulder"):
        if key in chest_targets:
            targets[key] = chest_targets[key]
    return targets


def site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def presolve_chest_turn(model, data, qpos_indices, q, targets):
    """Choose chest turn that best matches the kinematic shoulder line.

    The chest has only one DoF in this bridge. Solving it explicitly prevents
    the arm joints from trying to fake chest turn during the early backswing.
    """
    chest_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chest_turn")
    chest_index = list(CONTROL_JOINTS).index("chest_turn")
    low, high = model.jnt_range[chest_joint_id]
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_shoulder_site")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_shoulder_site")
    left_target = np.asarray(targets["left_shoulder"], dtype=np.float64)
    right_target = np.asarray(targets["right_shoulder"], dtype=np.float64)

    best_turn = q[chest_index]
    best_error = float("inf")
    for turn in np.linspace(low, high, 181):
        test_q = q.copy()
        test_q[chest_index] = turn
        data.qpos[qpos_indices] = test_q
        mujoco.mj_forward(model, data)
        error = (
            np.linalg.norm(data.site_xpos[left_site] - left_target)
            + np.linalg.norm(data.site_xpos[right_site] - right_target)
        )
        if error < best_error:
            best_error = error
            best_turn = turn
    q[chest_index] = best_turn
    data.qpos[qpos_indices] = q
    mujoco.mj_forward(model, data)
    return q


def set_kinematic_target_markers(model, data, targets):
    marker_keys = (
        ("target_left_wrist_body", "left_wrist"),
        ("target_right_wrist_body", "right_wrist"),
        ("target_clubhead_body", "clubhead"),
        ("target_left_elbow_body", "left_elbow"),
        ("target_right_elbow_body", "right_elbow"),
    )
    for body_name, target_key in marker_keys:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = model.body_mocapid[body_id]
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = np.asarray(targets[target_key], dtype=np.float64)


def residual_and_jacobian(model, data, qvel_indices, targets, hand, club_name, progress):
    residuals = []
    jac_rows = []
    for site_name, target_key, weight in IK_SITE_TARGETS:
        weight = target_site_weight(site_name, weight, progress)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        target = physical_site_target(site_name, targets, hand, club_name)
        if target is None:
            target = np.asarray(targets[target_key], dtype=np.float64)
        current = data.site_xpos[site_id].copy()
        residuals.append((target - current) * weight)
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac_rows.append(jacp[:, qvel_indices] * weight)
    for site_a, site_b, weight in IK_SITE_LOCKS:
        weight = grip_lock_weight(weight, progress)
        site_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_a)
        site_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_b)
        residuals.append((data.site_xpos[site_b_id] - data.site_xpos[site_a_id]) * weight)
        jacp_a = np.zeros((3, model.nv), dtype=np.float64)
        jacp_b = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp_a, jacr, site_a_id)
        mujoco.mj_jacSite(model, data, jacp_b, jacr, site_b_id)
        jac_rows.append((jacp_a[:, qvel_indices] - jacp_b[:, qvel_indices]) * weight)

    weight = face_square_weight(progress)
    if weight > 0.0:
        clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "clubhead_site")
        face_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_face_normal_site")
        target_vector = square_face_normal(club_name) * FACE_NORMAL_MARKER_LEN
        current_vector = data.site_xpos[face_id] - data.site_xpos[clubhead_id]
        residuals.append((target_vector - current_vector) * weight)

        jacp_head = np.zeros((3, model.nv), dtype=np.float64)
        jacp_face = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp_head, jacr, clubhead_id)
        mujoco.mj_jacSite(model, data, jacp_face, jacr, face_id)
        jac_rows.append((jacp_face[:, qvel_indices] - jacp_head[:, qvel_indices]) * weight)
    return np.concatenate(residuals), np.vstack(jac_rows)


def elbow_flex_radians(shoulder, elbow, wrist):
    upper = np.asarray(shoulder, dtype=np.float64) - np.asarray(elbow, dtype=np.float64)
    forearm = np.asarray(wrist, dtype=np.float64) - np.asarray(elbow, dtype=np.float64)
    upper /= max(np.linalg.norm(upper), 1e-9)
    forearm /= max(np.linalg.norm(forearm), 1e-9)
    angle = math.acos(float(np.clip(np.dot(upper, forearm), -1.0, 1.0)))
    return math.pi - angle


def joint_residuals_and_jacobian(model, qvel_indices, q, progress, targets):
    """Phase-aware anatomical priors for the IK solve.

    The lead arm should stay essentially straight through the backswing and
    impact. Without this prior, the solver can match the hand target by folding
    the lead elbow backward, which looks visually and anatomically wrong.
    """
    residuals = []
    jac_rows = []
    joint_index = {name: index for index, name in enumerate(CONTROL_JOINTS)}

    def add_joint_target(name, target_radians, weight):
        row = np.zeros((1, len(CONTROL_JOINTS)), dtype=np.float64)
        row[0, joint_index[name]] = 1.0
        residuals.append(np.array([(target_radians - q[joint_index[name]]) * weight]))
        jac_rows.append(row * weight)

    if progress <= 0.50:
        if progress <= 0.275:
            t = progress / 0.275
            target_right_flex = math.radians(8.0 + 28.0 * (t * t * (3.0 - 2.0 * t)))
            right_weight = 3.0
        else:
            t = (progress - 0.275) / 0.225
            t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
            target_right_flex = math.radians(36.0 + 56.0 * t)
            right_weight = 2.2
        add_joint_target("right_elbow_flex", target_right_flex, right_weight)
    elif progress < 0.64:
        t = (progress - 0.50) / 0.14
        target_right_flex = math.radians(92.0 - 34.0 * smooth_window(t, -0.01, 0.25, 1.0, 1.01))
        add_joint_target("right_elbow_flex", target_right_flex, 1.2)

    if 0.50 < progress < 0.66:
        t = (progress - 0.50) / 0.16
        t = clamp01(t)
        lead = t * t * (3.0 - 2.0 * t)
        target_chest = math.radians(-94.0 + 36.0 * lead)
        add_joint_target("chest_turn", target_chest, 2.0)

    if progress <= 0.84:
        add_joint_target("left_elbow_flex", 0.0, 5.0)
    elif progress < 0.93:
        blend = (progress - 0.84) / 0.09
        add_joint_target("left_elbow_flex", 0.0, 5.0 * (1.0 - blend))
    else:
        target_left_flex = elbow_flex_radians(
            targets["left_shoulder"],
            targets["left_elbow"],
            targets["left_wrist"],
        )
        add_joint_target("left_elbow_flex", target_left_flex, 2.5)

    if not residuals:
        return None, None
    return np.concatenate(residuals), np.vstack(jac_rows)


def temporal_residuals_and_jacobian(q, reference_q, progress):
    """Discourage frame-to-frame IK flips in the finish."""
    weight = 0.25
    if progress >= 0.88:
        weight = 3.0
    residual = (reference_q - q) * weight
    jacobian = np.eye(len(CONTROL_JOINTS), dtype=np.float64) * weight
    return residual, jacobian


def solve_ik_frame(
    model,
    data,
    qpos_indices,
    qvel_indices,
    progress,
    initial_qpos,
    hand=DEFAULT_HAND,
    club_name=DEFAULT_CLUB,
    iterations=80,
    damping=0.08,
    max_step=0.06,
):
    q = np.asarray(initial_qpos, dtype=np.float64).copy()
    targets = kinematic_targets(progress, hand, club_name)
    q = presolve_chest_turn(model, data, qpos_indices, q, targets)
    joint_range = model.jnt_range[
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CONTROL_JOINTS]
    ]

    best_q = q.copy()
    best_error = float("inf")
    for _ in range(iterations):
        data.qpos[qpos_indices] = q
        mujoco.mj_forward(model, data)
        residual, jacobian = residual_and_jacobian(
            model, data, qvel_indices, targets, hand, club_name, progress
        )
        joint_residual, joint_jacobian = joint_residuals_and_jacobian(
            model, qvel_indices, q, progress, targets
        )
        if joint_residual is not None:
            residual = np.concatenate([residual, joint_residual])
            jacobian = np.vstack([jacobian, joint_jacobian])
        temporal_residual, temporal_jacobian = temporal_residuals_and_jacobian(
            q, initial_qpos, progress
        )
        residual = np.concatenate([residual, temporal_residual])
        jacobian = np.vstack([jacobian, temporal_jacobian])
        error = float(np.linalg.norm(residual))
        if error < best_error:
            best_error = error
            best_q = q.copy()
        if error < 0.015:
            break
        lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(jacobian.shape[0])
        step = jacobian.T @ np.linalg.solve(lhs, residual)
        step = np.clip(step, -max_step, max_step)
        q = np.clip(q + step, joint_range[:, 0], joint_range[:, 1])

    data.qpos[qpos_indices] = best_q
    mujoco.mj_forward(model, data)
    return best_q, best_error


def solve_ik_trajectory(
    club_name=DEFAULT_CLUB,
    hand=DEFAULT_HAND,
    frames=121,
    iterations=80,
):
    model = make_joint_model(club_name, hand)
    data = mujoco.MjData(model)
    reset_to_baseline(model, data)
    qpos_indices = joint_qpos_indices(model)
    qvel_indices = joint_qvel_indices(model)
    q = data.qpos[qpos_indices].copy()
    solved = []
    errors = []
    for frame in range(frames):
        progress = frame / max(frames - 1, 1)
        q, error = solve_ik_frame(
            model,
            data,
            qpos_indices,
            qvel_indices,
            progress,
            q,
            hand=hand,
            club_name=club_name,
            iterations=iterations,
        )
        solved.append(q.copy())
        errors.append(error)
    return np.asarray(solved), np.asarray(errors)


def trajectory_report(path):
    payload, qpos = load_trajectory(path)
    model = make_joint_model(payload["club"], payload["hand"])
    data = mujoco.MjData(model)
    reset_to_baseline(model, data)
    qpos_indices = joint_qpos_indices(model)
    rows = []
    for progress in np.linspace(0.0, 1.0, 9):
        data.qpos[qpos_indices] = interp_trajectory(qpos, progress)
        mujoco.mj_forward(model, data)
        targets = kinematic_targets(progress, payload["hand"], payload["club"])
        errors = {}
        for site_name, target_key, _ in IK_SITE_TARGETS:
            errors[f"{site_name}->{target_key}"] = float(
                np.linalg.norm(site_position(model, data, site_name) - np.asarray(targets[target_key]))
            )
        errors["right_grip_lock"] = float(
            np.linalg.norm(
                site_position(model, data, "right_grip_site")
                - site_position(model, data, "club_trail_grip_site")
            )
        )
        rows.append((progress, errors))
    return rows


def save_trajectory(path, qpos, errors, club_name, hand):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "club": club_name,
        "hand": hand,
        "joints": CONTROL_JOINTS,
        "qpos_radians": qpos.tolist(),
        "errors": errors.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_trajectory(path):
    payload = json.loads(Path(path).read_text())
    return payload, np.asarray(payload["qpos_radians"], dtype=np.float64)


def interp_trajectory(qpos, progress):
    scaled = max(0.0, min(1.0, progress)) * (len(qpos) - 1)
    lo = int(math.floor(scaled))
    hi = min(len(qpos) - 1, lo + 1)
    t = scaled - lo
    return qpos[lo] + (qpos[hi] - qpos[lo]) * t


def interp_trajectory_velocity(qpos, progress, timestep):
    before = interp_trajectory(qpos, max(0.0, progress - 1.0 / max(SWING_STEPS - 1, 1)))
    after = interp_trajectory(qpos, min(1.0, progress + 1.0 / max(SWING_STEPS - 1, 1)))
    return (after - before) / max(2.0 * timestep, 1e-9)


def play_ik_trajectory(path, speed=6.0):
    payload, qpos = load_trajectory(path)
    model = make_joint_model(payload["club"], payload["hand"])
    data = mujoco.MjData(model)
    reset_to_baseline(model, data)
    qpos_indices = joint_qpos_indices(model)
    print("IK bridge playback")
    print("Club:", payload["club"])
    print("Arm colors: right arm = blue/green, left arm = orange/yellow")
    print("Ball and tee are fixed to the accepted kinematic address position.")
    print("Mean IK error:", round(float(np.mean(payload["errors"])), 5))
    print("Max IK error:", round(float(np.max(payload["errors"])), 5))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            progress = (step % SWING_STEPS) / max(SWING_STEPS - 1, 1)
            data.qpos[qpos_indices] = interp_trajectory(qpos, progress)
            set_kinematic_target_markers(
                model,
                data,
                kinematic_targets(progress, payload["hand"], payload["club"]),
            )
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * speed)
            step += 1


def play_pd_trajectory(path, speed=6.0):
    payload, qpos = load_trajectory(path)
    model = make_joint_model(payload["club"], payload["hand"])
    data = mujoco.MjData(model)
    reset_to_baseline(model, data)
    qpos_indices = joint_qpos_indices(model)
    qvel_indices = joint_qvel_indices(model)
    print("PD IK bridge playback")
    print("Arm colors: right arm = blue/green, left arm = orange/yellow")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            progress = (step % SWING_STEPS) / max(SWING_STEPS - 1, 1)
            target = interp_trajectory(qpos, progress)
            set_kinematic_target_markers(
                model,
                data,
                kinematic_targets(progress, payload["hand"], payload["club"]),
            )
            target_vel = interp_trajectory_velocity(qpos, progress, model.opt.timestep)
            error = target - data.qpos[qpos_indices]
            vel_error = target_vel - data.qvel[qvel_indices]
            data.ctrl[:] = np.clip(PD_KP * error + PD_KD * vel_error, -TORQUE_LIMITS, TORQUE_LIMITS)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * speed)
            step += 1


def main():
    parser = argparse.ArgumentParser(description="Solve or view IK bridge trajectory.")
    parser.add_argument("--club", default=DEFAULT_CLUB)
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=("right", "left"))
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--out", default="artifacts/ik/two_arm_joint_ik_7iron_right.json")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--pd", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--speed", type=float, default=6.0)
    args = parser.parse_args()

    if args.view:
        if args.pd:
            play_pd_trajectory(args.out, args.speed)
        else:
            play_ik_trajectory(args.out, args.speed)
        return
    if args.report:
        for progress, errors in trajectory_report(args.out):
            worst = sorted(errors.items(), key=lambda item: item[1], reverse=True)[:4]
            print("progress", round(float(progress), 3))
            for name, value in worst:
                print(" ", name, round(value, 4))
        return

    qpos, errors = solve_ik_trajectory(
        club_name=args.club,
        hand=args.hand,
        frames=args.frames,
        iterations=args.iterations,
    )
    save_trajectory(args.out, qpos, errors, args.club, args.hand)
    print("saved", args.out)
    print("mean_error", round(float(np.mean(errors)), 5))
    print("max_error", round(float(np.max(errors)), 5))


if __name__ == "__main__":
    main()
