import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from golf_core.common import CLUB_PRESETS, get_club_preset
from golf_core.right_arm_cem import (
    JOINT_NAMES,
    apply_pd_controls,
    default_finish_pose,
    default_impact_pose,
    default_top_pose,
    target_angles,
)
from golf_core.right_arm_static import (
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    make_static_model,
    normalize_hand,
    print_setup_report,
)


BIOMECH_SWING_CANDIDATE = {
    "hand": "right",
    "top_step": 347,
    "top_hold": 93,
    "down_start_step": 440,
    "impact_step": 670,
    "finish_step": 747,
    "elbow_lag": 57,
    "wrist_lag": 102,
    "address_pose": (0.0000, -0.1309, 0.0000, 0.0000, -0.3927, 0.0000, 0.0000),
    "top_pose": (-1.6462, -1.9082, -0.1057, 0.6920, -0.6109, -0.6099, 0.3444),
    "impact_pose": (0.1327, 0.3020, -0.2100, 0.3104, 0.3047, 0.2676, 0.1631),
    "finish_pose": (0.8344, 0.6029, 0.8372, 0.3892, 1.1437, 0.3461, 0.7266),
}

MIN_FORWARD_CLUB_SPEED = 0.50
MIN_POST_IMPACT_DISTANCE = 0.03
POST_IMPACT_WINDOW_STEPS = 20
TRAIL_STRIDE = 3
MAX_TRAIL_POINTS = 260
SHAFT_SNAPSHOT_STRIDE = 8
PLANE_LOCK_KP = 8500.0
PLANE_LOCK_KD = 180.0
PLANE_LOCK_MAX_FORCE = 4500.0


def parse_args():
    parser = argparse.ArgumentParser(description="Replay the 7-joint biomechanics golf swing.")
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    parser.add_argument(
        "--force-plane",
        action="store_true",
        help="One-off debug mode: strongly guide the clubhead and shaft back onto the visual swing plane.",
    )
    return parser.parse_args()


def club_hit_ball(data, club_head_geom_id, ball_geom_id):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {club_head_geom_id, ball_geom_id}:
            return True
    return False


def normalize_vec(values):
    length = float(np.linalg.norm(values))
    if length < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.asarray(values, dtype=float) / length


def swing_plane_from_setup(data, shoulder_site_id, ball_id):
    shoulder = np.asarray(data.site_xpos[shoulder_site_id], dtype=float)
    ball = np.asarray(data.xpos[ball_id], dtype=float)
    target_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    normal = normalize_vec(np.cross(target_axis, shoulder - ball))
    return ball.copy(), normal


def plane_distance_signed(point, plane_point, plane_normal):
    return float(np.dot(np.asarray(point, dtype=float) - plane_point, plane_normal))


def clipped_plane_force(distance, normal_speed, plane_normal):
    magnitude = -PLANE_LOCK_KP * distance - PLANE_LOCK_KD * normal_speed
    magnitude = max(-PLANE_LOCK_MAX_FORCE, min(PLANE_LOCK_MAX_FORCE, magnitude))
    return plane_normal * magnitude


def apply_force_at_point(model, data, force, point, body_id):
    mujoco.mj_applyFT(
        model,
        data,
        np.asarray(force, dtype=float),
        np.zeros(3),
        np.asarray(point, dtype=float),
        body_id,
        data.qfrc_applied,
    )


def apply_plane_lock(
    model,
    data,
    plane_point,
    plane_normal,
    club_head_geom_id,
    wrist_site_id,
    previous_head_pos,
    previous_wrist_pos,
):
    data.qfrc_applied[:] = 0.0
    head_pos = np.asarray(data.geom_xpos[club_head_geom_id], dtype=float)
    wrist_pos = np.asarray(data.site_xpos[wrist_site_id], dtype=float)
    shaft_mid = 0.5 * (head_pos + wrist_pos)
    previous_shaft_mid = 0.5 * (previous_head_pos + previous_wrist_pos)

    head_velocity = (head_pos - previous_head_pos) / model.opt.timestep
    wrist_velocity = (wrist_pos - previous_wrist_pos) / model.opt.timestep
    shaft_mid_velocity = (shaft_mid - previous_shaft_mid) / model.opt.timestep

    head_body_id = model.geom_bodyid[club_head_geom_id]
    wrist_body_id = model.site_bodyid[wrist_site_id]

    for point, velocity, body_id, share in (
        (head_pos, head_velocity, head_body_id, 0.45),
        (wrist_pos, wrist_velocity, wrist_body_id, 0.30),
        (shaft_mid, shaft_mid_velocity, head_body_id, 0.25),
    ):
        distance = plane_distance_signed(point, plane_point, plane_normal)
        normal_speed = float(np.dot(velocity, plane_normal))
        force = clipped_plane_force(distance, normal_speed, plane_normal) * share
        apply_force_at_point(model, data, force, point, body_id)


def add_trail_capsule(scene, start, end, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.0, 0.0]),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.array(rgba),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1


def draw_club_trails(scene, head_trail, wrist_trail):
    scene.ngeom = 0
    for i in range(1, len(head_trail)):
        add_trail_capsule(scene, head_trail[i - 1], head_trail[i], 0.006, (1.0, 0.78, 0.05, 0.72))
    for i in range(1, len(wrist_trail)):
        add_trail_capsule(scene, wrist_trail[i - 1], wrist_trail[i], 0.004, (0.05, 0.45, 1.0, 0.46))
    for i in range(0, min(len(head_trail), len(wrist_trail)), SHAFT_SNAPSHOT_STRIDE):
        add_trail_capsule(scene, wrist_trail[i], head_trail[i], 0.003, (0.05, 0.05, 0.05, 0.28))


if __name__ == "__main__":
    args = parse_args()
    hand = normalize_hand(args.hand)
    preset = get_club_preset(args.club)
    candidate = dict(BIOMECH_SWING_CANDIDATE)
    candidate["hand"] = hand
    if candidate.get("address_pose") is None:
        from golf_core.right_arm_cem import address_pose

        candidate["address_pose"] = address_pose(hand)
    if hand != "right":
        candidate["top_pose"] = default_top_pose(hand)
        candidate["impact_pose"] = default_impact_pose(hand)
        candidate["finish_pose"] = default_finish_pose(hand)

    print(
        "Selected club:",
        preset["label"],
        "loft_deg",
        preset["loft_deg"],
        "shaft_mass",
        preset["shaft_mass"],
        "head_mass",
        preset["head_mass"],
    )
    print("Hand:", hand)
    print("Joint names:", JOINT_NAMES)

    model = make_static_model(args.club, hand, club_contact=True, include_actuators=True)
    data = mujoco.MjData(model)
    apply_setup_pose(model, data, hand)
    print_setup_report(model, data, hand)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    shoulder_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shoulder_site")
    wrist_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wrist_site")
    plane_point, plane_normal = swing_plane_from_setup(data, shoulder_site_id, ball_id)
    if args.force_plane:
        print("FORCE-PLANE MODE ENABLED: clubhead and shaft are being guided onto the blue swing plane.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        first_hit_step = None
        impact_ball_x = None
        impact_club_vx = 0.0
        post_impact_distance = 0.0
        post_impact_max_ball_vx = 0.0
        previous_head_pos = data.geom_xpos[club_head_geom_id].copy()
        previous_wrist_pos = data.site_xpos[wrist_site_id].copy()
        previous_ball_pos = data.xpos[ball_id].copy()
        head_trail = [previous_head_pos.copy()]
        wrist_trail = [data.site_xpos[wrist_site_id].copy()]

        while viewer.is_running():
            ctrl_energy = apply_pd_controls(data, candidate, step)
            target = target_angles(candidate, step)
            if args.force_plane:
                apply_plane_lock(
                    model,
                    data,
                    plane_point,
                    plane_normal,
                    club_head_geom_id,
                    wrist_site_id,
                    previous_head_pos,
                    previous_wrist_pos,
                )
            mujoco.mj_step(model, data)

            head_pos = data.geom_xpos[club_head_geom_id]
            wrist_pos = data.site_xpos[wrist_site_id]
            ball_pos = data.xpos[ball_id]
            club_velocity = (head_pos - previous_head_pos) / model.opt.timestep
            ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep
            if step % TRAIL_STRIDE == 0:
                head_trail.append(head_pos.copy())
                wrist_trail.append(wrist_pos.copy())
                head_trail = head_trail[-MAX_TRAIL_POINTS:]
                wrist_trail = wrist_trail[-MAX_TRAIL_POINTS:]

            if club_hit_ball(data, club_head_geom_id, ball_geom_id) and first_hit_step is None:
                first_hit_step = step
                impact_ball_x = ball_pos[0]
                impact_club_vx = club_velocity[0]
                down_start_step = candidate.get(
                    "down_start_step",
                    candidate["top_step"] + candidate.get("top_hold", 0),
                )
                active_downswing = down_start_step <= step <= candidate["impact_step"] + 55
                print("club head contacted ball at step:", step)
                print("active downswing:", active_downswing)
                print("club vx at impact:", round(impact_club_vx, 3))
                print("elbow bend at impact deg:", round(math.degrees(data.qpos[3]), 3))

            if first_hit_step is not None and step <= first_hit_step + POST_IMPACT_WINDOW_STEPS:
                post_impact_max_ball_vx = max(post_impact_max_ball_vx, ball_velocity[0])
                if impact_ball_x is not None:
                    post_impact_distance = max(post_impact_distance, ball_pos[0] - impact_ball_x)

            if step == (first_hit_step or -100) + POST_IMPACT_WINDOW_STEPS:
                valid = impact_club_vx >= MIN_FORWARD_CLUB_SPEED and post_impact_distance >= MIN_POST_IMPACT_DISTANCE
                print("valid post-impact launch:", valid)
                print("post-impact max ball vx:", round(post_impact_max_ball_vx, 3))
                print("post-impact distance:", round(post_impact_distance, 3))

            if step % 100 == 0:
                print(
                    "step:",
                    step,
                    "target:",
                    tuple(round(v, 3) for v in target),
                    "q:",
                    tuple(round(data.qpos[i], 3) for i in range(len(JOINT_NAMES))),
                    "club vx:",
                    round(club_velocity[0], 3),
                    "ball x:",
                    round(ball_pos[0], 3),
                    "elbow deg:",
                    round(math.degrees(data.qpos[3]), 2),
                    "ctrl energy:",
                    round(ctrl_energy, 2),
                )

            draw_club_trails(viewer.user_scn, head_trail, wrist_trail)
            previous_head_pos = head_pos.copy()
            previous_wrist_pos = wrist_pos.copy()
            previous_ball_pos = ball_pos.copy()
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 8)
