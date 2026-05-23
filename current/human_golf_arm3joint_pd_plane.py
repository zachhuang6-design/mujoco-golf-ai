import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_3joint_common import CLUB_PRESETS, get_club_preset
from golf_3joint_plane_common import (
    ADDRESS_POSE,
    DEFAULT_PLANE_ANGLE_DEG,
    make_plane_model,
    print_plane_setup,
)
from human_cem_pd_trained3joint_plane import apply_pd_controls, target_angles


PD_SWING_CANDIDATE = {
    "top_step": 309,
    "top_hold": 93,
    "down_start_step": 402,
    "impact_step": 700,
    "finish_step": 760,
    "elbow_lag": 59,
    "wrist_lag": 67,
    "address_pose": ADDRESS_POSE,
    "top_pose": (1.5708, 1.5708, 1.5708),
    "impact_pose": (-0.3167, 0.1497, -0.4383),
    "finish_pose": (-1.2696, 0.2234, -0.9416),
}

MIN_FORWARD_CLUB_SPEED = 0.50
MIN_POST_IMPACT_DISTANCE = 0.03
POST_IMPACT_WINDOW_STEPS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a 3D-plane PD 3-joint golf swing.")
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--plane-angle-deg", type=float, default=DEFAULT_PLANE_ANGLE_DEG)
    return parser.parse_args()


def club_hit_ball(data, club_head_geom_id, ball_geom_id):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {club_head_geom_id, ball_geom_id}:
            return True
    return False


if __name__ == "__main__":
    args = parse_args()
    preset = get_club_preset(args.club)
    print_plane_setup(args.club, args.plane_angle_deg)
    print("Viewer club label:", preset["label"])

    model = make_plane_model(args.club, args.plane_angle_deg)
    data = mujoco.MjData(model)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        first_hit_step = None
        impact_ball_x = None
        impact_club_vx = 0.0
        post_impact_distance = 0.0
        post_impact_max_ball_vx = 0.0
        previous_tip_pos = data.site_xpos[club_tip_id].copy()
        previous_ball_pos = data.xpos[ball_id].copy()

        while viewer.is_running():
            ctrl_energy = apply_pd_controls(data, PD_SWING_CANDIDATE, step)
            target = target_angles(PD_SWING_CANDIDATE, step)
            mujoco.mj_step(model, data)

            tip_pos = data.site_xpos[club_tip_id]
            ball_pos = data.xpos[ball_id]
            club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
            ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep

            if club_hit_ball(data, club_head_geom_id, ball_geom_id) and first_hit_step is None:
                first_hit_step = step
                impact_ball_x = ball_pos[0]
                impact_club_vx = club_velocity[0]
                down_start_step = PD_SWING_CANDIDATE.get(
                    "down_start_step",
                    PD_SWING_CANDIDATE["top_step"] + PD_SWING_CANDIDATE.get("top_hold", 0),
                )
                active_downswing = down_start_step <= step <= PD_SWING_CANDIDATE["impact_step"] + 45
                print("club head contacted ball at step:", step)
                print("active downswing:", active_downswing)
                print("club vx at impact:", round(impact_club_vx, 3))
                print("elbow bend at impact deg:", round(math.degrees(data.qpos[1]), 3))

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
                    tuple(round(data.qpos[i], 3) for i in range(3)),
                    "club vx:",
                    round(club_velocity[0], 3),
                    "ball x:",
                    round(ball_pos[0], 3),
                    "elbow deg:",
                    round(math.degrees(data.qpos[1]), 2),
                    "ctrl energy:",
                    round(ctrl_energy, 2),
                )

            previous_tip_pos = tip_pos.copy()
            previous_ball_pos = ball_pos.copy()
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 8)
