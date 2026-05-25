import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_3joint_common import CLUB_PRESETS, get_club_preset
from human_right_arm_biomech_cem import (
    JOINT_NAMES,
    apply_pd_controls,
    default_finish_pose,
    default_impact_pose,
    default_top_pose,
    target_angles,
)
from human_right_arm_biomech_static import (
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


def parse_args():
    parser = argparse.ArgumentParser(description="Replay the 6-joint biomechanics golf swing.")
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    return parser.parse_args()


def club_hit_ball(data, club_head_geom_id, ball_geom_id):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {club_head_geom_id, ball_geom_id}:
            return True
    return False


if __name__ == "__main__":
    args = parse_args()
    hand = normalize_hand(args.hand)
    preset = get_club_preset(args.club)
    candidate = dict(BIOMECH_SWING_CANDIDATE)
    candidate["hand"] = hand
    if candidate.get("address_pose") is None:
        from human_right_arm_biomech_cem import address_pose

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
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

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
            ctrl_energy = apply_pd_controls(data, candidate, step)
            target = target_angles(candidate, step)
            mujoco.mj_step(model, data)

            tip_pos = data.site_xpos[club_tip_id]
            ball_pos = data.xpos[ball_id]
            club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
            ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep

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

            previous_tip_pos = tip_pos.copy()
            previous_ball_pos = ball_pos.copy()
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 8)
