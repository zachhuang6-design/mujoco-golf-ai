import time

import mujoco
import mujoco.viewer

from golf_3joint_common import (
    BASELINE_CANDIDATE,
    apply_three_joint_controls,
    make_single_arm_model,
)


MIN_FORWARD_CLUB_SPEED = 0.50
MIN_FORWARD_BALL_SPEED = 0.20

# Replace this dictionary with the output from human_random_ai_trained3joint.py.
SWING_CANDIDATE = {
    "t1": 124,
    "t2": 216,
    "t3": 348,
    "s1": 4.997,
    "e1": 0.249,
    "w1": 2.921,
    "s2": -6.561,
    "e2": -2.513,
    "w2": -1.031,
    "s3": -3.305,
    "e3": -3.849,
    "w3": -3.284,
}


model = make_single_arm_model()
data = mujoco.MjData(model)

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)


def club_hit_ball():
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {club_head_geom_id, ball_geom_id}:
            return True
    return False


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        printed_first_hit = False
        previous_tip_pos = data.site_xpos[club_tip_id].copy()
        previous_ball_pos = data.xpos[ball_id].copy()

        while viewer.is_running():
            apply_three_joint_controls(data, SWING_CANDIDATE, step)
            mujoco.mj_step(model, data)

            tip_pos = data.site_xpos[club_tip_id]
            ball_pos = data.xpos[ball_id]
            club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
            ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep

            if club_hit_ball() and not printed_first_hit:
                active_downswing = SWING_CANDIDATE["t1"] <= step <= SWING_CANDIDATE["t3"] + 20
                valid_impact = (
                    active_downswing
                    and club_velocity[0] >= MIN_FORWARD_CLUB_SPEED
                    and ball_velocity[0] >= MIN_FORWARD_BALL_SPEED
                )
                print("club head contacted ball at step:", step)
                print("valid downswing impact:", valid_impact)
                print("club vx at impact:", round(club_velocity[0], 3))
                print("ball vx at impact:", round(ball_velocity[0], 3))
                print("ball vz at impact:", round(ball_velocity[2], 3))
                printed_first_hit = True

            if step % 100 == 0:
                print(
                    "step:", step,
                    "club tip x:", round(tip_pos[0], 3),
                    "z:", round(tip_pos[2], 3),
                    "club vx:", round(club_velocity[0], 3),
                    "ball x:", round(ball_pos[0], 3),
                    "hit:", printed_first_hit,
                )

            previous_tip_pos = tip_pos.copy()
            previous_ball_pos = ball_pos.copy()
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 10)