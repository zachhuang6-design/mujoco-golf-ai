import time

import mujoco
import mujoco.viewer

from golf_3joint_common import (
    BASELINE_CANDIDATE,
    apply_three_joint_controls,
    make_single_arm_model,
)


# Replace this dictionary with the output from human_random_ai_trained3joint.py.
SWING_CANDIDATE = {
    **BASELINE_CANDIDATE,
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

        while viewer.is_running():
            apply_three_joint_controls(data, SWING_CANDIDATE, step)
            mujoco.mj_step(model, data)

            if club_hit_ball() and not printed_first_hit:
                print("club head contacted ball at step:", step)
                printed_first_hit = True

            if step % 100 == 0:
                tip_x = data.site_xpos[club_tip_id][0]
                tip_z = data.site_xpos[club_tip_id][2]
                current_ball_x = data.xpos[ball_id][0]
                print(
                    "step:", step,
                    "club tip x:", round(tip_x, 3),
                    "z:", round(tip_z, 3),
                    "ball x:", round(current_ball_x, 3),
                    "hit:", printed_first_hit,
                )

            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 10)