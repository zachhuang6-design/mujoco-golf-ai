import time

import mujoco
import mujoco.viewer

from golf_3joint_common import (
    BASELINE_CANDIDATE,
    MAX_ELBOW_CTRL,
    MAX_SHOULDER_CTRL,
    MAX_WRIST_CTRL,
    apply_three_joint_controls,
    ball_mass,
    ball_radius,
    ball_x,
    ball_z,
    club_len,
    forearm_len,
    head_mass,
    shaft_mass,
    shoulder_height,
    tee_half_height,
    tee_x,
    tee_z,
    upper_arm_len,
)


TRAINED_CANDIDATE = {
    "t1": 160,
    "t2": 291,
    "t3": 395,
    "s1": 9.248,
    "e1": 0.0,
    "w1": 3.372,
    "s2": -8.22,
    "e2": 0.0,
    "w2": -3.979,
    "s3": -10.0,
    "e3": 0.0,
    "w3": -0.646,
}


def arm_xml(prefix, y, color, tip_color):
    return f"""
    <body name="{prefix}_arm" pos="0 {y:.2f} {shoulder_height}">
      <joint name="{prefix}_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="{prefix}_upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="{color}"/>

      <body name="{prefix}_forearm" pos="0 0 -{upper_arm_len:.6f}">
        <joint name="{prefix}_elbow" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="{prefix}_forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1"/>

        <body name="{prefix}_club" pos="0 0 -{forearm_len:.6f}">
          <joint name="{prefix}_wrist" type="hinge" axis="0 1 0" range="-100 100" damping="0.45" armature="0.02"/>
          <inertial mass="{shaft_mass + head_mass}" pos="0 0 -{club_len / 2:.6f}" diaginertia="0.03 0.03 0.0001"/>
          <geom name="{prefix}_club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" rgba="0.1 0.1 0.1 1"/>
          <geom name="{prefix}_club_head_geom" type="box" pos="0 0 -{club_len:.6f}" size="0.04 0.02 0.02" rgba="0.2 0.2 0.2 1"/>
          <site name="{prefix}_club_tip" pos="0 0 -{club_len:.6f}" size="0.04" rgba="{tip_color}"/>
        </body>
      </body>
    </body>

    <body name="{prefix}_tee" pos="{tee_x:.6f} {y:.2f} {tee_z:.6f}">
      <geom name="{prefix}_tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>

    <body name="{prefix}_ball" pos="{ball_x:.6f} {y:.2f} {ball_z:.6f}">
      <joint type="free"/>
      <geom name="{prefix}_ball_geom" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" rgba="1 1 1 1"/>
    </body>
"""


def build_compare_xml():
    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>
    {arm_xml("baseline", 0.20, "0.3 0.3 0.9 1", "1 0 0 1")}
    {arm_xml("trained", -0.20, "0.9 0.3 0.3 1", "0 1 0 1")}
  </worldbody>

  <actuator>
    <motor joint="baseline_shoulder" gear="7" ctrlrange="-{MAX_SHOULDER_CTRL} {MAX_SHOULDER_CTRL}" ctrllimited="true"/>
    <motor joint="baseline_elbow" gear="5" ctrlrange="-{MAX_ELBOW_CTRL} {MAX_ELBOW_CTRL}" ctrllimited="true"/>
    <motor joint="baseline_wrist" gear="5" ctrlrange="-{MAX_WRIST_CTRL} {MAX_WRIST_CTRL}" ctrllimited="true"/>
    <motor joint="trained_shoulder" gear="7" ctrlrange="-{MAX_SHOULDER_CTRL} {MAX_SHOULDER_CTRL}" ctrllimited="true"/>
    <motor joint="trained_elbow" gear="5" ctrlrange="-{MAX_ELBOW_CTRL} {MAX_ELBOW_CTRL}" ctrllimited="true"/>
    <motor joint="trained_wrist" gear="5" ctrlrange="-{MAX_WRIST_CTRL} {MAX_WRIST_CTRL}" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(build_compare_xml())
data = mujoco.MjData(model)

baseline_ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "baseline_ball")
trained_ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trained_ball")

data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running():
            apply_three_joint_controls(data, BASELINE_CANDIDATE, step, ctrl_offset=0)
            apply_three_joint_controls(data, TRAINED_CANDIDATE, step, ctrl_offset=3)

            mujoco.mj_step(model, data)

            if step % 100 == 0:
                print(
                    f"step={step} | "
                    f"baseline_x={data.xpos[baseline_ball_id][0]:.3f} | "
                    f"trained_x={data.xpos[trained_ball_id][0]:.3f}"
                )

            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 4)
