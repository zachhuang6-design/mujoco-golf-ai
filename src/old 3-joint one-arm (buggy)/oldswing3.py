import math
import time
import mujoco
import mujoco.viewer

# 1 inch = 0.0254 meters
INCH = 0.0254

upper_arm_len = 13 * INCH
forearm_len = 13 * INCH
club_len = 36 * INCH
ball_radius = 0.84 * INCH
shaft_mass = 0.100
head_mass = 0.270
ball_mass = 0.046

shoulder_height = 2.05
club_tip_x = 0.0
club_tip_z_drop = 0.0
ball_x = 0.05
tee_x = ball_x
club_angle = 0.0

xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="arm" pos="0 0 {shoulder_height}">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="forearm" pos="0 0 -{upper_arm_len:.6f}">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1"/>

        <body name="club" pos="0 0 -{forearm_len:.6f}">
          <joint name="wrist" type="hinge" axis="0 1 0" range="-100 100" damping="0.45" armature="0.02"/>
          <inertial mass="{shaft_mass + head_mass}" pos="0 0 0" diaginertia="0.0001 0.0001 0.0001"/>
          <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" rgba="0.1 0.1 0.1 1"/>
          <geom name="club_head_geom" type="box" pos="0 0 -{club_len:.6f}" size="0.04 0.02 0.02" rgba="0.2 0.2 0.2 1"/>
          <site name="club_tip" pos="0 0 -{club_len:.6f}" size="0.04" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>

    <body name="tee" pos="{tee_x:.6f} 0 0.36">
      <geom name="tee_geom" type="cylinder" size="0.01 0.08" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball" pos="{ball_x:.6f} 0 {0.36 + 0.08 + ball_radius:.6f}">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="7"/>
    <motor joint="elbow" gear="5"/>
    <motor joint="wrist" gear="5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

# Neutral starting pose
for i in range(model.nq):
    data.qpos[i] = model.qpos0[i]

data.qpos[0] = 0.0
data.qpos[1] = 0.0
mujoco.mj_forward(model, data)

if __name__ == '__main__':
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            # Three actuators are defined in this model: shoulder (ctrl[0]), elbow (ctrl[1]), and wrist (ctrl[2]).
            if step < 204:
                data.ctrl[0] = 7.610
                data.ctrl[1] = 4.000
                data.ctrl[2] = 3.580

            elif step < 313:
                data.ctrl[0] = -8.737
                data.ctrl[1] = -2.344
                data.ctrl[2] = -4.110

            elif step < 462:
                data.ctrl[0] = -10.000
                data.ctrl[1] = -0.186
                data.ctrl[2] = -4.704

            else:
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = 0.0

            # Advance the simulation every loop iteration so the controls take effect.
            mujoco.mj_step(model, data)

            if step % 100 == 0:
                tip_x = data.site_xpos[club_tip_id][0]
                tip_z = data.site_xpos[club_tip_id][2]
                ball_x = data.xpos[ball_id][0]
                print(
                    "step:", step,
                    "club tip x:", round(tip_x, 3),
                    "z:", round(tip_z, 3),
                    "ball x:", round(ball_x, 3),
                )

            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep * 15)
