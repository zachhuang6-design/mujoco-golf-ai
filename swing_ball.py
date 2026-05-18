import time
import math
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="arm" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-120 120"/>
      <geom name="club" type="capsule" fromto="0 0 0 0 0 -0.8" size="0.035" rgba="0.2 0.2 0.2 1"/>
    </body>

    <body name="ball" pos="0.35 0 0.08">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="0.1" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running() and step < 1000:
        # Apply a simple timed swing torque
        if step < 250:
            data.ctrl[0] = -4.4
        elif step < 700:
            data.ctrl[0] = 1.3
        else:
            data.ctrl[0] = 0.0

        mujoco.mj_step(model, data)
        #ball_x = data.xpos[ball_id][0]
        #print("Ball x position:", ball_x)
        viewer.sync()

        step += 1
        time.sleep(model.opt.timestep * 3)
