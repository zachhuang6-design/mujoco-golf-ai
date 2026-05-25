import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- Two-joint model:
         shoulder = main arm swing
         wrist = club release
    -->
    <body name="arm" pos="0 0 1.35">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="club_body" pos="0 0 -0.42">
        <joint name="wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>

        <!-- Shorter club for now so it does not immediately dig into the floor -->
        <geom name="club" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
        <site name="club_tip" pos="0.10 0 -0.55" size="0.06" rgba="1 0 0 1"/>
      </body>
    </body>

    <body name="tee" pos="0.25 0 0.30">
      <geom name="tee" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <!-- Ball farther forward so it is hit later, not immediately -->
    <body name="ball" pos="0.25 0 0.48">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="7"/>
    <motor joint="wrist" gear="5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Initial pose:
# qpos[0] = shoulder angle
# qpos[1] = elbow angle
#
# This starts the club at neutral address position.
data.qpos[0] = 0.0
data.qpos[1] = 0.0

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        # backswing torque setup
        if step < 150:
            data.ctrl[0] = 7.0
            data.ctrl[1] = 4.0
        # downswing torque setup (initial)
        elif step < 275:
            data.ctrl[0] = -5.0
            data.ctrl[1] = -2.0

        # downswing torque correction for shoulder lead
        elif step < 375:
            data.ctrl[0] = -10.0
            data.ctrl[1] = -2.0
        #zero torque natural release followthrough
        else:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            tip_x = data.site_xpos[club_tip_id][0]
            tip_z = data.site_xpos[club_tip_id][2]
            ball_x = data.xpos[ball_id][0]
            print(
                "step:",
                step,
                "club tip x:",
                round(tip_x, 3),
                "z:",
                round(tip_z, 3),
                "ball x:",
                round(ball_x, 3),
            )

        viewer.sync()
        step += 1

        # for slower viewing speed, change (model.opt.timestep * n) for n times slower than real time
        time.sleep(model.opt.timestep * 5)