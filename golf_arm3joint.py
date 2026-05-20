import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- A simplified upper body/arm system in the x-z plane -->
    <body name="upper_arm" pos="0 0 1.45">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-120 120" damping="2.0" armature="0.05"/>
      <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="forearm" pos="0 0 -0.35">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-100 60" damping="1.5" armature="0.03"/>
        <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.035" rgba="0.3 0.9 0.3 1"/>

        <body name="hand_club" pos="0 0 -0.30">
          <joint name="wrist" type="hinge" axis="0 1 0" range="-100 100" damping="1.0" armature="0.02"/>
          <geom name="club" type="capsule" fromto="0 0 0 0 0 -0.70" size="0.025" rgba="0.1 0.1 0.1 1"/>
          <site name="club_tip" pos="0 0 -0.70" size="0.06" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>

    <!-- Put the ball closer and slightly larger for debugging -->
    <body name="ball" pos="0.45 0 0.08">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="2"/>
    <motor joint="elbow" gear="2"/>
    <motor joint="wrist" gear="1.5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

# Initial pose: start with the arm pulled back instead of straight down.
# qpos[0] = shoulder angle
# qpos[1] = elbow angle
# qpos[2] = wrist angle
data.qpos[0] = -0.75
data.qpos[1] = 0.45
data.qpos[2] = 0.25

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running() and step < 1200:
        # Phase 1: hold / slight backswing
        if step < 200:
            data.ctrl[0] = -0.5   # shoulder
            data.ctrl[1] = 0.2    # elbow
            data.ctrl[2] = 0.1    # wrist

        # Phase 2: main downswing
        elif step < 520:
            data.ctrl[0] = 2.8
            data.ctrl[1] = -1.2
            data.ctrl[2] = -0.4

        # Phase 3: wrist release / club snap
        elif step < 760:
            data.ctrl[0] = 1.2
            data.ctrl[1] = -0.5
            data.ctrl[2] = 2.5

        # Phase 4: follow-through
        else:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = 0.0

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            tip_x = data.site_xpos[club_tip_id][0]
            tip_z = data.site_xpos[club_tip_id][2]
            print("step:", step, "club tip x:", round(tip_x, 3), "z:", round(tip_z, 3))

        viewer.sync()

        step += 1
        time.sleep(model.opt.timestep * 3)
