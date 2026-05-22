import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- Dominant right arm, carries the club -->
    <body name="right_arm" pos="0 0 1.35">
      <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="right_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="right_forearm" pos="0 0 -0.42">
        <joint name="right_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="right_club" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
        <site name="right_club_tip" pos="0.10 0 -0.55" size="0.06" rgba="1 0 0 1"/>
      </body>
    </body>

    <!-- Support left arm, visual/passive partner -->
    <body name="left_arm" pos="0 -0.18 1.35">
      <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="left_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.9 0.3 0.3 1"/>

      <body name="left_forearm" pos="0 0 -0.42">
        <joint name="left_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="left_club_visual" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.3 0.3 0.3 0.6"/>
      </body>
    </body>

    <body name="tee" pos="0.25 0 0.30">
      <geom name="tee_geom" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball" pos="0.25 0 0.48">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="right_shoulder" gear="7"/>
    <motor joint="right_wrist" gear="5"/>
    <motor joint="left_shoulder" gear="2"/>
    <motor joint="left_wrist" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

hardcoded_candidate = {
    "t1": 150,
    "t2": 275,
    "t3": 375,
    "s1": 7.0,
    "w1": 4.0,
    "s2": -5.0,
    "w2": -2.0,
    "s3": -10.0,
    "w3": -2.0,
    "support_shoulder_scale": 0.5,
    "support_wrist_scale": 0.4,
}

# Initialize all joint qpos from XML defaults.
data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)

print("Running two-arm visual swing")
print(hardcoded_candidate)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        if step < hardcoded_candidate["t1"]:
            right_shoulder = hardcoded_candidate["s1"]
            right_wrist = hardcoded_candidate["w1"]
        elif step < hardcoded_candidate["t2"]:
            right_shoulder = hardcoded_candidate["s2"]
            right_wrist = hardcoded_candidate["w2"]
        elif step < hardcoded_candidate["t3"]:
            right_shoulder = hardcoded_candidate["s3"]
            right_wrist = hardcoded_candidate["w3"]
        else:
            right_shoulder = 0.0
            right_wrist = 0.0

        support_shoulder = hardcoded_candidate["support_shoulder_scale"] * right_shoulder
        support_wrist = hardcoded_candidate["support_wrist_scale"] * right_wrist

        data.ctrl[0] = right_shoulder
        data.ctrl[1] = right_wrist
        data.ctrl[2] = support_shoulder
        data.ctrl[3] = support_wrist

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(
                f"step={step} | ball x={data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'ball')][0]:.3f}"
            )

        viewer.sync()
        step += 1
        time.sleep(model.opt.timestep * 4)
