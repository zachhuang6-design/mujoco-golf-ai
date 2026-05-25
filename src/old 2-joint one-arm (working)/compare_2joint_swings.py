import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- Baseline arm -->
    <body name="arm1" pos="0 0.35 1.35">
      <joint name="shoulder1" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="arm1_geom" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="club_body1" pos="0 0 -0.42">
        <joint name="wrist1" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="club1" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
        <site name="club_tip1" pos="0.10 0 -0.55" size="0.06" rgba="1 0 0 1"/>
      </body>
    </body>

    <body name="tee1" pos="0.25 0.35 0.30">
      <geom name="tee1_geom" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball1" pos="0.25 0.35 0.48">
      <joint type="free"/>
      <geom name="golf_ball1" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>

    <!-- Best arm -->
    <body name="arm2" pos="0 -0.35 1.35">
      <joint name="shoulder2" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="arm2_geom" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.9 0.3 0.3 1"/>

      <body name="club_body2" pos="0 0 -0.42">
        <joint name="wrist2" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="club2" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
        <site name="club_tip2" pos="0.10 0 -0.55" size="0.06" rgba="0 1 0 1"/>
      </body>
    </body>

    <body name="tee2" pos="0.25 -0.35 0.30">
      <geom name="tee2_geom" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball2" pos="0.25 -0.35 0.48">
      <joint type="free"/>
      <geom name="golf_ball2" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder1" gear="7"/>
    <motor joint="wrist1" gear="5"/>
    <motor joint="shoulder2" gear="7"/>
    <motor joint="wrist2" gear="5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

club_tip1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip1")
club_tip2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip2")
ball1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball1")
ball2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball2")

# Hardcoded baseline swing
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
}

# Best found swing from the random search
best_candidate = {
    "t1": 160,
    "t2": 291,
    "t3": 395,
    "s1": 9.248,
    "w1": 3.372,
    "s2": -8.22,
    "w2": -3.979,
    "s3": -10.0,
    "w3": -0.646,
}

def set_hinge_qpos(joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = value


# Starting pose for both arms.
# Set ONLY the hinge joints, not the free-joint balls.
set_hinge_qpos("shoulder1", 0.0)
set_hinge_qpos("wrist1", 0.0)
set_hinge_qpos("shoulder2", 0.0)
set_hinge_qpos("wrist2", 0.0)

mujoco.mj_forward(model, data)

print("Baseline candidate:", hardcoded_candidate)
print("Best candidate:", best_candidate)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        if step < hardcoded_candidate["t1"]:
            data.ctrl[0] = hardcoded_candidate["s1"]
            data.ctrl[1] = hardcoded_candidate["w1"]
        elif step < hardcoded_candidate["t2"]:
            data.ctrl[0] = hardcoded_candidate["s2"]
            data.ctrl[1] = hardcoded_candidate["w2"]
        elif step < hardcoded_candidate["t3"]:
            data.ctrl[0] = hardcoded_candidate["s3"]
            data.ctrl[1] = hardcoded_candidate["w3"]
        else:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0

        if step < best_candidate["t1"]:
            data.ctrl[2] = best_candidate["s1"]
            data.ctrl[3] = best_candidate["w1"]
        elif step < best_candidate["t2"]:
            data.ctrl[2] = best_candidate["s2"]
            data.ctrl[3] = best_candidate["w2"]
        elif step < best_candidate["t3"]:
            data.ctrl[2] = best_candidate["s3"]
            data.ctrl[3] = best_candidate["w3"]
        else:
            data.ctrl[2] = 0.0
            data.ctrl[3] = 0.0

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(
                f"step={step} | ball1 x={data.xpos[ball1_id][0]:.3f} | ball2 x={data.xpos[ball2_id][0]:.3f}"
            )

        viewer.sync()
        step += 1
        time.sleep(model.opt.timestep * 10)
