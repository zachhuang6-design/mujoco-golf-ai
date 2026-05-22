import math
import time
import mujoco
import mujoco.viewer

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

    <body name="baseline_arm" pos="0 0.20 {shoulder_height}">
      <joint name="baseline_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="baseline_upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="baseline_forearm" pos="0 0 -{upper_arm_len:.6f}">
        <joint name="baseline_elbow" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="baseline_forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1"/>

        <body name="baseline_club" pos="0 0 -{forearm_len:.6f}">
          <joint name="baseline_wrist" type="hinge" axis="0 1 0" range="-100 100" damping="0.45" armature="0.02"/>
          <inertial mass="{shaft_mass + head_mass}" pos="0 0 0" diaginertia="0.0001 0.0001 0.0001"/>
          <geom name="baseline_club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" rgba="0.1 0.1 0.1 1"/>
          <geom name="baseline_club_head_geom" type="box" pos="0 0 -{club_len:.6f}" size="0.04 0.02 0.02" rgba="0.2 0.2 0.2 1"/>
          <site name="baseline_club_tip" pos="0 0 -{club_len:.6f}" size="0.04" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>

    <body name="baseline_tee" pos="{tee_x:.6f} 0.20 0.36">
      <geom name="baseline_tee_geom" type="cylinder" size="0.01 0.08" rgba="1 0.5 0 1"/>
    </body>

    <body name="baseline_ball" pos="{ball_x:.6f} 0.20 {0.36 + 0.08 + ball_radius:.6f}">
      <joint type="free"/>
      <geom name="baseline_ball_geom" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" rgba="1 1 1 1"/>
    </body>

    <body name="opt_arm" pos="0 -0.20 {shoulder_height}">
      <joint name="opt_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="opt_upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="0.9 0.3 0.3 1"/>

      <body name="opt_forearm" pos="0 0 -{upper_arm_len:.6f}">
        <joint name="opt_elbow" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="opt_forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1"/>

        <body name="opt_club" pos="0 0 -{forearm_len:.6f}">
          <joint name="opt_wrist" type="hinge" axis="0 1 0" range="-100 100" damping="0.45" armature="0.02"/>
          <inertial mass="{shaft_mass + head_mass}" pos="0 0 0" diaginertia="0.0001 0.0001 0.0001"/>
          <geom name="opt_club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" rgba="0.1 0.1 0.1 1"/>
          <geom name="opt_club_head_geom" type="box" pos="0 0 -{club_len:.6f}" size="0.04 0.02 0.02" rgba="0.2 0.2 0.2 1"/>
          <site name="opt_club_tip" pos="0 0 -{club_len:.6f}" size="0.04" rgba="0 1 0 1"/>
        </body>
      </body>
    </body>

    <body name="opt_tee" pos="{tee_x:.6f} -0.20 0.36">
      <geom name="opt_tee_geom" type="cylinder" size="0.01 0.08" rgba="1 0.5 0 1"/>
    </body>

    <body name="opt_ball" pos="{ball_x:.6f} -0.20 {0.36 + 0.08 + ball_radius:.6f}">
      <joint type="free"/>
      <geom name="opt_ball_geom" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="baseline_shoulder" gear="7"/>
    <motor joint="baseline_elbow" gear="5"/>
    <motor joint="baseline_wrist" gear="5"/>
    <motor joint="opt_shoulder" gear="7"/>
    <motor joint="opt_elbow" gear="5"/>
    <motor joint="opt_wrist" gear="5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ball_baseline_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "baseline_ball")
ball_opt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "opt_ball")

baseline_candidate = {
    "t1": 150,
    "t2": 275,
    "t3": 375,
    "s1": 7.0,
    "e1": 0.0,
    "w1": 4.0,
    "s2": -5.0,
    "e2": 0.0,
    "w2": -2.0,
    "s3": -10.0,
    "e3": 0.0,
    "w3": -2.0,
}

best_candidate = {
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

# Initialize free bodies from XML defaults.
data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        if step < baseline_candidate["t1"]:
            shoulder = baseline_candidate["s1"]
            elbow = baseline_candidate["e1"]
            wrist = baseline_candidate["w1"]
        elif step < baseline_candidate["t2"]:
            shoulder = baseline_candidate["s2"]
            elbow = baseline_candidate["e2"]
            wrist = baseline_candidate["w2"]
        elif step < baseline_candidate["t3"]:
            shoulder = baseline_candidate["s3"]
            elbow = baseline_candidate["e3"]
            wrist = baseline_candidate["w3"]
        else:
            shoulder = 0.0
            elbow = 0.0
            wrist = 0.0

        data.ctrl[0] = shoulder
        data.ctrl[1] = elbow
        data.ctrl[2] = wrist

        if step < best_candidate["t1"]:
            opt_shoulder = best_candidate["s1"]
            opt_elbow = best_candidate["e1"]
            opt_wrist = best_candidate["w1"]
        elif step < best_candidate["t2"]:
            opt_shoulder = best_candidate["s2"]
            opt_elbow = best_candidate["e2"]
            opt_wrist = best_candidate["w2"]
        elif step < best_candidate["t3"]:
            opt_shoulder = best_candidate["s3"]
            opt_elbow = best_candidate["e3"]
            opt_wrist = best_candidate["w3"]
        else:
            opt_shoulder = 0.0
            opt_elbow = 0.0
            opt_wrist = 0.0

        data.ctrl[3] = opt_shoulder
        data.ctrl[4] = opt_elbow
        data.ctrl[5] = opt_wrist

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(
                f"step={step} | baseline_x={data.xpos[ball_baseline_id][0]:.3f} | opt_x={data.xpos[ball_opt_id][0]:.3f}"
            )

        viewer.sync()
        step += 1
        time.sleep(model.opt.timestep * 4)
