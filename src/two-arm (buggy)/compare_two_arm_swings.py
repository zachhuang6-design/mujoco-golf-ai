import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- Baseline two-arm system -->
    <body name="baseline_right_arm" pos="0 0.35 1.35">
      <joint name="baseline_right_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="baseline_right_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="baseline_right_forearm" pos="0 0 -0.42">
        <joint name="baseline_right_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="baseline_right_club" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
      </body>
    </body>

    <body name="baseline_left_arm" pos="0 -0.18 1.35">
      <joint name="baseline_left_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="baseline_left_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.9 0.3 0.3 1"/>
      <body name="baseline_left_forearm" pos="0 0 -0.42">
        <joint name="baseline_left_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="baseline_left_club_visual" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.3 0.3 0.3 0.6"/>
      </body>
    </body>

    <body name="baseline_tee" pos="0.25 0.35 0.30">
      <geom name="baseline_tee_geom" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <body name="baseline_ball" pos="0.25 0.35 0.48">
      <joint type="free"/>
      <geom name="baseline_ball_geom" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>

    <!-- Optimized two-arm system -->
    <body name="opt_right_arm" pos="0 -0.35 1.35">
      <joint name="opt_right_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="opt_right_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.3 0.3 0.9 1"/>

      <body name="opt_right_forearm" pos="0 0 -0.42">
        <joint name="opt_right_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="opt_right_club" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.05 0.05 0.05 1"/>
      </body>
    </body>

    <body name="opt_left_arm" pos="0 -0.53 1.35">
      <joint name="opt_left_shoulder" type="hinge" axis="0 1 0" range="-160 160" damping="0.8" armature="0.04"/>
      <geom name="opt_left_upper_arm" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.04" rgba="0.9 0.3 0.3 1"/>
      <body name="opt_left_forearm" pos="0 0 -0.42">
        <joint name="opt_left_wrist" type="hinge" axis="0 1 0" range="-150 150" damping="0.45" armature="0.02"/>
        <geom name="opt_left_club_visual" type="capsule" fromto="0 0 0 0.10 0 -0.55" size="0.025" rgba="0.3 0.3 0.3 0.6"/>
      </body>
    </body>

    <body name="opt_tee" pos="0.25 -0.35 0.30">
      <geom name="opt_tee_geom" type="cylinder" size="0.02 0.10" rgba="1 0.5 0 1"/>
    </body>

    <body name="opt_ball" pos="0.25 -0.35 0.48">
      <joint type="free"/>
      <geom name="opt_ball_geom" type="sphere" size="0.08" mass="0.045" rgba="1 1 1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="baseline_right_shoulder" gear="7"/>
    <motor joint="baseline_right_wrist" gear="5"/>
    <motor joint="baseline_left_shoulder" gear="2"/>
    <motor joint="baseline_left_wrist" gear="1"/>
    <motor joint="opt_right_shoulder" gear="7"/>
    <motor joint="opt_right_wrist" gear="5"/>
    <motor joint="opt_left_shoulder" gear="2"/>
    <motor joint="opt_left_wrist" gear="1"/>
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
    "w1": 4.0,
    "s2": -5.0,
    "w2": -2.0,
    "s3": -10.0,
    "w3": -2.0,
    "support_shoulder_scale": 0.5,
    "support_wrist_scale": 0.4,
}

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
    "support_shoulder_scale": 0.5,
    "support_wrist_scale": 0.4,
}

# Initialize free bodies from XML default positions.
data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)

print("Baseline candidate:", baseline_candidate)
print("Best candidate:", best_candidate)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        if step < baseline_candidate["t1"]:
            baseline_right_shoulder = baseline_candidate["s1"]
            baseline_right_wrist = baseline_candidate["w1"]
        elif step < baseline_candidate["t2"]:
            baseline_right_shoulder = baseline_candidate["s2"]
            baseline_right_wrist = baseline_candidate["w2"]
        elif step < baseline_candidate["t3"]:
            baseline_right_shoulder = baseline_candidate["s3"]
            baseline_right_wrist = baseline_candidate["w3"]
        else:
            baseline_right_shoulder = 0.0
            baseline_right_wrist = 0.0

        baseline_left_shoulder = baseline_candidate["support_shoulder_scale"] * baseline_right_shoulder
        baseline_left_wrist = baseline_candidate["support_wrist_scale"] * baseline_right_wrist

        if step < best_candidate["t1"]:
            opt_right_shoulder = best_candidate["s1"]
            opt_right_wrist = best_candidate["w1"]
        elif step < best_candidate["t2"]:
            opt_right_shoulder = best_candidate["s2"]
            opt_right_wrist = best_candidate["w2"]
        elif step < best_candidate["t3"]:
            opt_right_shoulder = best_candidate["s3"]
            opt_right_wrist = best_candidate["w3"]
        else:
            opt_right_shoulder = 0.0
            opt_right_wrist = 0.0

        opt_left_shoulder = best_candidate["support_shoulder_scale"] * opt_right_shoulder
        opt_left_wrist = best_candidate["support_wrist_scale"] * opt_right_wrist

        data.ctrl[0] = baseline_right_shoulder
        data.ctrl[1] = baseline_right_wrist
        data.ctrl[2] = baseline_left_shoulder
        data.ctrl[3] = baseline_left_wrist
        data.ctrl[4] = opt_right_shoulder
        data.ctrl[5] = opt_right_wrist
        data.ctrl[6] = opt_left_shoulder
        data.ctrl[7] = opt_left_wrist

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(
                f"step={step} | baseline_x={data.xpos[ball_baseline_id][0]:.3f} | opt_x={data.xpos[ball_opt_id][0]:.3f}"
            )

        viewer.sync()
        step += 1
        time.sleep(model.opt.timestep * 4)
