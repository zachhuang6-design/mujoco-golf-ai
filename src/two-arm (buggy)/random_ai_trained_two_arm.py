import random
import mujoco

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

    <!-- Support left arm, visual partner -->
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
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

# Training settings
INITIAL_BALL_X = 0.25
MAX_SHOULDER_TORQUE = 10.0
MAX_WRIST_TORQUE = 4.0
EPISODE_STEPS = 1200
NUM_TRIALS = 1000
random.seed(0)


def clip(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def random_near(base, spread, min_value, max_value):
    value = random.uniform(base - spread, base + spread)
    return clip(value, min_value, max_value)


def make_candidate():
    t1 = random.randint(100, 220)
    t2 = random.randint(t1 + 60, t1 + 180)
    t3 = random.randint(t2 + 60, t2 + 180)

    s1 = random_near(7.0, 3.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    w1 = random_near(4.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    s2 = random_near(-5.0, 4.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    w2 = random_near(-2.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    s3 = random_near(-10.0, 3.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    w3 = random_near(-2.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    support_shoulder_scale = random_near(0.5, 0.25, 0.0, 1.0)
    support_wrist_scale = random_near(0.4, 0.25, 0.0, 1.0)

    return {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "s1": s1,
        "w1": w1,
        "s2": s2,
        "w2": w2,
        "s3": s3,
        "w3": w3,
        "support_shoulder_scale": support_shoulder_scale,
        "support_wrist_scale": support_wrist_scale,
    }


def get_torques(candidate, step):
    if step < candidate["t1"]:
        shoulder = candidate["s1"]
        wrist = candidate["w1"]
    elif step < candidate["t2"]:
        shoulder = candidate["s2"]
        wrist = candidate["w2"]
    elif step < candidate["t3"]:
        shoulder = candidate["s3"]
        wrist = candidate["w3"]
    else:
        shoulder = 0.0
        wrist = 0.0

    shoulder = clip(shoulder, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    wrist = clip(wrist, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    return shoulder, wrist


def simulate_swing(candidate):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder")]] = 0.0
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist")]] = 0.0
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder")]] = 0.0
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist")]] = 0.0
    mujoco.mj_forward(model, data)

    max_ball_x = INITIAL_BALL_X

    for step in range(EPISODE_STEPS):
        shoulder, wrist = get_torques(candidate, step)
        support_shoulder = candidate["support_shoulder_scale"] * shoulder
        support_wrist = candidate["support_wrist_scale"] * wrist

        data.ctrl[0] = shoulder
        data.ctrl[1] = wrist
        data.ctrl[2] = support_shoulder
        data.ctrl[3] = support_wrist

        mujoco.mj_step(model, data)

        ball_x = data.xpos[ball_id][0]
        if ball_x > max_ball_x:
            max_ball_x = ball_x

    return max_ball_x - INITIAL_BALL_X


def print_candidate(candidate):
    print("t1:", candidate["t1"])
    print("t2:", candidate["t2"])
    print("t3:", candidate["t3"])
    print("s1:", round(candidate["s1"], 3), "w1:", round(candidate["w1"], 3))
    print("s2:", round(candidate["s2"], 3), "w2:", round(candidate["w2"], 3))
    print("s3:", round(candidate["s3"], 3), "w3:", round(candidate["w3"], 3))
    print("support_shoulder_scale:", round(candidate["support_shoulder_scale"], 3))
    print("support_wrist_scale:", round(candidate["support_wrist_scale"], 3))


def print_code_to_paste(candidate):
    print()
    print("============================================================")
    print("Paste this control block into your visual script:")
    print("============================================================")
    print()
    print(f"if step < {candidate['t1']}:\n"
          f"    data.ctrl[0] = {candidate['s1']:.3f}\n"
          f"    data.ctrl[1] = {candidate['w1']:.3f}\n"
          f"    data.ctrl[2] = {candidate['support_shoulder_scale'] * candidate['s1']:.3f}\n"
          f"    data.ctrl[3] = {candidate['support_wrist_scale'] * candidate['w1']:.3f}\n\n"
          f"elif step < {candidate['t2']}:\n"
          f"    data.ctrl[0] = {candidate['s2']:.3f}\n"
          f"    data.ctrl[1] = {candidate['w2']:.3f}\n"
          f"    data.ctrl[2] = {candidate['support_shoulder_scale'] * candidate['s2']:.3f}\n"
          f"    data.ctrl[3] = {candidate['support_wrist_scale'] * candidate['w2']:.3f}\n\n"
          f"elif step < {candidate['t3']}:\n"
          f"    data.ctrl[0] = {candidate['s3']:.3f}\n"
          f"    data.ctrl[1] = {candidate['w3']:.3f}\n"
          f"    data.ctrl[2] = {candidate['support_shoulder_scale'] * candidate['s3']:.3f}\n"
          f"    data.ctrl[3] = {candidate['support_wrist_scale'] * candidate['w3']:.3f}\n\n"
          f"else:\n"
          f"    data.ctrl[0] = 0.0\n"
          f"    data.ctrl[1] = 0.0\n"
          f"    data.ctrl[2] = 0.0\n"
          f"    data.ctrl[3] = 0.0")


def random_search():
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

    hardcoded_reward = simulate_swing(hardcoded_candidate)
    print("Hardcoded baseline reward:", round(hardcoded_reward, 4))

    best_reward = hardcoded_reward
    best_candidate = hardcoded_candidate

    for trial in range(1, NUM_TRIALS + 1):
        candidate = make_candidate()
        reward = simulate_swing(candidate)

        if reward > best_reward:
            best_reward = reward
            best_candidate = candidate
            print("New best found at trial", trial, "reward", round(best_reward, 4))
            print_candidate(best_candidate)
            print_code_to_paste(best_candidate)

        if trial % 100 == 0:
            print("Trial", trial, "best reward so far", round(best_reward, 4))

    print("Training complete")
    print("Final best reward:", round(best_reward, 4))
    print("Improvement over hardcoded:", round(best_reward - hardcoded_reward, 4))


if __name__ == "__main__":
    random_search()
