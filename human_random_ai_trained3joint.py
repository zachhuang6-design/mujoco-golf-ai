import math
import random
import mujoco

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

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

INITIAL_BALL_X = ball_x
MAX_SHOULDER_TORQUE = 10.0
MAX_ELBOW_TORQUE = 4.0
MAX_WRIST_TORQUE = 4.0
EPISODE_STEPS = 1200
NUM_TRIALS = 10000
random.seed(0)


def clip(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def random_near(base, spread, min_value, max_value):
    return clip(random.uniform(base - spread, base + spread), min_value, max_value)


def make_candidate():
    t1 = random.randint(100, 220)
    t2 = random.randint(t1 + 60, t1 + 180)
    t3 = random.randint(t2 + 60, t2 + 180)

    s1 = random_near(7.0, 3.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    e1 = random_near(0.0, 2.0, -MAX_ELBOW_TORQUE, MAX_ELBOW_TORQUE)
    w1 = random_near(4.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    s2 = random_near(-5.0, 4.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    e2 = random_near(0.0, 2.0, -MAX_ELBOW_TORQUE, MAX_ELBOW_TORQUE)
    w2 = random_near(-2.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)
    s3 = random_near(-10.0, 3.0, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    e3 = random_near(0.0, 2.0, -MAX_ELBOW_TORQUE, MAX_ELBOW_TORQUE)
    w3 = random_near(-2.0, 2.0, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)

    return {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "s1": s1,
        "e1": e1,
        "w1": w1,
        "s2": s2,
        "e2": e2,
        "w2": w2,
        "s3": s3,
        "e3": e3,
        "w3": w3,
    }


def get_torques(candidate, step):
    if step < candidate["t1"]:
        shoulder = candidate["s1"]
        elbow = candidate["e1"]
        wrist = candidate["w1"]
    elif step < candidate["t2"]:
        shoulder = candidate["s2"]
        elbow = candidate["e2"]
        wrist = candidate["w2"]
    elif step < candidate["t3"]:
        shoulder = candidate["s3"]
        elbow = candidate["e3"]
        wrist = candidate["w3"]
    else:
        shoulder = 0.0
        elbow = 0.0
        wrist = 0.0
    return (
        clip(shoulder, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE),
        clip(elbow, -MAX_ELBOW_TORQUE, MAX_ELBOW_TORQUE),
        clip(wrist, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE),
    )


def simulate_swing(candidate):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)

    max_ball_x = INITIAL_BALL_X
    for step in range(EPISODE_STEPS):
        shoulder, elbow, wrist = get_torques(candidate, step)
        data.ctrl[0] = shoulder
        data.ctrl[1] = elbow
        data.ctrl[2] = wrist
        mujoco.mj_step(model, data)
        ball_x = data.xpos[ball_id][0]
        if ball_x > max_ball_x:
            max_ball_x = ball_x
    return max_ball_x - INITIAL_BALL_X


def print_candidate(candidate):
    print("t1:", candidate["t1"])
    print("t2:", candidate["t2"])
    print("t3:", candidate["t3"])
    print(
        "s1:", round(candidate["s1"], 3),
        "e1:", round(candidate["e1"], 3),
        "w1:", round(candidate["w1"], 3),
    )
    print(
        "s2:", round(candidate["s2"], 3),
        "e2:", round(candidate["e2"], 3),
        "w2:", round(candidate["w2"], 3),
    )
    print(
        "s3:", round(candidate["s3"], 3),
        "e3:", round(candidate["e3"], 3),
        "w3:", round(candidate["w3"], 3),
    )


def print_copy_paste_controls(candidate):
    print("\n# Copy these control assignments into human_golf_arm2joint.py:\n")
    print("if step < {}:".format(candidate["t1"]))
    print("    data.ctrl[0] = {}".format(round(candidate["s1"], 3)))
    print("    data.ctrl[1] = {}".format(round(candidate["e1"], 3)))
    print("    data.ctrl[2] = {}".format(round(candidate["w1"], 3)))
    print("elif step < {}:".format(candidate["t2"]))
    print("    data.ctrl[0] = {}".format(round(candidate["s2"], 3)))
    print("    data.ctrl[1] = {}".format(round(candidate["e2"], 3)))
    print("    data.ctrl[2] = {}".format(round(candidate["w2"], 3)))
    print("elif step < {}:".format(candidate["t3"]))
    print("    data.ctrl[0] = {}".format(round(candidate["s3"], 3)))
    print("    data.ctrl[1] = {}".format(round(candidate["e3"], 3)))
    print("    data.ctrl[2] = {}".format(round(candidate["w3"], 3)))
    print("else:")
    print("    data.ctrl[0] = 0.0")
    print("    data.ctrl[1] = 0.0")
    print("    data.ctrl[2] = 0.0")


def search():
    hardcoded = {
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
    hardcoded_reward = simulate_swing(hardcoded)
    print("Hardcoded baseline reward:", round(hardcoded_reward, 4))

    best_reward = hardcoded_reward
    best_candidate = hardcoded

    for trial in range(1, NUM_TRIALS + 1):
        candidate = make_candidate()
        reward = simulate_swing(candidate)
        if reward > best_reward:
            best_reward = reward
            best_candidate = candidate
            print("New best at trial", trial, "reward", round(best_reward, 4))
            print_candidate(best_candidate)
        if trial % 100 == 0:
            print("Trial", trial, "best reward", round(best_reward, 4))

    print("Training complete")
    print("Final best reward:", round(best_reward, 4))
    print("Improvement:", round(best_reward - hardcoded_reward, 4))
    print_candidate(best_candidate)
    print_copy_paste_controls(best_candidate)


if __name__ == "__main__":
    search()
