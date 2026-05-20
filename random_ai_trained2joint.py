import random
import mujoco

# ============================================================
# MuJoCo XML: copied from your working 2-joint swing setup
# ============================================================

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

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

# ============================================================
# Training settings
# ============================================================

INITIAL_BALL_X = 0.25

# Your requirement:
# The AI cannot fake better performance by using unlimited torque.
# All torques get clipped to these limits.
MAX_SHOULDER_TORQUE = 10.0
MAX_WRIST_TORQUE = 4.0

# Number of physics steps per swing.
# Your original swing stops applying torque after step 375,
# but the ball needs time to fly/roll after impact.
EPISODE_STEPS = 1200

# Number of random swings to try.
# Start with 1000. Later you can try 5000 or 10000.
NUM_TRIALS = 1000

# Set this for repeatability.
random.seed(0)


# ============================================================
# Helper functions
# ============================================================

def clip(value, min_value, max_value):
    """Keep value inside [min_value, max_value]."""
    return max(min_value, min(max_value, value))


def random_near(base, spread, min_value, max_value):
    """
    Sample a random number near a base value, then clip it.

    Example:
        random_near(7.0, 2.0, -10.0, 10.0)
    samples around 7.0, usually between about 5 and 9,
    but never outside -10 to 10.
    """
    value = random.uniform(base - spread, base + spread)
    return clip(value, min_value, max_value)


def make_candidate():
    """
    Create one candidate swing.

    This searches near your working hand-coded swing:

        if step < 150:
            shoulder = 7.0
            wrist = 4.0
        elif step < 275:
            shoulder = -5.0
            wrist = -2.0
        elif step < 375:
            shoulder = -10.0
            wrist = -2.0

    The AI is allowed to adjust:
    - when each phase ends
    - shoulder torque in each phase
    - wrist torque in each phase

    But it cannot exceed the torque limits.
    """

    # Phase timing.
    # Keep these ordered: t1 < t2 < t3.
    t1 = random.randint(100, 220)
    t2 = random.randint(t1 + 60, t1 + 180)
    t3 = random.randint(t2 + 60, t2 + 180)

    # Phase 1: backswing.
    # Your original: shoulder = 7.0, wrist = 4.0
    s1 = random_near(
        base=7.0,
        spread=3.0,
        min_value=-MAX_SHOULDER_TORQUE,
        max_value=MAX_SHOULDER_TORQUE,
    )
    w1 = random_near(
        base=4.0,
        spread=2.0,
        min_value=-MAX_WRIST_TORQUE,
        max_value=MAX_WRIST_TORQUE,
    )

    # Phase 2: initial downswing.
    # Your original: shoulder = -5.0, wrist = -2.0
    s2 = random_near(
        base=-5.0,
        spread=4.0,
        min_value=-MAX_SHOULDER_TORQUE,
        max_value=MAX_SHOULDER_TORQUE,
    )
    w2 = random_near(
        base=-2.0,
        spread=2.0,
        min_value=-MAX_WRIST_TORQUE,
        max_value=MAX_WRIST_TORQUE,
    )

    # Phase 3: shoulder lead / acceleration.
    # Your original: shoulder = -10.0, wrist = -2.0
    s3 = random_near(
        base=-10.0,
        spread=3.0,
        min_value=-MAX_SHOULDER_TORQUE,
        max_value=MAX_SHOULDER_TORQUE,
    )
    w3 = random_near(
        base=-2.0,
        spread=2.0,
        min_value=-MAX_WRIST_TORQUE,
        max_value=MAX_WRIST_TORQUE,
    )

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
    }


def get_torques(candidate, step):
    """
    Given a candidate swing and the current step,
    return the shoulder and wrist torques.
    """

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

    # Safety clipping.
    # This guarantees torque limits are never violated.
    shoulder = clip(shoulder, -MAX_SHOULDER_TORQUE, MAX_SHOULDER_TORQUE)
    wrist = clip(wrist, -MAX_WRIST_TORQUE, MAX_WRIST_TORQUE)

    return shoulder, wrist


def simulate_swing(candidate):
    """
    Run one invisible MuJoCo simulation and return the reward.

    Reward = final ball x-position - initial ball x-position.

    Bigger reward means the ball traveled farther in the +x direction.
    """

    data = mujoco.MjData(model)

    # Same starting pose as your working hardcoded swing.
    data.qpos[0] = 0.0  # shoulder
    data.qpos[1] = 0.0  # wrist

    mujoco.mj_forward(model, data)

    max_ball_x = INITIAL_BALL_X

    for step in range(EPISODE_STEPS):
        shoulder, wrist = get_torques(candidate, step)

        data.ctrl[0] = shoulder
        data.ctrl[1] = wrist

        mujoco.mj_step(model, data)

        ball_x = data.xpos[ball_id][0]
        if ball_x > max_ball_x:
            max_ball_x = ball_x

    # Use max_ball_x instead of final ball_x.
    # This is useful because sometimes the ball moves forward, then bounces back.
    distance = max_ball_x - INITIAL_BALL_X

    return distance


def print_candidate(candidate):
    """Pretty-print a candidate swing."""
    print("t1:", candidate["t1"])
    print("t2:", candidate["t2"])
    print("t3:", candidate["t3"])
    print("s1:", round(candidate["s1"], 3), "w1:", round(candidate["w1"], 3))
    print("s2:", round(candidate["s2"], 3), "w2:", round(candidate["w2"], 3))
    print("s3:", round(candidate["s3"], 3), "w3:", round(candidate["w3"], 3))


def print_code_to_paste(candidate):
    """
    Print the if/elif block that you can paste back into your visual file.
    """
    print()
    print("============================================================")
    print("Paste this control block into your visual script:")
    print("============================================================")
    print()
    print(f"""if step < {candidate["t1"]}:
    data.ctrl[0] = {candidate["s1"]:.3f}
    data.ctrl[1] = {candidate["w1"]:.3f}

elif step < {candidate["t2"]}:
    data.ctrl[0] = {candidate["s2"]:.3f}
    data.ctrl[1] = {candidate["w2"]:.3f}

elif step < {candidate["t3"]}:
    data.ctrl[0] = {candidate["s3"]:.3f}
    data.ctrl[1] = {candidate["w3"]:.3f}

else:
    data.ctrl[0] = 0.0
    data.ctrl[1] = 0.0
""")


# ============================================================
# Hardcoded baseline swing
# ============================================================

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

hardcoded_reward = simulate_swing(hardcoded_candidate)

print("============================================================")
print("Hardcoded baseline swing")
print("============================================================")
print("Hardcoded x-distance:", round(hardcoded_reward, 4))
print_candidate(hardcoded_candidate)
print()


# ============================================================
# Main training loop
# ============================================================

best_reward = hardcoded_reward
best_candidate = hardcoded_candidate

trial = 0

print("Starting random AI training...")
print("Goal: find a swing better than the hardcoded baseline.")
print("Torque limits:")
print("  shoulder:", -MAX_SHOULDER_TORQUE, "to", MAX_SHOULDER_TORQUE)
print("  wrist:", -MAX_WRIST_TORQUE, "to", MAX_WRIST_TORQUE)
print()

while True:
    candidate = make_candidate()
    reward = simulate_swing(candidate)

    trial += 1

    if reward > best_reward:
        best_reward = reward
        best_candidate = candidate

        print("Better-than-baseline swing found!")
        print("Trial:", trial)
        print("Hardcoded x-distance:", round(hardcoded_reward, 4))
        print("New x-distance:", round(best_reward, 4))
        print("Improvement:", round(best_reward - hardcoded_reward, 4))
        print_candidate(best_candidate)
        print_code_to_paste(best_candidate)
        break

    if trial % 100 == 0:
        print(
            "Completed trial:",
            trial,
            "| Hardcoded:",
            round(hardcoded_reward, 4),
            "| Best random so far:",
            round(best_reward, 4),
        )