import random
import mujoco

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

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")


def run_swing(backswing_torque, downswing_torque):
    data = mujoco.MjData(model)

    for step in range(1000):
        if step < 250:
            data.ctrl[0] = backswing_torque
        elif step < 700:
            data.ctrl[0] = downswing_torque
        else:
            data.ctrl[0] = 0.0

        mujoco.mj_step(model, data)

    final_ball_x = data.xpos[ball_id][0]
    return final_ball_x


best_score = -999
best_backswing = None
best_downswing = None

for trial in range(100):
    backswing = random.uniform(-5.0, 0.0)
    downswing = random.uniform(0.0, 10.0)

    score = run_swing(backswing, downswing)

    if score > best_score:
        best_score = score
        best_backswing = backswing
        best_downswing = downswing

    print(
        "Trial:",
        trial,
        "Score:",
        round(score, 3),
        "Backswing:",
        round(backswing, 3),
        "Downswing:",
        round(downswing, 3),
    )

print()
print("Best score:", best_score)
print("Best backswing torque:", best_backswing)
print("Best downswing torque:", best_downswing)
