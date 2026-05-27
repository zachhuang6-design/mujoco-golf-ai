"""Joint-accurate two-arm MuJoCo golf model and baseline controller.

This is the first physics-first version of the two-arm swing. Unlike the
current mocap visual sketches, this model uses actual MuJoCo joints,
actuators, segment masses, club masses, a free ball, and bounded torques.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from golf_core.common import (
    BALL_CONTACT_ATTRS,
    CLUB_CONTACT_ATTRS,
    ball_mass,
    ball_radius,
    get_club_preset,
    normalize_club_name,
    tee_half_height,
)


DEFAULT_CLUB = "7iron"
DEFAULT_HAND = "right"

INCH = 0.0254
CHEST_WIDTH = 20.0 * INCH
# Match the accepted kinematic arm proportions instead of generic 13"/12".
# This removes a major source of IK compensation and elbow/hand drift.
UPPER_ARM_LEN = 0.3444
FOREARM_LEN = 0.2732

# Match the accepted kinematic address pose. The jointed model should start in
# the same world-space neighborhood before IK tries to solve the moving poses.
CHEST_POS = (0.105, 0.46, 1.688)
RIGHT_SHOULDER_OFFSET = (-CHEST_WIDTH * 0.5, 0.0, -0.032)
LEFT_SHOULDER_OFFSET = (CHEST_WIDTH * 0.5, 0.0, 0.032)
SPINE_AXIS = (0.0, -math.sin(math.radians(35.0)), math.cos(math.radians(35.0)))

ADDRESS_LEFT_WRIST = (0.105, 0.398, 1.165)
ADDRESS_RIGHT_WRIST = (0.105, 0.360, 1.099)
ADDRESS_CLUBHEAD = (0.105, 0.000, 0.475)
ADDRESS_CLUB_VECTOR = tuple(
    ADDRESS_CLUBHEAD[i] - ADDRESS_LEFT_WRIST[i] for i in range(3)
)
CLUB_LEN = math.sqrt(sum(value * value for value in ADDRESS_CLUB_VECTOR))
TRAIL_GRIP_DOWN_SHAFT = 0.0762
FACE_NORMAL_MARKER_LEN = 0.12
CLUB_HEAD_HALF_THICKNESS = 0.018
CLUB_HEAD_HALF_WIDTH = 0.050
CLUB_HEAD_HALF_HEIGHT = 0.032

# Keep the address ball/tee in the same world-space neighborhood as the
# accepted kinematic model. Do not move the ball dynamically during visual
# debugging; otherwise the tee and ball appear unrelated.
BALL_X = 0.14
BALL_Y = 0.0
BALL_Z = 0.4813
TEE_Z = BALL_Z - ball_radius - tee_half_height

CONTROL_JOINTS = (
    "chest_turn",
    "left_shoulder_turn",
    "left_shoulder_lift",
    "left_shoulder_roll",
    "left_elbow_flex",
    "left_forearm_rotation",
    "left_wrist_flex",
    "left_wrist_deviation",
    "right_shoulder_turn",
    "right_shoulder_lift",
    "right_shoulder_roll",
    "right_elbow_flex",
    "right_forearm_rotation",
    "right_wrist_flex",
    "right_wrist_deviation",
)

TORQUE_LIMITS = np.array(
    [
        120.0,
        75.0,
        75.0,
        45.0,
        60.0,
        22.0,
        16.0,
        14.0,
        75.0,
        75.0,
        45.0,
        60.0,
        22.0,
        16.0,
        14.0,
    ],
    dtype=np.float64,
)

PD_KP = np.array(
    [
        420.0,
        260.0,
        260.0,
        150.0,
        190.0,
        75.0,
        65.0,
        55.0,
        260.0,
        260.0,
        150.0,
        190.0,
        75.0,
        65.0,
        55.0,
    ],
    dtype=np.float64,
)

PD_KD = np.array(
    [
        28.0,
        18.0,
        18.0,
        10.0,
        13.0,
        5.0,
        4.0,
        4.0,
        18.0,
        18.0,
        10.0,
        13.0,
        5.0,
        4.0,
        4.0,
    ],
    dtype=np.float64,
)

SWING_STEPS = 900


def rad(degrees):
    return math.radians(degrees)


def smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _vec(values):
    return " ".join(f"{value:.6f}" for value in values)


def _actuator_xml(name, limit):
    return f'<motor name="{name}_motor" joint="{name}" gear="1" ctrlrange="{-limit:.6f} {limit:.6f}" ctrllimited="true"/>'


def _target_marker_body(name, rgba):
    return f"""
    <body name="target_{name}_body" mocap="true" pos="0 0 -1">
      <geom name="target_{name}_geom" type="sphere" size="0.018" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>
"""


def _arm_xml(side, shoulder_offset, club_xml=""):
    wrist_sign = -1.0 if side == "right" else 1.0
    elbow_sign = -1.0
    rgba_upper = "0.25 0.38 1.0 1" if side == "right" else "0.95 0.45 0.2 1"
    rgba_fore = "0.2 0.85 0.35 1" if side == "right" else "0.95 0.75 0.2 1"
    return f"""
      <body name="{side}_shoulder_turn_body" pos="{_vec(shoulder_offset)}">
        <inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>
        <joint name="{side}_shoulder_turn" type="hinge" axis="0 0 1" range="-160 160" damping="2.5" armature="0.05" limited="true"/>
        <body name="{side}_shoulder_lift_body">
          <inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>
          <joint name="{side}_shoulder_lift" type="hinge" axis="1 0 0" range="-160 160" damping="2.5" armature="0.05" limited="true"/>
          <body name="{side}_upper_arm">
            <joint name="{side}_shoulder_roll" type="hinge" axis="0 0 1" range="-140 140" damping="1.5" armature="0.03" limited="true"/>
            <geom name="{side}_upper_arm_geom" type="capsule" fromto="0 0 0 0 0 -{UPPER_ARM_LEN:.6f}" size="0.034" mass="1.90" rgba="{rgba_upper}"/>

            <body name="{side}_elbow_flex_body" pos="0 0 -{UPPER_ARM_LEN:.6f}">
              <inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>
              <site name="{side}_elbow_site" pos="0 0 0" size="0.012" rgba="1 0.5 0 1"/>
              <joint name="{side}_elbow_flex" type="hinge" axis="{elbow_sign:.1f} 0 0" range="0 135" damping="1.8" armature="0.035" limited="true"/>
              <body name="{side}_forearm">
                <joint name="{side}_forearm_rotation" type="hinge" axis="0 0 -1" range="-120 120" damping="0.8" armature="0.018" limited="true"/>
                <geom name="{side}_forearm_geom" type="capsule" fromto="0 0 0 0 0 -{FOREARM_LEN:.6f}" size="0.030" mass="1.25" rgba="{rgba_fore}"/>

                <body name="{side}_wrist_flex_body" pos="0 0 -{FOREARM_LEN:.6f}">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>
                  <joint name="{side}_wrist_flex" type="hinge" axis="{wrist_sign:.1f} 0 0" range="-90 90" damping="0.6" armature="0.012" limited="true"/>
                  <body name="{side}_hand">
                    <joint name="{side}_wrist_deviation" type="hinge" axis="0 1 0" range="-45 45" damping="0.6" armature="0.012" limited="true"/>
                    <geom name="{side}_hand_geom" type="sphere" size="0.038" mass="0.45" rgba="0.9 0.78 0.62 1"/>
                    <site name="{side}_grip_site" pos="0 0 0" size="0.012" rgba="0 0.8 1 1"/>
{club_xml}
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
"""


def build_joint_model_xml(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    club_name = normalize_club_name(club_name)
    preset = get_club_preset(club_name)
    loft = rad(preset["loft_deg"])
    face_center_x, face_center_y, face_center_z = ADDRESS_CLUB_VECTOR
    face_normal = (math.cos(loft), 0.0, math.sin(loft))
    heel = (
        face_center_x,
        face_center_y + CLUB_HEAD_HALF_WIDTH,
        face_center_z,
    )
    shaft_dx, shaft_dy, shaft_dz = heel
    shaft_len = math.sqrt(shaft_dx * shaft_dx + shaft_dy * shaft_dy + shaft_dz * shaft_dz)
    shaft_unit = tuple(value / shaft_len for value in heel)
    trail_grip = tuple(value * TRAIL_GRIP_DOWN_SHAFT for value in shaft_unit)
    head_center_x = face_center_x - CLUB_HEAD_HALF_THICKNESS * face_normal[0]
    head_center_y = face_center_y
    head_center_z = face_center_z - CLUB_HEAD_HALF_THICKNESS * face_normal[2]
    face_marker_x = face_center_x + FACE_NORMAL_MARKER_LEN * face_normal[0]
    face_marker_y = face_center_y + FACE_NORMAL_MARKER_LEN * face_normal[1]
    face_marker_z = face_center_z + FACE_NORMAL_MARKER_LEN * face_normal[2]
    actuator_xml = "\n    ".join(
        _actuator_xml(name, float(limit)) for name, limit in zip(CONTROL_JOINTS, TORQUE_LIMITS)
    )
    club_xml = f"""
            <body name="club" pos="0 0 0">
              <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 {shaft_dx:.6f} {shaft_dy:.6f} {shaft_dz:.6f}" size="0.010" mass="{preset["shaft_mass"]:.6f}" rgba="0.05 0.05 0.05 1"/>
              <geom name="club_head_geom" type="box" pos="{head_center_x:.6f} {head_center_y:.6f} {head_center_z:.6f}" quat="{math.cos(-loft / 2.0):.6f} 0 {math.sin(-loft / 2.0):.6f} 0" size="{CLUB_HEAD_HALF_THICKNESS:.6f} {CLUB_HEAD_HALF_WIDTH:.6f} {CLUB_HEAD_HALF_HEIGHT:.6f}" mass="{preset["head_mass"]:.6f}" {CLUB_CONTACT_ATTRS} rgba="0.16 0.16 0.16 1"/>
              <site name="club_grip_site" pos="0 0 0" size="0.012" rgba="0 0.7 1 1"/>
              <site name="club_trail_grip_site" pos="{trail_grip[0]:.6f} {trail_grip[1]:.6f} {trail_grip[2]:.6f}" size="0.011" rgba="0.8 0.2 1 1"/>
              <site name="club_mid_site" pos="{shaft_dx * 0.5:.6f} {shaft_dy * 0.5:.6f} {shaft_dz * 0.5:.6f}" size="0.009" rgba="1 0.8 0 1"/>
              <site name="club_heel_site" pos="{heel[0]:.6f} {heel[1]:.6f} {heel[2]:.6f}" size="0.010" rgba="0.1 1 0.1 1"/>
              <site name="clubhead_site" pos="{face_center_x:.6f} {face_center_y:.6f} {face_center_z:.6f}" size="0.012" rgba="1 0.1 0.1 1"/>
              <site name="club_face_normal_site" pos="{face_marker_x:.6f} {face_marker_y:.6f} {face_marker_z:.6f}" size="0.010" rgba="0 1 1 1"/>
            </body>
"""
    return f"""
<mujoco model="two_arm_joint_golf">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <default>
    <joint damping="1" armature="0.01"/>
    <geom condim="3" friction="0.9 0.02 0.001"/>
  </default>

  <worldbody>
    <light pos="0 1 3" dir="0 -0.3 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="8 8 0.1" rgba="0.76 0.86 0.72 1"/>

    <body name="tee" pos="{BALL_X:.6f} {BALL_Y:.6f} {TEE_Z:.6f}">
      <geom name="tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball" pos="{BALL_X:.6f} {BALL_Y:.6f} {BALL_Z:.6f}">
      <joint name="ball_free" type="free"/>
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass:.6f}" {BALL_CONTACT_ATTRS} rgba="1 1 1 1"/>
    </body>

    {_target_marker_body("left_wrist", "1 0.3 0.1 0.45")}
    {_target_marker_body("right_wrist", "0.1 0.5 1 0.45")}
    {_target_marker_body("clubhead", "1 1 0 0.55")}
    {_target_marker_body("left_elbow", "1 0.6 0.1 0.35")}
    {_target_marker_body("right_elbow", "0.1 1 0.4 0.35")}

    <body name="chest" pos="{_vec(CHEST_POS)}">
      <joint name="chest_turn" type="hinge" axis="{_vec(SPINE_AXIS)}" range="-125 125" damping="4.0" armature="0.08" limited="true"/>
      <geom name="chest_bar" type="capsule" fromto="{_vec(RIGHT_SHOULDER_OFFSET)} {_vec(LEFT_SHOULDER_OFFSET)}" size="0.055" mass="16.0" rgba="0.75 0.55 0.38 1"/>
      <site name="right_shoulder_site" pos="{_vec(RIGHT_SHOULDER_OFFSET)}" size="0.018" rgba="0.1 0.1 1 1"/>
      <site name="left_shoulder_site" pos="{_vec(LEFT_SHOULDER_OFFSET)}" size="0.018" rgba="1 0.2 0.1 1"/>

{_arm_xml("right", RIGHT_SHOULDER_OFFSET)}
{_arm_xml("left", LEFT_SHOULDER_OFFSET, club_xml)}
    </body>
  </worldbody>

  <equality>
    <weld name="right_hand_rigid_grip" site1="right_grip_site" site2="club_trail_grip_site" solref="0.006 1" solimp="0.99 0.995 0.001"/>
  </equality>

  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>
"""


def make_joint_model(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    return mujoco.MjModel.from_xml_string(build_joint_model_xml(club_name, hand))


def joint_qpos_indices(model):
    indices = []
    for joint_name in CONTROL_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        indices.append(model.jnt_qposadr[joint_id])
    return np.asarray(indices, dtype=np.int32)


def joint_qvel_indices(model):
    indices = []
    for joint_name in CONTROL_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        indices.append(model.jnt_dofadr[joint_id])
    return np.asarray(indices, dtype=np.int32)


BASELINE_KEYFRAMES = (
    (0.00, (0, 0, 0, 0, 4, 0, -8, 0, 0, 0, 0, 6, 0, 8, 0)),
    (0.22, (38, 8, 18, -8, 4, -12, -12, -4, 8, 8, -10, 18, 28, 22, 8)),
    (0.50, (95, 20, 72, -12, 2, -35, -35, -10, 30, 48, 18, 92, 55, 55, 18)),
    (0.64, (48, 10, 38, -6, 2, -14, -10, -4, 18, 28, 6, 48, 20, 22, 7)),
    (0.78, (0, 0, 0, 0, 3, 0, 10, 2, 0, 0, 0, 8, 0, -6, 0)),
    (0.90, (-20, -6, 35, 8, 18, 20, 24, 6, -8, 42, 8, 8, -10, -12, -4)),
    (1.00, (-45, -14, 70, 18, 95, 35, 34, 10, -16, 72, 12, 18, -20, -18, -8)),
)


def baseline_qpos(progress):
    progress = max(0.0, min(1.0, float(progress)))
    for index in range(len(BASELINE_KEYFRAMES) - 1):
        t0, q0_deg = BASELINE_KEYFRAMES[index]
        t1, q1_deg = BASELINE_KEYFRAMES[index + 1]
        if progress <= t1:
            local = smootherstep((progress - t0) / max(t1 - t0, 1e-9))
            q0 = np.radians(np.asarray(q0_deg, dtype=np.float64))
            q1 = np.radians(np.asarray(q1_deg, dtype=np.float64))
            return q0 + (q1 - q0) * local
    return np.radians(np.asarray(BASELINE_KEYFRAMES[-1][1], dtype=np.float64))


def baseline_qvel(progress, dt=1.0 / SWING_STEPS):
    before = baseline_qpos(max(0.0, progress - dt))
    after = baseline_qpos(min(1.0, progress + dt))
    return (after - before) / max(2.0 * dt * SWING_STEPS * 0.002, 1e-9)


def baseline_pd_control(model, data, qpos_indices, qvel_indices, step, residual=None):
    progress = min(1.0, step / max(SWING_STEPS - 1, 1))
    target = baseline_qpos(progress)
    target_vel = baseline_qvel(progress)
    qpos = data.qpos[qpos_indices]
    qvel = data.qvel[qvel_indices]
    ctrl = PD_KP * (target - qpos) + PD_KD * (target_vel - qvel)
    if residual is not None:
        ctrl = ctrl + residual
    data.ctrl[:] = np.clip(ctrl, -TORQUE_LIMITS, TORQUE_LIMITS)
    return target, ctrl


def reset_to_baseline(model, data):
    mujoco.mj_resetData(model, data)
    qpos_indices = joint_qpos_indices(model)
    data.qpos[qpos_indices] = baseline_qpos(0.0)
    mujoco.mj_forward(model, data)


def align_ball_to_address_clubface(model, data):
    """Place the ball just in front of the clubface for the current address."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "clubhead_site")
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = model.jnt_qposadr[ball_joint_id]
    clubhead = data.site_xpos[site_id].copy()
    data.qpos[ball_qpos : ball_qpos + 3] = (
        clubhead[0] + ball_radius + 0.008,
        clubhead[1],
        clubhead[2],
    )
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[model.jnt_dofadr[ball_joint_id] : model.jnt_dofadr[ball_joint_id] + 6] = 0.0


def run_baseline_viewer(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND, speed=6.0):
    model = make_joint_model(club_name, hand)
    data = mujoco.MjData(model)
    qpos_indices = joint_qpos_indices(model)
    qvel_indices = joint_qvel_indices(model)
    reset_to_baseline(model, data)
    print("Joint-accurate two-arm baseline")
    print("Club:", get_club_preset(club_name)["label"])
    print("Controlled DOF:", len(CONTROL_JOINTS), "=", "chest + 7 left arm + 7 right arm")
    print("No pauses: baseline progresses continuously from address to finish.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if step >= SWING_STEPS:
                reset_to_baseline(model, data)
                step = 0
            baseline_pd_control(model, data, qpos_indices, qvel_indices, step)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * speed)
            step += 1


def main():
    parser = argparse.ArgumentParser(description="Run the joint-accurate two-arm baseline swing.")
    parser.add_argument("--club", default=DEFAULT_CLUB)
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=("right", "left"))
    parser.add_argument("--speed", type=float, default=6.0)
    args = parser.parse_args()
    run_baseline_viewer(args.club, args.hand, args.speed)


if __name__ == "__main__":
    main()
