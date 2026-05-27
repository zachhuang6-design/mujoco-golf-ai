"""Joint-accurate two-arm MuJoCo golf model and baseline controller.

This is the first physics-first version of the two-arm swing. Unlike the
current mocap visual sketches, this model uses actual MuJoCo joints,
actuators, segment masses, club masses, a free ball, and bounded torques.
"""

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from golf_core.common import (
    BALL_CONTACT_ATTRS,
    CLUB_CONTACT_ATTRS,
    CONTACT_SOLIMP,
    CONTACT_SOLREF,
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
BALL_X = 0.1053
BALL_Y = -0.0079
BALL_Z = 0.4903
TEE_Z = BALL_Z - ball_radius - tee_half_height

CONTROL_JOINTS = (
    "chest_turn",
    "chest_forward_bend",
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

JOINT_POWER_SCALE = 1.20

TORQUE_LIMITS = np.array(
    [
        120.0,
        85.0,
        75.0,
        75.0,
        45.0,
        60.0,
        28.0,
        22.0,
        18.0,
        75.0,
        75.0,
        45.0,
        60.0,
        28.0,
        22.0,
        18.0,
    ],
    dtype=np.float64,
) * JOINT_POWER_SCALE

PD_KP = np.array(
    [
        420.0,
        300.0,
        260.0,
        260.0,
        150.0,
        190.0,
        75.0,
        65.0,
        34.0,
        260.0,
        260.0,
        150.0,
        190.0,
        28.0,
        65.0,
        6.0,
    ],
    dtype=np.float64,
)

PD_KD = np.array(
    [
        28.0,
        24.0,
        18.0,
        18.0,
        10.0,
        13.0,
        5.0,
        4.0,
        2.5,
        18.0,
        18.0,
        10.0,
        13.0,
        2.0,
        4.0,
        0.4,
    ],
    dtype=np.float64,
)

SWING_STEPS = 900
DEFAULT_IK_TRAJECTORY = "artifacts/ik/two_arm_joint_ik_7iron_right.json"
SETUP_PAUSE_STEPS = 80
TOP_PAUSE_STEPS = 45
FINISH_ROLLOUT_STEPS = 900
TRANSITION_TOP_PROGRESS = 0.50
TRANSITION_IMPACT_PROGRESS = 0.78
ARM_RELEASE_DELAY = 0.12
TAKEAWAY_PROGRESS = 0.22
WRIST_ACTION_START_PROGRESS = 0.30
TAKEAWAY_FACE_HOLD_PROGRESS = 0.28
TAKEAWAY_FACE_RELEASE_PROGRESS = 0.40
CONNECTED_ARMS_START_PROGRESS = 0.52
CONNECTED_ARMS_FULL_PROGRESS = 0.68
IMPACT_DELIVERY_START_PROGRESS = 0.54
IMPACT_DELIVERY_FULL_PROGRESS = 0.62
IMPACT_DELIVERY_HOLD_PROGRESS = 0.92
SPINE_IMPACT_BEND_DEG = -22.0
SPINE_FINISH_BEND_DEG = -5.0
CONTROL_FILTER_ALPHA = 0.20
MAX_CTRL_STEP_FRACTION = 0.025
MIN_DRIVE_FRACTION = 0.018
TARGET_VELOCITY_DEADBAND = 0.015
SETUP_HOLD_KP_MULT = 1.8
SETUP_HOLD_KD_MULT = 2.4
SETUP_LEFT_ULNAR_DEG = 1.5
SETUP_RIGHT_RADIAL_DEG = 18.0

JOINT_GROUPS = {
    "torso": ("chest_turn", "chest_forward_bend"),
    "left_arm": (
        "left_shoulder_turn",
        "left_shoulder_lift",
        "left_shoulder_roll",
        "left_elbow_flex",
        "left_forearm_rotation",
        "left_wrist_flex",
        "left_wrist_deviation",
    ),
    "right_arm": (
        "right_shoulder_turn",
        "right_shoulder_lift",
        "right_shoulder_roll",
        "right_elbow_flex",
        "right_forearm_rotation",
        "right_wrist_flex",
        "right_wrist_deviation",
    ),
}

LEGACY_IK_JOINTS = tuple(name for name in CONTROL_JOINTS if name != "chest_forward_bend")


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


def _spine_anchor_xml():
    axis = np.asarray(SPINE_AXIS, dtype=np.float64)
    chest = np.asarray(CHEST_POS, dtype=np.float64)
    spine_bottom = chest - axis * 0.58
    spine_top = chest + axis * 0.22
    pelvis_left = spine_bottom + np.asarray((-0.22, 0.0, -0.03), dtype=np.float64)
    pelvis_right = spine_bottom + np.asarray((0.22, 0.0, -0.03), dtype=np.float64)
    return f"""
    <geom name="fixed_spine_axis" type="capsule" fromto="{_vec(spine_bottom)} {_vec(spine_top)}" size="0.026" rgba="0.18 0.18 0.18 0.55" contype="0" conaffinity="0"/>
    <geom name="fixed_pelvis_bar" type="capsule" fromto="{_vec(pelvis_left)} {_vec(pelvis_right)}" size="0.045" rgba="0.42 0.32 0.24 0.55" contype="0" conaffinity="0"/>
    <geom name="chest_hinge_marker" type="sphere" pos="{_vec(CHEST_POS)}" size="0.032" rgba="0.05 0.05 0.05 0.65" contype="0" conaffinity="0"/>
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
                <joint name="{side}_forearm_rotation" type="hinge" axis="0 0 -1" range="-120 120" damping="1.2" armature="0.024" limited="true"/>
                <geom name="{side}_forearm_geom" type="capsule" fromto="0 0 0 0 0 -{FOREARM_LEN:.6f}" size="0.030" mass="1.25" rgba="{rgba_fore}"/>

                <body name="{side}_wrist_flex_body" pos="0 0 -{FOREARM_LEN:.6f}">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>
                  <joint name="{side}_wrist_flex" type="hinge" axis="{wrist_sign:.1f} 0 0" range="-90 90" damping="1.0" armature="0.020" limited="true"/>
                  <body name="{side}_hand">
                    <joint name="{side}_wrist_deviation" type="hinge" axis="0 1 0" range="-70 50" damping="1.0" armature="0.020" limited="true"/>
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
    <geom name="floor" type="plane" pos="0 0 0" size="8 8 0.1" friction="0.03 0.002 0.0001" rgba="0.76 0.86 0.72 1"/>
    {_spine_anchor_xml()}

    <body name="tee" pos="{BALL_X:.6f} {BALL_Y:.6f} {TEE_Z:.6f}">
      <geom name="tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball" pos="{BALL_X:.6f} {BALL_Y:.6f} {BALL_Z:.6f}">
      <joint name="ball_free" type="free" damping="0" armature="0"/>
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass:.6f}" {BALL_CONTACT_ATTRS} rgba="1 1 1 1"/>
    </body>

    {_target_marker_body("left_wrist", "1 0.3 0.1 0.45")}
    {_target_marker_body("right_wrist", "0.1 0.5 1 0.45")}
    {_target_marker_body("clubhead", "1 1 0 0.55")}
    {_target_marker_body("left_elbow", "1 0.6 0.1 0.35")}
    {_target_marker_body("right_elbow", "0.1 1 0.4 0.35")}

    <body name="chest" pos="{_vec(CHEST_POS)}">
      <joint name="chest_turn" type="hinge" axis="{_vec(SPINE_AXIS)}" range="-125 125" damping="4.0" armature="0.08" limited="true"/>
      <joint name="chest_forward_bend" type="hinge" axis="0 1 0" range="-25 18" damping="5.0" armature="0.10" limited="true"/>
      <geom name="chest_bar" type="capsule" fromto="{_vec(RIGHT_SHOULDER_OFFSET)} {_vec(LEFT_SHOULDER_OFFSET)}" size="0.055" mass="16.0" rgba="0.75 0.55 0.38 1"/>
      <site name="right_shoulder_site" pos="{_vec(RIGHT_SHOULDER_OFFSET)}" size="0.018" rgba="0.1 0.1 1 1"/>
      <site name="left_shoulder_site" pos="{_vec(LEFT_SHOULDER_OFFSET)}" size="0.018" rgba="1 0.2 0.1 1"/>

{_arm_xml("right", RIGHT_SHOULDER_OFFSET)}
{_arm_xml("left", LEFT_SHOULDER_OFFSET, club_xml)}
    </body>
  </worldbody>

  <contact>
    <pair geom1="club_head_geom" geom2="golf_ball" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" condim="4" friction="0.9 0.01 0.0001"/>
    <pair geom1="club_shaft_geom" geom2="golf_ball" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" condim="4" friction="0.35 0.005 0.0001"/>
    <pair geom1="floor" geom2="golf_ball" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" condim="4" friction="0.015 0.001 0.00001"/>
  </contact>

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
    (
        0.00,
        (
            2.117,
            0.0,
            -75.577,
            -26.259,
            37.336,
            0.0,
            -37.336,
            7.228,
            -22.302,
            65.793,
            -21.553,
            8.358,
            7.989,
            0.0,
            8.0,
            18.0,
        ),
    ),
    (0.22, (38, 0, 8, 18, -8, 4, -12, -12, -4, 8, 8, -10, 18, 28, 22, 8)),
    (0.50, (95, 0, 20, 72, -12, 2, -35, -35, -10, 30, 48, 18, 92, 55, 55, 18)),
    (0.64, (48, -8, 10, 38, -6, 2, -14, -10, -4, 18, 28, 6, 48, 20, 22, 7)),
    (0.78, (0, -12, 0, 0, 0, 3, 0, 10, 2, 0, 0, 0, 8, 0, -6, 0)),
    (0.90, (-20, -8, -6, 35, 8, 18, 20, 24, 6, -8, 42, 8, 8, -10, -12, -4)),
    (1.00, (-45, -5, -14, 70, 18, 95, 35, 34, 10, -16, 72, 12, 18, -20, -18, -8)),
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


def load_qpos_trajectory(path):
    payload = json.loads(Path(path).read_text())
    joints = tuple(payload.get("joints", ()))
    qpos = np.asarray(payload["qpos_radians"], dtype=np.float64)
    if joints == tuple(CONTROL_JOINTS):
        qpos = apply_spine_bend_profile(qpos)
        qpos = apply_biomechanics_cleanup_profile(qpos)
        qpos = apply_transition_sequence_profile(qpos)
        qpos = apply_face_square_wrist_profile(qpos)
        qpos = apply_impact_delivery_profile(qpos)
        qpos = apply_takeaway_wrist_hold_profile(qpos)
        return payload, qpos
    if joints == LEGACY_IK_JOINTS:
        qpos = insert_spine_bend_column(qpos)
        qpos = apply_biomechanics_cleanup_profile(qpos)
        qpos = apply_transition_sequence_profile(qpos)
        qpos = apply_face_square_wrist_profile(qpos)
        qpos = apply_impact_delivery_profile(qpos)
        qpos = apply_takeaway_wrist_hold_profile(qpos)
        payload["joints"] = list(CONTROL_JOINTS)
        return payload, qpos
    raise ValueError("Trajectory joints do not match the joint physics model.")


def spine_bend_profile(progress):
    impact_bend = rad(SPINE_IMPACT_BEND_DEG)
    finish_bend = rad(SPINE_FINISH_BEND_DEG)
    p = max(0.0, min(1.0, float(progress)))
    if p < 0.56:
        return 0.0
    if p < 0.78:
        return impact_bend * smootherstep((p - 0.56) / 0.22)
    return impact_bend + (finish_bend - impact_bend) * smootherstep((p - 0.78) / 0.22)


def insert_spine_bend_column(qpos):
    progress = np.linspace(0.0, 1.0, len(qpos))
    bend = np.asarray([spine_bend_profile(value) for value in progress], dtype=np.float64)
    return np.insert(qpos, 1, bend, axis=1)


def apply_spine_bend_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    progress = np.linspace(0.0, 1.0, len(qpos))
    qpos[:, 1] = np.asarray([spine_bend_profile(value) for value in progress], dtype=np.float64)
    return qpos


def bell_profile(progress, start, peak, end):
    p = max(0.0, min(1.0, float(progress)))
    if p <= start or p >= end:
        return 0.0
    if p <= peak:
        return smootherstep((p - start) / max(peak - start, 1e-9))
    return 1.0 - smootherstep((p - peak) / max(end - peak, 1e-9))


def apply_biomechanics_cleanup_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    progress = np.linspace(0.0, 1.0, len(qpos))
    index = {name: i for i, name in enumerate(CONTROL_JOINTS)}
    for row, p in zip(qpos, progress):
        setup = 1.0 - smootherstep(p / 0.18)
        top = bell_profile(p, WRIST_ACTION_START_PROGRESS, 0.50, 0.61)
        impact = bell_profile(p, 0.60, 0.78, 0.87)

        # Address grip bias: a little more lead-hand ulnar deviation and trail
        # hand radial deviation sets the shaft more directly behind the ball.
        row[index["left_wrist_deviation"]] += rad(SETUP_LEFT_ULNAR_DEG) * setup
        row[index["right_wrist_deviation"]] += rad(SETUP_RIGHT_RADIAL_DEG) * setup

        # Flatten the club at the top. The old target put the lead wrist into a
        # very bowed/cupped extreme and left the trail wrist under-set, which
        # made the club look too vertical and above the plane.
        row[index["left_wrist_flex"]] += rad(86.0) * top
        row[index["left_wrist_deviation"]] -= rad(30.0) * top
        row[index["right_wrist_flex"]] += rad(34.0) * top

        # Impact cleanup: less open chest/torso lead and a straighter trail arm.
        # This gives the club more room to reach the ball without an arm-wrist
        # compensation move.
        row[index["chest_turn"]] -= rad(14.0) * impact
        row[index["right_elbow_flex"]] -= rad(16.0) * impact
    return qpos


def transition_release_progress(progress, delay=ARM_RELEASE_DELAY):
    p = max(0.0, min(1.0, float(progress)))
    if p <= TRANSITION_TOP_PROGRESS or p >= TRANSITION_IMPACT_PROGRESS:
        return p
    local = (p - TRANSITION_TOP_PROGRESS) / (TRANSITION_IMPACT_PROGRESS - TRANSITION_TOP_PROGRESS)
    if local <= delay:
        released = 0.0
    else:
        released = smoothstep((local - delay) / max(1.0 - delay, 1e-9))
    return TRANSITION_TOP_PROGRESS + released * (TRANSITION_IMPACT_PROGRESS - TRANSITION_TOP_PROGRESS)


def apply_transition_sequence_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    source = qpos.copy()
    progress = np.linspace(0.0, 1.0, len(qpos))
    index = {name: i for i, name in enumerate(CONTROL_JOINTS)}
    arm_joint_names = [name for name in CONTROL_JOINTS if not name.startswith("chest_")]
    arm_indices = [index[name] for name in arm_joint_names]
    top_pose = interp_qpos_trajectory(source, TRANSITION_TOP_PROGRESS)
    impact_pose = interp_qpos_trajectory(source, TRANSITION_IMPACT_PROGRESS)
    smooth_release_names = (
        "left_forearm_rotation",
        "left_wrist_flex",
        "left_wrist_deviation",
        "right_shoulder_roll",
        "right_elbow_flex",
    )
    smooth_release_indices = [index[name] for name in smooth_release_names]

    for row, p in zip(qpos, progress):
        if p <= TRANSITION_TOP_PROGRESS or p >= TRANSITION_IMPACT_PROGRESS:
            continue
        local = (p - TRANSITION_TOP_PROGRESS) / (TRANSITION_IMPACT_PROGRESS - TRANSITION_TOP_PROGRESS)
        arm_progress = transition_release_progress(p)
        delayed_arm = interp_qpos_trajectory(source, arm_progress)
        row[arm_indices] = delayed_arm[arm_indices]

        release = smoothstep(local)
        row[smooth_release_indices] = (
            top_pose[smooth_release_indices]
            + (impact_pose[smooth_release_indices] - top_pose[smooth_release_indices]) * release
        )

        transition = bell_profile(p, TRANSITION_TOP_PROGRESS, 0.64, TRANSITION_IMPACT_PROGRESS)
        row[index["left_wrist_flex"]] += rad(5.0) * transition
        row[index["left_wrist_deviation"]] -= rad(4.0) * transition
        row[index["right_elbow_flex"]] -= rad(5.0) * transition
    return qpos


def window_profile(progress, start, full_start, full_end, end):
    p = max(0.0, min(1.0, float(progress)))
    if p <= start or p >= end:
        return 0.0
    if p < full_start:
        return smootherstep((p - start) / max(full_start - start, 1e-9))
    if p <= full_end:
        return 1.0
    return 1.0 - smootherstep((p - full_end) / max(end - full_end, 1e-9))


def apply_face_square_wrist_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    progress = np.linspace(0.0, 1.0, len(qpos))
    index = {name: i for i, name in enumerate(CONTROL_JOINTS)}
    for row, p in zip(qpos, progress):
        strength = window_profile(p, WRIST_ACTION_START_PROGRESS, 0.50, TRANSITION_IMPACT_PROGRESS, 0.86)
        if strength <= 0.0:
            continue

        phase = smootherstep(
            (p - TRANSITION_TOP_PROGRESS)
            / max(TRANSITION_IMPACT_PROGRESS - TRANSITION_TOP_PROGRESS, 1e-9)
        )
        # The previous pass over-held extension and left the physical club high
        # and open. This pass flexes the wrists the opposite way into impact:
        # lead wrist slightly flexed, trail wrist releasing out of extension.
        desired_left_flex = rad(4.0 + (10.0 - 4.0) * phase)
        desired_right_flex = rad(32.0 + (0.0 - 32.0) * phase)
        desired_left_deviation = rad(15.0 + (-22.0 - 15.0) * phase)

        row[index["left_wrist_flex"]] = (
            row[index["left_wrist_flex"]] * (1.0 - strength) + desired_left_flex * strength
        )
        row[index["right_wrist_flex"]] = (
            row[index["right_wrist_flex"]] * (1.0 - strength) + desired_right_flex * strength
        )
        row[index["left_wrist_deviation"]] = (
            row[index["left_wrist_deviation"]] * (1.0 - strength) + desired_left_deviation * strength
        )
    return qpos


def apply_impact_delivery_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    address = qpos[0].copy()
    impact_target = address.copy()
    index = {name: i for i, name in enumerate(CONTROL_JOINTS)}
    impact_target[index["chest_turn"]] = rad(-6.0)
    impact_target[index["chest_forward_bend"]] = rad(-20.0)
    impact_target[index["left_shoulder_turn"]] = address[index["left_shoulder_turn"]] + rad(-20.0)
    impact_target[index["right_shoulder_turn"]] = address[index["right_shoulder_turn"]] + rad(20.0)
    impact_target[index["left_shoulder_lift"]] = address[index["left_shoulder_lift"]] + rad(20.0)
    impact_target[index["right_shoulder_lift"]] = address[index["right_shoulder_lift"]] + rad(20.0)
    impact_target[index["right_elbow_flex"]] = rad(12.0)
    impact_target[index["left_wrist_flex"]] = rad(10.0)
    impact_target[index["right_wrist_flex"]] = rad(0.0)
    impact_target[index["left_wrist_deviation"]] = address[index["left_wrist_deviation"]]

    full_body_names = (
        "chest_turn",
        "chest_forward_bend",
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
    full_body_indices = [index[name] for name in full_body_names]
    connected_arm_names = (
        "left_shoulder_turn",
        "left_shoulder_lift",
        "left_shoulder_roll",
        "left_elbow_flex",
        "left_forearm_rotation",
        "right_shoulder_turn",
        "right_shoulder_lift",
        "right_shoulder_roll",
        "right_elbow_flex",
        "right_forearm_rotation",
    )
    connected_arm_indices = [index[name] for name in connected_arm_names]
    progress = np.linspace(0.0, 1.0, len(qpos))
    for row, p in zip(qpos, progress):
        # After the transition, bring the upper arms back toward the address
        # ribcage relationship earlier than the wrists release. This keeps the
        # "armpits connected" look without changing setup or early takeaway.
        connection = window_profile(
            p,
            CONNECTED_ARMS_START_PROGRESS,
            CONNECTED_ARMS_FULL_PROGRESS,
            0.82,
            0.90,
        )
        if connection > 0.0:
            row[connected_arm_indices] = (
                row[connected_arm_indices] * (1.0 - connection)
                + impact_target[connected_arm_indices] * connection
            )

        # Give the torque-limited model more time to get the hands back down
        # near the body. The target already passes through the ball, but the
        # physical club lags high unless impact delivery starts before the
        # nominal strike frame and holds through early followthrough.
        delivery = window_profile(
            p,
            IMPACT_DELIVERY_START_PROGRESS,
            IMPACT_DELIVERY_FULL_PROGRESS,
            IMPACT_DELIVERY_HOLD_PROGRESS,
            1.0,
        )
        if delivery <= 0.0:
            continue
        row[full_body_indices] = (
            row[full_body_indices] * (1.0 - delivery)
            + impact_target[full_body_indices] * delivery
        )
    return qpos


def apply_takeaway_wrist_hold_profile(qpos):
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    address = qpos[0].copy()
    progress = np.linspace(0.0, 1.0, len(qpos))
    index = {name: i for i, name in enumerate(CONTROL_JOINTS)}
    takeaway_face_names = (
        "left_forearm_rotation",
        "left_wrist_flex",
        "left_wrist_deviation",
        "right_forearm_rotation",
        "right_wrist_flex",
        "right_wrist_deviation",
    )
    takeaway_face_indices = [index[name] for name in takeaway_face_names]
    for row, p in zip(qpos, progress):
        if p <= TAKEAWAY_FACE_HOLD_PROGRESS:
            hold = 1.0
        elif p < TAKEAWAY_FACE_RELEASE_PROGRESS:
            hold = 1.0 - smootherstep(
                (p - TAKEAWAY_FACE_HOLD_PROGRESS)
                / max(TAKEAWAY_FACE_RELEASE_PROGRESS - TAKEAWAY_FACE_HOLD_PROGRESS, 1e-9)
            )
        else:
            hold = 0.0
        if hold > 0.0:
            row[takeaway_face_indices] = (
                row[takeaway_face_indices] * (1.0 - hold)
                + address[takeaway_face_indices] * hold
            )
    return qpos


def interp_qpos_trajectory(qpos, progress):
    scaled = max(0.0, min(1.0, float(progress))) * (len(qpos) - 1)
    lo = int(math.floor(scaled))
    hi = min(len(qpos) - 1, lo + 1)
    t = scaled - lo
    return qpos[lo] + (qpos[hi] - qpos[lo]) * t


def interp_qvel_trajectory(qpos, progress, timestep):
    delta = 1.0 / max(SWING_STEPS - 1, 1)
    before = interp_qpos_trajectory(qpos, max(0.0, progress - delta))
    after = interp_qpos_trajectory(qpos, min(1.0, progress + delta))
    return (after - before) / max(2.0 * delta * SWING_STEPS * timestep, 1e-9)


def resample_qpos_trajectory(qpos, steps=SWING_STEPS):
    if len(qpos) == steps:
        return qpos.copy()
    source = np.linspace(0.0, 1.0, len(qpos))
    target = np.linspace(0.0, 1.0, steps)
    resampled = np.empty((steps, qpos.shape[1]), dtype=np.float64)
    for joint_index in range(qpos.shape[1]):
        resampled[:, joint_index] = np.interp(target, source, qpos[:, joint_index])
    return resampled


def smooth_qpos_trajectory(qpos, passes=3, kernel=(1.0, 4.0, 6.0, 4.0, 1.0)):
    smoothed = np.asarray(qpos, dtype=np.float64).copy()
    weights = np.asarray(kernel, dtype=np.float64)
    weights /= np.sum(weights)
    radius = len(weights) // 2
    for _ in range(passes):
        padded = np.pad(smoothed, ((radius, radius), (0, 0)), mode="edge")
        next_values = np.empty_like(smoothed)
        for index in range(len(smoothed)):
            window = padded[index : index + len(weights)]
            next_values[index] = np.sum(window * weights[:, None], axis=0)
        next_values[0] = smoothed[0]
        next_values[-1] = smoothed[-1]
        smoothed = next_values
    return smoothed


def prepare_control_trajectory(qpos, steps=SWING_STEPS):
    return smooth_qpos_trajectory(resample_qpos_trajectory(qpos, steps=steps), passes=5)


def filter_control(raw_ctrl, previous_ctrl, target_vel):
    if previous_ctrl is None:
        previous_ctrl = np.zeros_like(raw_ctrl)
    blended = previous_ctrl + CONTROL_FILTER_ALPHA * (raw_ctrl - previous_ctrl)
    max_step = TORQUE_LIMITS * MAX_CTRL_STEP_FRACTION
    filtered = previous_ctrl + np.clip(blended - previous_ctrl, -max_step, max_step)

    drive_floor = TORQUE_LIMITS * MIN_DRIVE_FRACTION
    moving = np.abs(target_vel) > TARGET_VELOCITY_DEADBAND
    too_small = np.abs(filtered) < drive_floor
    signs = np.sign(target_vel)
    filtered[moving & too_small] = signs[moving & too_small] * drive_floor[moving & too_small]
    return np.clip(filtered, -TORQUE_LIMITS, TORQUE_LIMITS)


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


def trajectory_pd_control(
    model,
    data,
    qpos_indices,
    qvel_indices,
    qpos_trajectory,
    step=None,
    residual=None,
    previous_ctrl=None,
    progress=None,
    velocity_scale=1.0,
    hold_target=False,
):
    if progress is None:
        progress = min(1.0, step / max(SWING_STEPS - 1, 1))
    target = interp_qpos_trajectory(qpos_trajectory, progress)
    if hold_target:
        target_vel = np.zeros_like(target)
    else:
        target_vel = interp_qvel_trajectory(qpos_trajectory, progress, model.opt.timestep) * velocity_scale
    qpos = data.qpos[qpos_indices]
    qvel = data.qvel[qvel_indices]
    kp = PD_KP * (SETUP_HOLD_KP_MULT if hold_target else 1.0)
    kd = PD_KD * (SETUP_HOLD_KD_MULT if hold_target else 1.0)
    ctrl = kp * (target - qpos) + kd * (target_vel - qvel)
    if residual is not None:
        ctrl = ctrl + residual
    if hold_target:
        ctrl = np.clip(ctrl, -TORQUE_LIMITS, TORQUE_LIMITS)
    else:
        ctrl = filter_control(np.clip(ctrl, -TORQUE_LIMITS, TORQUE_LIMITS), previous_ctrl, target_vel)
    data.ctrl[:] = ctrl
    return target, ctrl


def reset_to_baseline(model, data):
    mujoco.mj_resetData(model, data)
    qpos_indices = joint_qpos_indices(model)
    data.qpos[qpos_indices] = baseline_qpos(0.0)
    mujoco.mj_forward(model, data)
    align_ball_to_address_clubface(model, data)
    mujoco.mj_forward(model, data)


def align_ball_to_address_clubface(model, data):
    """Place the ball just in front of the clubface for the current address."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "clubhead_site")
    normal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_face_normal_site")
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = model.jnt_qposadr[ball_joint_id]
    clubhead = data.site_xpos[site_id].copy()
    face_normal = data.site_xpos[normal_id].copy() - clubhead
    face_normal /= max(np.linalg.norm(face_normal), 1e-9)
    ball_center = clubhead + face_normal * (ball_radius + 0.006)
    ball_center[2] = max(ball_center[2], ball_radius + tee_half_height + TEE_Z)
    data.qpos[ball_qpos : ball_qpos + 3] = ball_center
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[model.jnt_dofadr[ball_joint_id] : model.jnt_dofadr[ball_joint_id] + 6] = 0.0


def _joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def site_position(model, data, name):
    return data.site_xpos[_site_id(model, name)].copy()


def body_position(model, data, name):
    return data.xpos[_body_id(model, name)].copy()


def face_angle_degrees(model, data, club_name):
    head = site_position(model, data, "clubhead_site")
    marker = site_position(model, data, "club_face_normal_site")
    current = marker - head
    current /= max(np.linalg.norm(current), 1e-9)
    loft = rad(get_club_preset(club_name)["loft_deg"])
    target = np.asarray((math.cos(loft), 0.0, math.sin(loft)), dtype=np.float64)
    return math.degrees(math.acos(float(np.clip(np.dot(current, target), -1.0, 1.0))))


def right_grip_gap(model, data):
    return float(
        np.linalg.norm(
            site_position(model, data, "right_grip_site")
            - site_position(model, data, "club_trail_grip_site")
        )
    )


def model_mass_summary(model):
    return {
        "chest_kg": float(model.body_subtreemass[_body_id(model, "chest")]),
        "right_arm_kg": float(model.body_subtreemass[_body_id(model, "right_shoulder_turn_body")]),
        "left_arm_and_club_kg": float(model.body_subtreemass[_body_id(model, "left_shoulder_turn_body")]),
        "club_kg": float(model.body_subtreemass[_body_id(model, "club")]),
        "ball_kg": float(model.body_subtreemass[_body_id(model, "ball")]),
    }


def print_physics_audit(club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    model = make_joint_model(club_name, hand)
    data = mujoco.MjData(model)
    reset_to_baseline(model, data)
    preset = get_club_preset(club_name)

    print("Two-arm joint physics audit")
    print("Club:", preset["label"])
    print("Handedness:", hand)
    print("Timestep:", model.opt.timestep)
    print("Integrator:", mujoco.mjtIntegrator(model.opt.integrator).name)
    print("Gravity:", [round(float(value), 4) for value in model.opt.gravity])
    print("Controlled DOF:", len(CONTROL_JOINTS), "= chest + 7 left arm + 7 right arm")
    print("Joint groups:")
    for group, names in JOINT_GROUPS.items():
        print(" ", group + ":", ", ".join(names))

    print("Joint limits and torque limits:")
    for name, torque_limit in zip(CONTROL_JOINTS, TORQUE_LIMITS):
        joint_id = _joint_id(model, name)
        axis = model.jnt_axis[joint_id]
        low, high = np.degrees(model.jnt_range[joint_id])
        print(
            " ",
            name,
            "range_deg",
            f"{low:.1f}..{high:.1f}",
            "axis",
            [round(float(value), 3) for value in axis],
            "torque_Nm",
            round(float(torque_limit), 3),
        )

    print("Mass summary:")
    for name, value in model_mass_summary(model).items():
        print(" ", name + ":", round(value, 4))
    print("Club preset masses:")
    print("  shaft_kg:", preset["shaft_mass"])
    print("  head_kg:", preset["head_mass"])
    print("  loft_deg:", preset["loft_deg"])
    print("Ball mass kg:", ball_mass)

    ball_pos = body_position(model, data, "ball")
    clubhead = site_position(model, data, "clubhead_site")
    heel = site_position(model, data, "club_heel_site")
    print("Address geometry:")
    print("  ball:", [round(float(value), 4) for value in ball_pos])
    print("  clubface center:", [round(float(value), 4) for value in clubhead])
    print("  heel:", [round(float(value), 4) for value in heel])
    print("  clubface-to-ball distance:", round(float(np.linalg.norm(clubhead - ball_pos)), 5))
    print("  face angle from square deg:", round(face_angle_degrees(model, data, club_name), 3))
    print("  right-hand grip gap m:", round(right_grip_gap(model, data), 6))

    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_hand_rigid_grip")
    print("Right-hand rigid weld present:", weld_id >= 0)
    print("Contact geoms present:")
    for geom_name in ("club_head_geom", "golf_ball", "floor", "tee_geom"):
        print(" ", geom_name + ":", _geom_id(model, geom_name) >= 0)


def export_joint_model_xml(path, club_name=DEFAULT_CLUB, hand=DEFAULT_HAND):
    model = make_joint_model(club_name, hand)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(path), model)
    print("Exported joint model XML:", path)


def run_baseline_viewer(
    club_name=DEFAULT_CLUB,
    hand=DEFAULT_HAND,
    speed=6.0,
    target_path=None,
    setup_pause_steps=SETUP_PAUSE_STEPS,
    finish_rollout_steps=FINISH_ROLLOUT_STEPS,
):
    model = make_joint_model(club_name, hand)
    data = mujoco.MjData(model)
    qpos_indices = joint_qpos_indices(model)
    qvel_indices = joint_qvel_indices(model)
    reset_to_baseline(model, data)
    qpos_trajectory = None
    target_path = Path(target_path) if target_path else None
    if target_path and target_path.exists():
        payload, qpos_trajectory = load_qpos_trajectory(target_path)
        qpos_trajectory = prepare_control_trajectory(qpos_trajectory)
        data.qpos[qpos_indices] = qpos_trajectory[0]
        data.qvel[qvel_indices] = 0.0
        mujoco.mj_forward(model, data)
        align_ball_to_address_clubface(model, data)
        mujoco.mj_forward(model, data)
    print("Joint-accurate two-arm baseline")
    print("Club:", get_club_preset(club_name)["label"])
    print("Controlled DOF:", len(CONTROL_JOINTS), "=", "chest + 7 left arm + 7 right arm")
    if qpos_trajectory is None:
        print("Target: internal rough keyframes.")
    else:
        print("Target:", target_path)
        print("Trajectory frames:", len(qpos_trajectory))
    print("Fixed spine/pelvis anchor is visual; chest_turn is the real world hinge.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        playhead = 0.0
        direction_age = 0
        paused_steps = 0
        top_paused_steps = 0
        top_pause_done = False
        swing_done = False
        rollout_steps = 0
        previous_ctrl = None
        while viewer.is_running():
            skip_physics_step = False
            if qpos_trajectory is None:
                if step < SWING_STEPS:
                    baseline_pd_control(model, data, qpos_indices, qvel_indices, step)
                    step += 1
                elif rollout_steps < finish_rollout_steps:
                    baseline_pd_control(model, data, qpos_indices, qvel_indices, SWING_STEPS - 1)
                    rollout_steps += 1
                else:
                    skip_physics_step = True
            else:
                top_playhead = TRANSITION_TOP_PROGRESS * (SWING_STEPS - 1)
                if not swing_done and not top_pause_done and playhead >= top_playhead:
                    playhead = top_playhead
                progress = playhead / max(SWING_STEPS - 1, 1)
                hold_setup = paused_steps < setup_pause_steps and playhead <= 0.0
                hold_top = (
                    not swing_done
                    and not top_pause_done
                    and playhead >= top_playhead
                    and top_paused_steps < TOP_PAUSE_STEPS
                )
                ramp = smootherstep(min(1.0, direction_age / 90.0))
                hold_finish = swing_done
                _, ctrl = trajectory_pd_control(
                    model,
                    data,
                    qpos_indices,
                    qvel_indices,
                    qpos_trajectory,
                    previous_ctrl=previous_ctrl,
                    progress=progress,
                    velocity_scale=0.0 if hold_finish else ramp,
                    hold_target=hold_setup or hold_top or hold_finish,
                )
                if hold_setup:
                    paused_steps += 1
                    skip_physics_step = True
                    previous_ctrl = None
                elif hold_top:
                    top_paused_steps += 1
                    skip_physics_step = True
                    previous_ctrl = None
                elif hold_finish:
                    rollout_steps += 1
                    previous_ctrl = ctrl.copy()
                else:
                    previous_ctrl = ctrl.copy()
                    if not top_pause_done and playhead >= top_playhead:
                        top_pause_done = True
                        direction_age = 0
                    playhead += max(0.35, ramp)
                    direction_age += 1
                    if playhead >= SWING_STEPS - 1:
                        playhead = float(SWING_STEPS - 1)
                        swing_done = True
                        rollout_steps = 0
                        previous_ctrl = None
            if not skip_physics_step:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * speed)


def main():
    parser = argparse.ArgumentParser(description="Run the joint-accurate two-arm baseline swing.")
    parser.add_argument("--club", default=DEFAULT_CLUB)
    parser.add_argument("--hand", default=DEFAULT_HAND, choices=("right", "left"))
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument(
        "--target",
        default=DEFAULT_IK_TRAJECTORY,
        help="Optional saved IK qpos trajectory for the torque-limited PD controller.",
    )
    parser.add_argument(
        "--setup-pause",
        type=int,
        default=SETUP_PAUSE_STEPS,
        help="Viewer frames to hold address before starting the swing.",
    )
    parser.add_argument(
        "--finish-rollout",
        type=int,
        default=FINISH_ROLLOUT_STEPS,
        help="Deprecated; the viewer now keeps physics running after the one-shot swing.",
    )
    parser.add_argument("--audit", action="store_true", help="Print the physical model assumptions and exit.")
    parser.add_argument("--export-xml", help="Save the generated MuJoCo XML to this path and exit.")
    args = parser.parse_args()
    if args.audit:
        print_physics_audit(args.club, args.hand)
        return
    if args.export_xml:
        export_joint_model_xml(args.export_xml, args.club, args.hand)
        return
    run_baseline_viewer(args.club, args.hand, args.speed, args.target, args.setup_pause, args.finish_rollout)


if __name__ == "__main__":
    main()
