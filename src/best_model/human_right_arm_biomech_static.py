import argparse
import math
import time

import mujoco
import mujoco.viewer

from golf_3joint_common import (
    BALL_CONTACT_ATTRS,
    CLUB_PRESETS,
    ball_mass,
    ball_radius,
    ball_x,
    ball_z,
    club_len,
    forearm_len,
    get_club_preset,
    lofted_club_head_geom,
    tee_half_height,
    tee_x,
    tee_z,
    upper_arm_len,
)


UPPER_ARM_Y_LEAN_DEG = 7.5
CLUB_MINOR_ANGLE_TO_POSITIVE_Y_DEG = 60.0
TARGET_CLUB_TIP_OFFSET_X = -0.035
TARGET_CLUB_TIP_OFFSET_Z = -0.006
DEFAULT_HAND = "right"
HAND_SIGNS = {
    "left": 1.0,
    "right": -1.0,
}

SHOULDER_TURN_RANGE_DEG = (-110.0, 110.0)
SHOULDER_LIFT_RANGE_DEG = (-150.0, 150.0)
SHOULDER_TWIST_RANGE_DEG = (-90.0, 90.0)
ELBOW_FLEX_RANGE_DEG = (0.0, 45.0)
WRIST_COCK_RANGE_DEG = (-80.0, 80.0)
WRIST_DEVIATION_RANGE_DEG = (-55.0, 55.0)
WRIST_ROLL_RANGE_DEG = (-95.0, 95.0)

CTRL_LIMITS = {
    "shoulder_turn": 9.0,
    "shoulder_lift": 12.0,
    "shoulder_long_axis_twist": 8.0,
    "elbow_flex": 6.0,
    "wrist_cock": 5.0,
    "wrist_deviation": 4.0,
    "wrist_roll": 4.0,
}


def deg(value):
    return math.radians(value)


def format_vec(values):
    return " ".join(f"{value:.6f}" for value in values)


def normalize_hand(hand):
    normalized = hand.lower().strip()
    if normalized not in HAND_SIGNS:
        raise ValueError(f"Unknown hand '{hand}'. Use 'right' or 'left'.")
    return normalized


def hand_sign(hand):
    return HAND_SIGNS[normalize_hand(hand)]


def segment_yz(length, angle_from_vertical_deg, hand=DEFAULT_HAND):
    angle = deg(angle_from_vertical_deg)
    return (
        0.0,
        hand_sign(hand) * length * math.sin(angle),
        -length * math.cos(angle),
    )


def setup_angles():
    arm_angle = UPPER_ARM_Y_LEAN_DEG
    club_angle_from_vertical = 90.0 - CLUB_MINOR_ANGLE_TO_POSITIVE_Y_DEG
    wrist_cock = club_angle_from_vertical - arm_angle
    return arm_angle, club_angle_from_vertical, wrist_cock


def setup_positions(hand=DEFAULT_HAND):
    arm_angle, club_angle_from_vertical, _ = setup_angles()
    upper = segment_yz(upper_arm_len, arm_angle, hand)
    forearm = segment_yz(forearm_len, arm_angle, hand)
    club = segment_yz(club_len, club_angle_from_vertical, hand)
    chain = tuple(upper[i] + forearm[i] + club[i] for i in range(3))
    target_tip = (
        ball_x + TARGET_CLUB_TIP_OFFSET_X,
        0.0,
        ball_z + TARGET_CLUB_TIP_OFFSET_Z,
    )
    shoulder = tuple(target_tip[i] - chain[i] for i in range(3))
    elbow = tuple(shoulder[i] + upper[i] for i in range(3))
    wrist = tuple(elbow[i] + forearm[i] for i in range(3))
    tip = tuple(wrist[i] + club[i] for i in range(3))
    return shoulder, elbow, wrist, tip


def axis_visuals_xml():
    return """
    <geom name="x_axis_target" type="capsule" fromto="0 0 0.255 0.850 0 0.255" size="0.006" rgba="1 0.05 0.05 1" contype="0" conaffinity="0"/>
    <geom name="y_axis_camera" type="capsule" fromto="0 0 0.275 0 0.850 0.275" size="0.006" rgba="0.05 0.75 0.15 1" contype="0" conaffinity="0"/>
    <geom name="z_axis_up" type="capsule" fromto="0 0 0.255 0 0 1.050" size="0.006" rgba="0.05 0.25 1 1" contype="0" conaffinity="0"/>
    <site name="x_axis_tip" pos="0.850 0 0.255" size="0.025" rgba="1 0.05 0.05 1"/>
    <site name="y_axis_tip" pos="0 0.850 0.275" size="0.025" rgba="0.05 0.75 0.15 1"/>
    <site name="z_axis_tip" pos="0 0 1.050" size="0.025" rgba="0.05 0.25 1 1"/>
"""


def build_static_xml(
    club_name="7iron",
    hand=DEFAULT_HAND,
    club_contact=False,
    include_actuators=False,
):
    hand = normalize_hand(hand)
    preset = get_club_preset(club_name)
    shoulder, elbow, wrist, tip = setup_positions(hand)
    arm_angle, _, _ = setup_angles()
    club_head_xml = lofted_club_head_geom("club_head_geom", club_name)
    if not club_contact:
        club_head_xml = club_head_xml.replace(
            'condim="4"',
            'contype="0" conaffinity="0" condim="4"',
        )
    actuator_xml = ""
    if include_actuators:
        actuator_xml = f"""
  <actuator>
    <motor joint="shoulder_turn" gear="6" ctrlrange="-{CTRL_LIMITS["shoulder_turn"]} {CTRL_LIMITS["shoulder_turn"]}" ctrllimited="true"/>
    <motor joint="shoulder_lift" gear="7" ctrlrange="-{CTRL_LIMITS["shoulder_lift"]} {CTRL_LIMITS["shoulder_lift"]}" ctrllimited="true"/>
    <motor joint="shoulder_long_axis_twist" gear="5" ctrlrange="-{CTRL_LIMITS["shoulder_long_axis_twist"]} {CTRL_LIMITS["shoulder_long_axis_twist"]}" ctrllimited="true"/>
    <motor joint="elbow_flex" gear="5" ctrlrange="-{CTRL_LIMITS["elbow_flex"]} {CTRL_LIMITS["elbow_flex"]}" ctrllimited="true"/>
    <motor joint="wrist_cock" gear="4" ctrlrange="-{CTRL_LIMITS["wrist_cock"]} {CTRL_LIMITS["wrist_cock"]}" ctrllimited="true"/>
    <motor joint="wrist_deviation" gear="3" ctrlrange="-{CTRL_LIMITS["wrist_deviation"]} {CTRL_LIMITS["wrist_deviation"]}" ctrllimited="true"/>
    <motor joint="wrist_roll" gear="3" ctrlrange="-{CTRL_LIMITS["wrist_roll"]} {CTRL_LIMITS["wrist_roll"]}" ctrllimited="true"/>
  </actuator>
"""

    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>
    {axis_visuals_xml()}

    <body name="shoulder_turn_frame" pos="{format_vec(shoulder)}">
      <inertial mass="0.001" pos="0 0 0" diaginertia="0.000001 0.000001 0.000001"/>
      <joint name="shoulder_turn" type="hinge" axis="0 0 1" range="{SHOULDER_TURN_RANGE_DEG[0]} {SHOULDER_TURN_RANGE_DEG[1]}" damping="1.0" armature="0.04"/>

      <body name="shoulder_lift_frame">
        <inertial mass="0.001" pos="0 0 0" diaginertia="0.000001 0.000001 0.000001"/>
        <joint name="shoulder_lift" type="hinge" axis="1 0 0" range="{SHOULDER_LIFT_RANGE_DEG[0]} {SHOULDER_LIFT_RANGE_DEG[1]}" damping="1.0" armature="0.04"/>

        <body name="upper_arm_twist_frame">
        <joint name="shoulder_long_axis_twist" type="hinge" axis="0 0 1" range="{SHOULDER_TWIST_RANGE_DEG[0]} {SHOULDER_TWIST_RANGE_DEG[1]}" damping="0.8" armature="0.03"/>
        <geom name="upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="0.3 0.3 0.9 1" contype="0" conaffinity="0"/>
        <geom name="upper_arm_twist_marker" type="box" pos="0.045 0 -{upper_arm_len / 2:.6f}" size="0.012 0.012 {upper_arm_len / 2:.6f}" rgba="0.9 0.9 0.15 0.85" contype="0" conaffinity="0"/>
        <site name="shoulder_site" pos="0 0 0" size="0.035" rgba="0.15 0.15 1 1"/>
        <site name="elbow_site" pos="0 0 -{upper_arm_len:.6f}" size="0.03" rgba="0 0.8 0.2 1"/>

        <body name="elbow_flex_frame" pos="0 0 -{upper_arm_len:.6f}">
          <joint name="elbow_flex" type="hinge" axis="-1 0 0" range="{ELBOW_FLEX_RANGE_DEG[0]} {ELBOW_FLEX_RANGE_DEG[1]}" damping="0.6" armature="0.02"/>
          <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1" contype="0" conaffinity="0"/>
          <geom name="elbow_flex_plane_marker" type="box" pos="0 0.015 -{forearm_len / 2:.6f}" size="0.010 0.004 {forearm_len / 2:.6f}" rgba="1 0.55 0 0.9" contype="0" conaffinity="0"/>
          <site name="wrist_site" pos="0 0 -{forearm_len:.6f}" size="0.026" rgba="0 0.75 0.9 1"/>

          <body name="wrist_cock_frame" pos="0 0 -{forearm_len:.6f}">
            <inertial mass="0.001" pos="0 0 0" diaginertia="0.000001 0.000001 0.000001"/>
            <joint name="wrist_cock" type="hinge" axis="1 0 0" range="{WRIST_COCK_RANGE_DEG[0]} {WRIST_COCK_RANGE_DEG[1]}" damping="0.35" armature="0.01"/>

            <body name="wrist_deviation_frame">
              <inertial mass="0.001" pos="0 0 0" diaginertia="0.000001 0.000001 0.000001"/>
              <joint name="wrist_deviation" type="hinge" axis="0 1 0" range="{WRIST_DEVIATION_RANGE_DEG[0]} {WRIST_DEVIATION_RANGE_DEG[1]}" damping="0.35" armature="0.01"/>

              <body name="wrist_roll_frame">
                <joint name="wrist_roll" type="hinge" axis="0 0 1" range="{WRIST_ROLL_RANGE_DEG[0]} {WRIST_ROLL_RANGE_DEG[1]}" damping="0.25" armature="0.01"/>
                <geom name="wrist_hemisphere_reference" type="sphere" pos="0 0 0" size="0.075" rgba="0.1 0.8 1 0.18" contype="0" conaffinity="0"/>
                <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" mass="{preset["shaft_mass"]:.6f}" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
                {club_head_xml}
                <site name="club_tip" pos="0 0 -{club_len:.6f}" size="0.01" rgba="1 0 0 0.55"/>
                <site name="club_face_center" pos="0 0 -{club_len:.6f}" size="0.012" rgba="0 0.6 1 0.7"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    </body>

    <body name="tee" pos="{tee_x:.6f} 0 {tee_z:.6f}">
      <geom name="tee_geom" type="cylinder" size="0.01 {tee_half_height:.6f}" rgba="1 0.5 0 1"/>
    </body>

    <body name="ball" pos="{ball_x:.6f} 0 {ball_z:.6f}">
      <joint type="free"/>
      <geom name="golf_ball" type="sphere" size="{ball_radius:.6f}" mass="{ball_mass}" {BALL_CONTACT_ATTRS} rgba="1 1 1 1"/>
    </body>

    <geom name="setup_upper_arm_line" type="capsule" fromto="{format_vec(shoulder)} {format_vec(elbow)}" size="0.006" rgba="0.1 0.1 1 0.35" contype="0" conaffinity="0"/>
    <geom name="setup_forearm_line" type="capsule" fromto="{format_vec(elbow)} {format_vec(wrist)}" size="0.006" rgba="0.1 1 0.1 0.35" contype="0" conaffinity="0"/>
    <geom name="setup_club_line" type="capsule" fromto="{format_vec(wrist)} {format_vec(tip)}" size="0.005" rgba="0 0 0 0.35" contype="0" conaffinity="0"/>
  </worldbody>
  {actuator_xml}
</mujoco>
"""


def make_static_model(
    club_name="7iron",
    hand=DEFAULT_HAND,
    club_contact=False,
    include_actuators=False,
):
    return mujoco.MjModel.from_xml_string(
        build_static_xml(club_name, hand, club_contact, include_actuators)
    )


def apply_setup_pose(model, data, hand=DEFAULT_HAND):
    hand = normalize_hand(hand)
    data.qpos[:] = model.qpos0
    _, _, wrist_cock = setup_angles()
    joint_values = {
        "shoulder_turn": 0.0,
        "shoulder_lift": hand_sign(hand) * deg(UPPER_ARM_Y_LEAN_DEG),
        "shoulder_long_axis_twist": 0.0,
        "elbow_flex": 0.0,
        "wrist_cock": hand_sign(hand) * deg(wrist_cock),
        "wrist_deviation": 0.0,
        "wrist_roll": 0.0,
    }
    for joint_name, value in joint_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def print_setup_report(model, data, hand=DEFAULT_HAND):
    hand = normalize_hand(hand)
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shoulder_site")
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "elbow_site")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wrist_site")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    shoulder = data.site_xpos[shoulder_id]
    elbow = data.site_xpos[elbow_id]
    wrist = data.site_xpos[wrist_id]
    club_tip = data.site_xpos[club_tip_id]
    ball = data.xpos[ball_id]

    arm_angle, club_angle, wrist_cock = setup_angles()
    print("Coordinate convention: +x target, +y toward camera, +z up")
    print("Handedness:", hand)
    print("Upper arm y-lean deg:", round(arm_angle, 3))
    print("Club shaft minor angle to +y deg:", CLUB_MINOR_ANGLE_TO_POSITIVE_Y_DEG)
    print("Wrist cock relative to straight arm deg:", round(wrist_cock, 3))
    print("Shoulder:", [round(float(v), 4) for v in shoulder])
    print("Elbow:", [round(float(v), 4) for v in elbow])
    print("Wrist:", [round(float(v), 4) for v in wrist])
    print("Club tip:", [round(float(v), 4) for v in club_tip])
    print("Ball:", [round(float(v), 4) for v in ball])
    print("Tip-to-ball distance:", round(math.dist(club_tip, ball), 4))
    print("setup qpos deg:", [round(math.degrees(float(data.qpos[i])), 3) for i in range(7)])


def parse_args():
    parser = argparse.ArgumentParser(description="Static right-arm biomechanical setup viewer.")
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = make_static_model(args.club, args.hand)
    data = mujoco.MjData(model)
    apply_setup_pose(model, data, args.hand)
    print_setup_report(model, data, args.hand)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Keep this as a static setup verifier. The joints are present, but
            # no controller is applied yet.
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 20)
