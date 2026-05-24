import math

import mujoco

from golf_3joint_common import (
    BALL_CONTACT_ATTRS,
    CLUB_PRESETS,
    DEFAULT_CLUB_NAME,
    EPISODE_STEPS,
    MAX_ELBOW_CTRL,
    MAX_SHOULDER_CTRL,
    MAX_WRIST_CTRL,
    ball_mass,
    ball_radius,
    ball_x,
    ball_z,
    club_len,
    forearm_len,
    get_club_launch_profile,
    get_club_preset,
    lofted_club_head_geom,
    normalize_club_name,
    tee_half_height,
    tee_x,
    tee_z,
    upper_arm_len,
)


DEFAULT_PLANE_ANGLE_DEG = 55.0

# Address is no longer the old perfectly straight hanging chain. The shoulder
# stays nearly vertical, the elbow has a small golfer-like bend, and the wrist
# offsets that bend so the club can still sit in the swing-plane setup.
ADDRESS_POSE = (
    math.radians(0.0),
    math.radians(18.0),
    math.radians(-18.0),
)

TOP_SET_POSE = (
    math.radians(90.0),
    math.radians(90.0),
    math.radians(90.0),
)

IMPACT_ELBOW_TARGET = math.radians(15.0)
IMPACT_ELBOW_TOLERANCE = math.radians(10.0)
TOP_SET_TOLERANCE = math.radians(18.0)
ADDRESS_TOLERANCE = math.radians(8.0)

JOINT_LIMITS_DEG = {
    "shoulder": (-160.0, 160.0),
    "elbow": (0.0, 135.0),
    "wrist": (-120.0, 120.0),
}

CTRL_LIMITS = (MAX_SHOULDER_CTRL, MAX_ELBOW_CTRL, MAX_WRIST_CTRL)


def clip(value, low, high):
    return max(low, min(high, value))


def plane_roll_from_angle(plane_angle_deg):
    return math.radians(90.0 - plane_angle_deg)


def x_rotation_quat(angle):
    return math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0


def rotate_x(vector, angle):
    x, y, z = vector
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        x,
        y * c - z * s,
        y * s + z * c,
    )


def segment_vector(length, angle):
    return (
        length * math.sin(angle),
        0.0,
        -length * math.cos(angle),
    )


def local_tip_position_for_pose(pose):
    shoulder, elbow, wrist = pose
    upper_angle = shoulder
    forearm_angle = shoulder + elbow
    club_angle = shoulder + elbow + wrist
    vectors = (
        segment_vector(upper_arm_len, upper_angle),
        segment_vector(forearm_len, forearm_angle),
        segment_vector(club_len, club_angle),
    )
    return tuple(sum(vector[i] for vector in vectors) for i in range(3))


def shoulder_position_for_address(plane_angle_deg):
    roll = plane_roll_from_angle(plane_angle_deg)
    local_tip = local_tip_position_for_pose(ADDRESS_POSE)
    world_tip_offset = rotate_x(local_tip, roll)

    # Start with the clubhead just behind the teed ball, like the 2D baseline,
    # while allowing the shoulder height to be determined by the address shape.
    target_tip_world = (ball_x - 0.14, 0.0, ball_z - 0.006)
    return tuple(target_tip_world[i] - world_tip_offset[i] for i in range(3))


def format_tuple(values):
    return " ".join(f"{value:.6f}" for value in values)


def build_plane_arm_xml(club_name=DEFAULT_CLUB_NAME, plane_angle_deg=DEFAULT_PLANE_ANGLE_DEG):
    club_name = normalize_club_name(club_name)
    preset = get_club_preset(club_name)
    roll = plane_roll_from_angle(plane_angle_deg)
    quat = x_rotation_quat(roll)
    shoulder_pos = shoulder_position_for_address(plane_angle_deg)
    club_head_xml = lofted_club_head_geom("club_head_geom", club_name)

    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0.235" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="swing_plane" pos="{format_tuple(shoulder_pos)}" quat="{format_tuple(quat)}">
      <geom name="swing_plane_visual" type="box" pos="0 0 -0.700000" size="1.400000 0.002000 1.050000" rgba="0.2 0.55 1 0.12" contype="0" conaffinity="0"/>

      <body name="arm">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="{JOINT_LIMITS_DEG["shoulder"][0]} {JOINT_LIMITS_DEG["shoulder"][1]}" damping="0.8" armature="0.04"/>
        <geom name="upper_arm" type="capsule" fromto="0 0 0 0 0 -{upper_arm_len:.6f}" size="0.04" rgba="0.3 0.3 0.9 1"/>

        <body name="forearm" pos="0 0 -{upper_arm_len:.6f}">
          <joint name="elbow" type="hinge" axis="0 1 0" range="{JOINT_LIMITS_DEG["elbow"][0]} {JOINT_LIMITS_DEG["elbow"][1]}" damping="0.45" armature="0.02"/>
          <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 -{forearm_len:.6f}" size="0.035" rgba="0.3 0.9 0.3 1"/>

          <body name="club" pos="0 0 -{forearm_len:.6f}">
            <joint name="wrist" type="hinge" axis="0 1 0" range="{JOINT_LIMITS_DEG["wrist"][0]} {JOINT_LIMITS_DEG["wrist"][1]}" damping="0.45" armature="0.02"/>
            <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" mass="{preset["shaft_mass"]:.6f}" rgba="0.1 0.1 0.1 1"/>
            {club_head_xml}
            <site name="club_tip" pos="0 0 -{club_len:.6f}" size="0.008" rgba="1 0 0 0.45"/>
            <site name="club_face_center" pos="0 0 -{club_len:.6f}" size="0.012" rgba="0 0.6 1 0.7"/>
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
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="7" ctrlrange="-{MAX_SHOULDER_CTRL} {MAX_SHOULDER_CTRL}" ctrllimited="true"/>
    <motor joint="elbow" gear="5" ctrlrange="-{MAX_ELBOW_CTRL} {MAX_ELBOW_CTRL}" ctrllimited="true"/>
    <motor joint="wrist" gear="5" ctrlrange="-{MAX_WRIST_CTRL} {MAX_WRIST_CTRL}" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def make_plane_model(club_name=DEFAULT_CLUB_NAME, plane_angle_deg=DEFAULT_PLANE_ANGLE_DEG):
    model = mujoco.MjModel.from_xml_string(build_plane_arm_xml(club_name, plane_angle_deg))
    model.qpos0[:3] = ADDRESS_POSE
    return model


def get_plane_launch_profile(club_name=DEFAULT_CLUB_NAME):
    return get_club_launch_profile(club_name)


def print_plane_setup(club_name, plane_angle_deg):
    preset = get_club_preset(club_name)
    launch_profile = get_plane_launch_profile(club_name)
    print(
        "Selected club:",
        preset["label"],
        "loft_deg",
        preset["loft_deg"],
        "shaft_mass",
        preset["shaft_mass"],
        "head_mass",
        preset["head_mass"],
    )
    print("Plane angle deg:", plane_angle_deg)
    print("Address pose:", tuple(round(math.degrees(v), 2) for v in ADDRESS_POSE))
    print("Top set pose:", tuple(round(math.degrees(v), 2) for v in TOP_SET_POSE))
    print("Launch profile:", launch_profile["label"])
