import mujoco
import math


INCH = 0.0254

upper_arm_len = 13 * INCH
forearm_len = 13 * INCH
club_len = 36 * INCH
ball_radius = 0.84 * INCH
ball_mass = 0.046

# Stiffer contacts make the strike behave less like a soft bump. The solref
# time constant is kept at 2x the timestep so MuJoCo can solve it stably.
CONTACT_SOLREF = "0.004 1"
CONTACT_SOLIMP = "0.99 0.995 0.001"
CLUB_CONTACT_ATTRS = (
    f'solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" '
    'condim="4" friction="0.8 0.02 0.001"'
)
BALL_CONTACT_ATTRS = (
    f'solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" '
    'condim="4" friction="0.45 0.01 0.001"'
)

CLUB_PRESETS = {
    "driver": {
        "label": "Driver",
        "loft_deg": 10.0,
        "shaft_mass": 0.060,
        "head_mass": 0.200,
    },
    "7iron": {
        "label": "7-iron",
        "loft_deg": 30.0,
        "shaft_mass": 0.100,
        "head_mass": 0.270,
    },
    "wedge": {
        "label": "Wedge",
        "loft_deg": 50.0,
        "shaft_mass": 0.100,
        "head_mass": 0.300,
    },
}

DEFAULT_CLUB_NAME = "7iron"
shaft_mass = CLUB_PRESETS[DEFAULT_CLUB_NAME]["shaft_mass"]
head_mass = CLUB_PRESETS[DEFAULT_CLUB_NAME]["head_mass"]

CLUB_LAUNCH_PROFILES = {
    "driver": {
        "label": "low launch / maximum forward speed",
        "distance_weight": 72.0,
        "height_weight": 70.0,
        "forward_speed_weight": 8.0,
        "vertical_speed_weight": 8.0,
        "target_vertical_speed": 0.55,
        "vertical_speed_tolerance": 0.75,
        "excess_vertical_speed_penalty": 90.0,
        "launch_score_weight": 180.0,
    },
    "7iron": {
        "label": "balanced launch / carry",
        "distance_weight": 55.0,
        "height_weight": 260.0,
        "forward_speed_weight": 4.0,
        "vertical_speed_weight": 30.0,
        "target_vertical_speed": 1.10,
        "vertical_speed_tolerance": 1.10,
        "excess_vertical_speed_penalty": 35.0,
        "launch_score_weight": 150.0,
    },
    "wedge": {
        "label": "high launch / height",
        "distance_weight": 34.0,
        "height_weight": 520.0,
        "forward_speed_weight": 2.5,
        "vertical_speed_weight": 58.0,
        "target_vertical_speed": 2.30,
        "vertical_speed_tolerance": 1.35,
        "excess_vertical_speed_penalty": 12.0,
        "launch_score_weight": 220.0,
    },
}

shoulder_height = 2.05
ball_x = 0.14
tee_x = ball_x
tee_center_z = 0.36
tee_half_height = 0.08
ball_tee_raise = 0.02
tee_z = tee_center_z + ball_tee_raise
ball_z = tee_z + tee_half_height + ball_radius

MAX_SHOULDER_CTRL = 10.0
MAX_ELBOW_CTRL = 4.0
MAX_WRIST_CTRL = 4.0

EPISODE_STEPS = 1200

BASELINE_CANDIDATE = {
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


def clip(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_club_name(club_name):
    normalized = club_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "driver": "driver",
        "7iron": "7iron",
        "seveniron": "7iron",
        "wedge": "wedge",
    }
    if normalized not in aliases:
        available = ", ".join(CLUB_PRESETS)
        raise ValueError(f"Unknown club '{club_name}'. Available clubs: {available}")
    return aliases[normalized]


def get_club_preset(club_name=DEFAULT_CLUB_NAME):
    return CLUB_PRESETS[normalize_club_name(club_name)]


def get_club_launch_profile(club_name=DEFAULT_CLUB_NAME):
    return CLUB_LAUNCH_PROFILES[normalize_club_name(club_name)]


def lofted_club_head_geom(name, club_name=DEFAULT_CLUB_NAME, rgba="0.2 0.2 0.2 1"):
    preset = get_club_preset(club_name)
    loft = math.radians(preset["loft_deg"])
    half_thickness = 0.018
    half_width = 0.050
    half_height = 0.032

    # The local +x face points toward the ball. Rotating about -Y gives that
    # face an upward normal, which is the 2D swing-plane equivalent of loft.
    quat_w = math.cos(-loft / 2.0)
    quat_y = math.sin(-loft / 2.0)
    normal_x = math.cos(loft)
    normal_z = math.sin(loft)
    center_x = -half_thickness * normal_x
    center_z = -club_len - half_thickness * normal_z

    return (
        f'<geom name="{name}" type="box" '
        f'pos="{center_x:.6f} 0 {center_z:.6f}" '
        f'quat="{quat_w:.6f} 0 {quat_y:.6f} 0" '
        f'size="{half_thickness:.6f} {half_width:.6f} {half_height:.6f}" '
        f'{CLUB_CONTACT_ATTRS} '
        f'mass="{preset["head_mass"]:.6f}" rgba="{rgba}"/>'
    )


def build_single_arm_xml(club_name=DEFAULT_CLUB_NAME):
    preset = get_club_preset(club_name)
    club_head_xml = lofted_club_head_geom("club_head_geom", club_name)
    return f"""
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
          <geom name="club_shaft_geom" type="capsule" fromto="0 0 0 0 0 -{club_len:.6f}" size="0.012" mass="{preset["shaft_mass"]:.6f}" rgba="0.1 0.1 0.1 1"/>
          {club_head_xml}
          <site name="club_tip" pos="0 0 -{club_len:.6f}" size="0.008" rgba="1 0 0 0.45"/>
          <site name="club_face_center" pos="0 0 -{club_len:.6f}" size="0.012" rgba="0 0.6 1 0.7"/>
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


def make_single_arm_model(club_name=DEFAULT_CLUB_NAME):
    return mujoco.MjModel.from_xml_string(build_single_arm_xml(club_name))


def controls_for_step(candidate, step):
    if step < candidate["t1"]:
        controls = candidate["s1"], candidate["e1"], candidate["w1"]
    elif step < candidate["t2"]:
        controls = candidate["s2"], candidate["e2"], candidate["w2"]
    elif step < candidate["t3"]:
        controls = candidate["s3"], candidate["e3"], candidate["w3"]
    else:
        controls = 0.0, 0.0, 0.0

    return (
        clip(controls[0], -MAX_SHOULDER_CTRL, MAX_SHOULDER_CTRL),
        clip(controls[1], -MAX_ELBOW_CTRL, MAX_ELBOW_CTRL),
        clip(controls[2], -MAX_WRIST_CTRL, MAX_WRIST_CTRL),
    )


def apply_three_joint_controls(data, candidate, step, ctrl_offset=0):
    shoulder, elbow, wrist = controls_for_step(candidate, step)
    data.ctrl[ctrl_offset] = shoulder
    data.ctrl[ctrl_offset + 1] = elbow
    data.ctrl[ctrl_offset + 2] = wrist


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
    print("\n# Copy these control assignments into human_golf_arm3joint.py:\n")
    print(f"if step < {candidate['t1']}:")
    print(f"    data.ctrl[0] = {candidate['s1']:.3f}")
    print(f"    data.ctrl[1] = {candidate['e1']:.3f}")
    print(f"    data.ctrl[2] = {candidate['w1']:.3f}")
    print(f"elif step < {candidate['t2']}:")
    print(f"    data.ctrl[0] = {candidate['s2']:.3f}")
    print(f"    data.ctrl[1] = {candidate['e2']:.3f}")
    print(f"    data.ctrl[2] = {candidate['w2']:.3f}")
    print(f"elif step < {candidate['t3']}:")
    print(f"    data.ctrl[0] = {candidate['s3']:.3f}")
    print(f"    data.ctrl[1] = {candidate['e3']:.3f}")
    print(f"    data.ctrl[2] = {candidate['w3']:.3f}")
    print("else:")
    print("    data.ctrl[0] = 0.0")
    print("    data.ctrl[1] = 0.0")
    print("    data.ctrl[2] = 0.0")
