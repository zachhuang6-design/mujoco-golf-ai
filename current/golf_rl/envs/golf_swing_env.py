import math
from pathlib import Path

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "GolfSwingEnv needs gymnasium. Install it with: "
        "../.venv/bin/python -m pip install gymnasium stable-baselines3"
    ) from exc

from golf_3joint_common import CLUB_PRESETS, normalize_club_name
from human_right_arm_biomech_static import (
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    make_static_model,
    normalize_hand,
)


DEFAULT_REWARD_CONFIG = {
    "impact": {
        "contact_bonus": 10.0,
        "speed_weight": 0.50,
        "face_weight": 0.0,
        "path_weight": 0.0,
        "center_weight": 0.0,
        "attack_angle_target_deg": -3.0,
        "attack_angle_weight": 0.0,
    },
    "flight": {
        "distance_weight": 0.0,
        "height_weight": 0.0,
        "target_line_weight": 0.0,
    },
    "shaping": {
        "progress_weight": 0.12,
        "downswing_progress_weight": 0.65,
        "speed_weight": 0.006,
        "plane_weight": 0.04,
        "action_weight": 0.001,
        "smoothness_weight": 0.002,
        "joint_comfort_weight": 0.18,
        "joint_velocity_weight": 0.004,
        "ground_penalty": 5.0,
        "no_contact_penalty": 15.0,
        "backswing_arc_weight": 0.08,
        "backswing_away_weight": 0.60,
        "backswing_completion_bonus": 3.0,
        "early_contact_penalty": 18.0,
        "min_backswing_arc": 0.55,
        "min_backswing_away": 0.20,
        "min_contact_step": 120.0,
    },
}


JOINT_NAMES = (
    "shoulder_turn",
    "shoulder_lift",
    "shoulder_long_axis_twist",
    "elbow_flex",
    "wrist_cock",
    "wrist_deviation",
    "wrist_roll",
)

JOINT_COMFORT_LIMITS_DEG = np.array(
    [
        [-90.0, 90.0],
        [-105.0, 105.0],
        [-55.0, 55.0],
        [0.0, 45.0],
        [-65.0, 65.0],
        [-30.0, 30.0],
        [-60.0, 60.0],
    ],
    dtype=np.float64,
)
JOINT_COMFORT_LIMITS_RAD = np.deg2rad(JOINT_COMFORT_LIMITS_DEG)
JOINT_VELOCITY_COMFORT = np.array([9.0, 9.0, 10.0, 9.0, 12.0, 12.0, 12.0], dtype=np.float64)


def deep_update(base, overrides):
    result = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def parse_simple_yaml(path):
    data = {}
    current_section = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_section = line[:-1]
                data[current_section] = {}
                continue
            if current_section is None or ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            value = value.strip()
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value
            data[current_section][key.strip()] = parsed_value
    return data


def load_reward_config(club_type, config_path=None):
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "configs" / f"{club_type}.yaml"
    if not path.exists():
        return DEFAULT_REWARD_CONFIG
    return deep_update(DEFAULT_REWARD_CONFIG, parse_simple_yaml(path))


def normalize_vector(vector, fallback):
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / length


class GolfSwingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        model_path=None,
        club_type="7iron",
        hand=DEFAULT_HAND,
        reward_config_path=None,
        max_steps=700,
        terminate_after_impact_steps=24,
        randomize_setup=False,
    ):
        super().__init__()
        self.club_type = normalize_club_name(club_type)
        self.hand = normalize_hand(hand)
        self.max_steps = int(max_steps)
        self.terminate_after_impact_steps = int(terminate_after_impact_steps)
        self.randomize_setup = bool(randomize_setup)
        self.reward_config = load_reward_config(self.club_type, reward_config_path)

        if model_path:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self.model = make_static_model(self.club_type, self.hand, club_contact=True, include_actuators=True)
        self.data = mujoco.MjData(self.model)

        self._cache_ids()
        self.torque_limits = self._actuator_torque_limits()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.step_count = 0
        self.impact_happened = False
        self.impact_step = None
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        self.clubhead_velocity = np.zeros(3, dtype=np.float64)
        self.wrist_velocity = np.zeros(3, dtype=np.float64)
        self.ball_velocity = np.zeros(3, dtype=np.float64)
        self.prev_clubhead_pos = np.zeros(3, dtype=np.float64)
        self.prev_wrist_pos = np.zeros(3, dtype=np.float64)
        self.prev_ball_pos = np.zeros(3, dtype=np.float64)
        self.prev_clubhead_dist = 0.0
        self.address_clubhead_pos = np.zeros(3, dtype=np.float64)
        self.backswing_arc = 0.0
        self.max_backswing_away = 0.0
        self.prev_backswing_arc_credit = 0.0
        self.prev_backswing_away_credit = 0.0
        self.backswing_completed = False
        self.backswing_bonus_paid = False
        self.early_contact = False
        self.plane_point = np.zeros(3, dtype=np.float64)
        self.plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.last_reward_terms = {}

        apply_setup_pose(self.model, self.data, self.hand)
        self._refresh_reference_state()
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _cache_ids(self):
        self.ball_body_id = self._body_id("ball")
        self.ball_geom_id = self._geom_id("golf_ball")
        self.floor_geom_id = self._geom_id("floor")
        self.club_head_geom_id = self._geom_id("club_head_geom")
        self.club_shaft_geom_id = self._geom_id("club_shaft_geom")
        self.clubhead_site_id = self._site_id("clubhead")
        self.club_face_site_id = self._site_id("club_face_center")
        self.wrist_site_id = self._site_id("wrist_site")
        self.shoulder_site_id = self._site_id("shoulder_site")
        self.joint_qpos_indices = np.array(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self.joint_qvel_indices = np.array(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in JOINT_NAMES],
            dtype=np.int32,
        )

    def _actuator_torque_limits(self):
        if self.model.nu == 0:
            raise ValueError("GolfSwingEnv requires actuators in the MuJoCo model.")
        limits = np.ones(self.model.nu, dtype=np.float32)
        for i in range(self.model.nu):
            low, high = self.model.actuator_ctrlrange[i]
            if self.model.actuator_ctrllimited[i]:
                limits[i] = max(abs(float(low)), abs(float(high)))
        return limits

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        apply_setup_pose(self.model, self.data, self.hand)
        if self.randomize_setup:
            qpos_noise = self.np_random.normal(0.0, 0.015, size=min(7, self.model.nq))
            self.data.qpos[: len(qpos_noise)] += qpos_noise
            mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.impact_happened = False
        self.impact_step = None
        self.previous_action[:] = 0.0
        self.clubhead_velocity[:] = 0.0
        self.wrist_velocity[:] = 0.0
        self.ball_velocity[:] = 0.0
        self._refresh_reference_state()
        self.backswing_arc = 0.0
        self.max_backswing_away = 0.0
        self.prev_backswing_arc_credit = 0.0
        self.prev_backswing_away_credit = 0.0
        self.backswing_completed = False
        self.backswing_bonus_paid = False
        self.early_contact = False
        self.last_reward_terms = {}
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action * self.torque_limits

        prev_action = self.previous_action.copy()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        self._update_velocity_cache()
        self._update_backswing_progress()
        reward, reward_terms = self._compute_reward(action, prev_action)
        self.previous_action = action.copy()

        terminated = self._is_terminal()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        info.update(reward_terms)
        self.last_reward_terms = reward_terms
        self._store_previous_positions()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _refresh_reference_state(self):
        mujoco.mj_forward(self.model, self.data)
        self._update_swing_plane()
        self._store_previous_positions()
        self.address_clubhead_pos = self._clubhead_pos().copy()
        self.prev_clubhead_dist = self._clubhead_to_ball_distance()

    def _update_velocity_cache(self):
        dt = float(self.model.opt.timestep)
        clubhead_pos = self._clubhead_pos()
        wrist_pos = self._wrist_pos()
        ball_pos = self._ball_pos()
        self.clubhead_velocity = (clubhead_pos - self.prev_clubhead_pos) / dt
        self.wrist_velocity = (wrist_pos - self.prev_wrist_pos) / dt
        self.ball_velocity = (ball_pos - self.prev_ball_pos) / dt

    def _store_previous_positions(self):
        self.prev_clubhead_pos = self._clubhead_pos().copy()
        self.prev_wrist_pos = self._wrist_pos().copy()
        self.prev_ball_pos = self._ball_pos().copy()

    def _update_swing_plane(self):
        shoulder = self._site_pos_by_id(self.shoulder_site_id)
        ball = self._ball_pos()
        target_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.plane_point = ball.copy()
        self.plane_normal = normalize_vector(np.cross(target_axis, shoulder - ball), [0.0, 1.0, 0.0])

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        clubhead_pos = self._clubhead_pos()
        wrist_pos = self._wrist_pos()
        ball_pos = self._ball_pos()
        rel = clubhead_pos - ball_pos
        shaft_direction = normalize_vector(clubhead_pos - wrist_pos, [0.0, 0.0, -1.0])
        face_normal = self._clubface_normal()
        plane_features = np.array(
            [
                self._point_plane_signed_distance(clubhead_pos),
                self._point_plane_signed_distance(wrist_pos),
                float(np.dot(self.clubhead_velocity, self.plane_normal)),
                float(np.dot(self.wrist_velocity, self.plane_normal)),
            ],
            dtype=np.float64,
        )
        shaping = self.reward_config["shaping"]
        phase_features = np.array(
            [
                self.step_count / max(1, self.max_steps),
                self.backswing_arc / max(1e-6, shaping["min_backswing_arc"]),
                self.max_backswing_away / max(1e-6, shaping["min_backswing_away"]),
                1.0 if self.backswing_completed else 0.0,
            ],
            dtype=np.float64,
        )
        obs = np.concatenate(
            [
                qpos,
                qvel,
                clubhead_pos,
                self.clubhead_velocity,
                wrist_pos,
                self.wrist_velocity,
                ball_pos,
                self.ball_velocity,
                rel,
                face_normal,
                shaft_direction,
                plane_features,
                phase_features,
                self.previous_action.astype(np.float64),
            ]
        )
        return obs.astype(np.float32)

    def _update_backswing_progress(self):
        clubhead_pos = self._clubhead_pos()
        step_arc = float(np.linalg.norm(clubhead_pos - self.prev_clubhead_pos))
        away = float(np.linalg.norm(clubhead_pos - self.address_clubhead_pos))
        if not self.impact_happened:
            self.backswing_arc += step_arc
            self.max_backswing_away = max(self.max_backswing_away, away)

        shaping = self.reward_config["shaping"]
        if (
            not self.backswing_completed
            and self.backswing_arc >= shaping["min_backswing_arc"]
            and self.max_backswing_away >= shaping["min_backswing_away"]
            and self.step_count >= int(shaping["min_contact_step"])
        ):
            self.backswing_completed = True

    def _compute_reward(self, action, previous_action):
        cfg = self.reward_config
        shaping = cfg["shaping"]
        impact_cfg = cfg["impact"]

        distance_to_ball = self._clubhead_to_ball_distance()
        progress = self.prev_clubhead_dist - distance_to_ball
        self.prev_clubhead_dist = distance_to_ball
        clubhead_speed = float(np.linalg.norm(self.clubhead_velocity))
        plane_error = self._swing_plane_error()
        action_cost = float(np.sum(action * action))
        smoothness_cost = float(np.sum((action - previous_action) ** 2))
        ball_contact = self._detect_ball_contact()
        ground_contact = self._detect_ground_contact()
        early_contact = ball_contact and not self.backswing_completed

        if self.backswing_completed:
            progress_reward = shaping["downswing_progress_weight"] * progress
        else:
            progress_reward = shaping["progress_weight"] * min(0.0, progress)
        speed_reward = shaping["speed_weight"] * clubhead_speed
        backswing_arc_credit = min(
            self.backswing_arc,
            shaping["min_backswing_arc"],
        )
        backswing_away_credit = min(
            self.max_backswing_away,
            shaping["min_backswing_away"],
        )
        backswing_arc_reward = shaping["backswing_arc_weight"] * max(
            0.0,
            backswing_arc_credit - self.prev_backswing_arc_credit,
        )
        backswing_away_reward = shaping["backswing_away_weight"] * max(
            0.0,
            backswing_away_credit - self.prev_backswing_away_credit,
        )
        self.prev_backswing_arc_credit = backswing_arc_credit
        self.prev_backswing_away_credit = backswing_away_credit
        backswing_completion_bonus = 0.0
        if self.backswing_completed and not self.backswing_bonus_paid:
            backswing_completion_bonus = shaping["backswing_completion_bonus"]
            self.backswing_bonus_paid = True
        plane_penalty = shaping["plane_weight"] * plane_error
        action_penalty = shaping["action_weight"] * action_cost
        smoothness_penalty = shaping["smoothness_weight"] * smoothness_cost
        joint_comfort_penalty = shaping["joint_comfort_weight"] * self._joint_comfort_error()
        joint_velocity_penalty = shaping["joint_velocity_weight"] * self._joint_velocity_error()
        ground_penalty = shaping["ground_penalty"] if ground_contact else 0.0
        no_contact_penalty = (
            shaping["no_contact_penalty"]
            if self.step_count >= self.max_steps and not self.impact_happened
            else 0.0
        )
        early_contact_penalty = shaping["early_contact_penalty"] if early_contact else 0.0
        ball_contact_reward = 0.0
        impact_speed_reward = 0.0
        face_penalty = 0.0
        path_penalty = 0.0
        center_penalty = 0.0
        attack_penalty = 0.0

        if ball_contact and not self.impact_happened:
            self.impact_happened = True
            self.impact_step = self.step_count
            self.early_contact = early_contact
            if not early_contact:
                face_error = self._face_angle_error()
                path_error = self._club_path_error()
                center_error = self._center_strike_error()
                attack_error = self._attack_angle_error(impact_cfg["attack_angle_target_deg"])
                ball_contact_reward = impact_cfg["contact_bonus"]
                impact_speed_reward = impact_cfg["speed_weight"] * clubhead_speed
                face_penalty = impact_cfg["face_weight"] * face_error
                path_penalty = impact_cfg["path_weight"] * path_error
                center_penalty = impact_cfg["center_weight"] * center_error
                attack_penalty = impact_cfg["attack_angle_weight"] * attack_error

        reward = (
            progress_reward
            + speed_reward
            + backswing_arc_reward
            + backswing_away_reward
            + backswing_completion_bonus
            + ball_contact_reward
            + impact_speed_reward
            - plane_penalty
            - action_penalty
            - smoothness_penalty
            - joint_comfort_penalty
            - joint_velocity_penalty
            - ground_penalty
            - no_contact_penalty
            - early_contact_penalty
            - face_penalty
            - path_penalty
            - center_penalty
            - attack_penalty
        )
        return reward, {
            "progress_reward": progress_reward,
            "speed_reward": speed_reward,
            "backswing_arc_reward": backswing_arc_reward,
            "backswing_away_reward": backswing_away_reward,
            "backswing_completion_bonus": backswing_completion_bonus,
            "ball_contact_reward": ball_contact_reward,
            "impact_speed_reward": impact_speed_reward,
            "ground_penalty": ground_penalty,
            "no_contact_penalty": no_contact_penalty,
            "early_contact_penalty": early_contact_penalty,
            "action_penalty": action_penalty,
            "smoothness_penalty": smoothness_penalty,
            "joint_comfort_penalty": joint_comfort_penalty,
            "joint_velocity_penalty": joint_velocity_penalty,
            "plane_penalty": plane_penalty,
            "face_penalty": face_penalty,
            "path_penalty": path_penalty,
            "center_penalty": center_penalty,
            "attack_penalty": attack_penalty,
            "clubhead_speed": clubhead_speed,
            "plane_error": plane_error,
            "distance_to_ball": distance_to_ball,
            "backswing_arc": self.backswing_arc,
            "max_backswing_away": self.max_backswing_away,
            "backswing_completed": self.backswing_completed,
            "early_contact": self.early_contact,
            "ball_contact": ball_contact,
            "ground_contact": ground_contact,
        }

    def _is_terminal(self):
        if self.impact_happened and self.impact_step is not None:
            return self.step_count >= self.impact_step + self.terminate_after_impact_steps
        return False

    def _get_info(self):
        return {
            "step": self.step_count,
            "impact_happened": self.impact_happened,
            "contact_names": self._contact_names(),
        }

    def _body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _site_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _joint_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _site_pos_by_id(self, site_id):
        return self.data.site_xpos[site_id].copy()

    def _clubhead_pos(self):
        return self.data.site_xpos[self.clubhead_site_id].copy()

    def _wrist_pos(self):
        return self.data.site_xpos[self.wrist_site_id].copy()

    def _ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()

    def _clubface_normal(self):
        xmat = self.data.geom_xmat[self.club_head_geom_id]
        return normalize_vector(np.array([xmat[0], xmat[3], xmat[6]], dtype=np.float64), [1.0, 0.0, 0.0])

    def _clubhead_to_ball_distance(self):
        return float(np.linalg.norm(self._clubhead_pos() - self._ball_pos()))

    def _point_plane_signed_distance(self, point):
        return float(np.dot(point - self.plane_point, self.plane_normal))

    def _swing_plane_error(self):
        clubhead = self._clubhead_pos()
        wrist = self._wrist_pos()
        shaft_mid = 0.5 * (clubhead + wrist)
        distances = [
            abs(self._point_plane_signed_distance(clubhead)),
            abs(self._point_plane_signed_distance(wrist)),
            abs(self._point_plane_signed_distance(shaft_mid)),
        ]
        velocities = [
            abs(float(np.dot(self.clubhead_velocity, self.plane_normal))),
            abs(float(np.dot(self.wrist_velocity, self.plane_normal))),
        ]
        return float(np.mean(distances) + 0.03 * np.mean(velocities))

    def _joint_comfort_error(self):
        qpos = self.data.qpos[self.joint_qpos_indices]
        low = JOINT_COMFORT_LIMITS_RAD[:, 0]
        high = JOINT_COMFORT_LIMITS_RAD[:, 1]
        scale = np.maximum(high - low, 1e-6)
        below = np.maximum(low - qpos, 0.0) / scale
        above = np.maximum(qpos - high, 0.0) / scale
        return float(np.sum((below + above) ** 2))

    def _joint_velocity_error(self):
        qvel = self.data.qvel[self.joint_qvel_indices]
        normalized = qvel / JOINT_VELOCITY_COMFORT
        return float(np.mean(normalized * normalized))

    def _detect_contact(self, geom_name_a, geom_name_b):
        geom_a = self._geom_id(geom_name_a)
        geom_b = self._geom_id(geom_name_b)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
                return True
        return False

    def _detect_ball_contact(self):
        return self._detect_contact("club_head_geom", "golf_ball")

    def _detect_ground_contact(self):
        return self._detect_contact("club_head_geom", "floor") or self._detect_contact("club_shaft_geom", "floor")

    def _contact_names(self):
        names = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            name_1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name_2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            names.append((name_1, name_2))
        return names

    def _face_angle_error(self):
        face_normal = self._clubface_normal()
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cosine = float(np.clip(np.dot(face_normal, target), -1.0, 1.0))
        return abs(math.acos(cosine))

    def _club_path_error(self):
        velocity = self.clubhead_velocity.copy()
        velocity[2] = 0.0
        direction = normalize_vector(velocity, [1.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cosine = float(np.clip(np.dot(direction, target), -1.0, 1.0))
        return abs(math.acos(cosine))

    def _center_strike_error(self):
        return float(np.linalg.norm(self._clubhead_pos() - self._ball_pos()))

    def _attack_angle_error(self, target_deg):
        velocity = self.clubhead_velocity.copy()
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        attack_angle = math.atan2(float(velocity[2]), max(horizontal_speed, 1e-9))
        return abs(attack_angle - math.radians(float(target_deg)))


if __name__ == "__main__":
    env = GolfSwingEnv()
    obs, info = env.reset(seed=0)
    print("obs_dim", obs.shape[0])
    print("action_dim", env.action_space.shape[0])
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print("reward", round(reward, 4), "terminated", terminated, "truncated", truncated)
