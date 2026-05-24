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

from golf_core.common import CLUB_PRESETS, normalize_club_name
from golf_core.right_arm_static import (
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    make_static_model,
    normalize_hand,
)


DEFAULT_REWARD_CONFIG = {
    "rewards": {
        "clubhead_x_velocity_weight": 2.0,
        "ball_distance_weight": 12.0,
        "ball_height_weight": 8.0,
    },
    "plane": {
        "position_weight": 18.0,
        "velocity_weight": 1.2,
        "position_improvement_weight": 3.0,
        "velocity_improvement_weight": 0.25,
        "position_tolerance": 0.025,
        "velocity_tolerance": 0.75,
        "barrier_limit": 4.0,
        "overflow_weight": 12.0,
        "max_loss": 35.0,
        "takeaway_multiplier": 3.0,
        "backswing_multiplier": 1.8,
        "downswing_multiplier": 1.0,
        "followthrough_multiplier": 2.6,
    },
    "flight": {
        "target_line_position_weight": 14.0,
        "target_line_velocity_weight": 1.8,
        "target_line_position_tolerance": 0.08,
        "target_line_velocity_tolerance": 1.0,
        "target_line_barrier_limit": 4.0,
        "target_line_overflow_weight": 5.0,
        "target_line_max_loss": 25.0,
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
        max_steps=1000,
        terminate_after_impact_steps=120,
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
        self.initial_ball_pos = np.zeros(3, dtype=np.float64)
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_height = 0.0
        self.prev_plane_position_error = 0.0
        self.prev_plane_velocity_error = 0.0
        self.backswing_arc = 0.0
        self.max_backswing_away = 0.0
        self.max_backswing_depth = 0.0
        self.max_backswing_height = 0.0
        self.prev_backswing_arc_credit = 0.0
        self.prev_backswing_away_credit = 0.0
        self.prev_backswing_depth_credit = 0.0
        self.prev_backswing_height_credit = 0.0
        self.backswing_completed = False
        self.backswing_bonus_paid = False
        self.early_contact = False
        self.valid_impact = False
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
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_height = 0.0
        self.prev_plane_position_error = 0.0
        self.prev_plane_velocity_error = 0.0
        self._refresh_reference_state()
        self.backswing_arc = 0.0
        self.max_backswing_away = 0.0
        self.max_backswing_depth = 0.0
        self.max_backswing_height = 0.0
        self.prev_backswing_arc_credit = 0.0
        self.prev_backswing_away_credit = 0.0
        self.prev_backswing_depth_credit = 0.0
        self.prev_backswing_height_credit = 0.0
        self.backswing_completed = False
        self.backswing_bonus_paid = False
        self.early_contact = False
        self.valid_impact = False
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
        self.initial_ball_pos = self._ball_pos().copy()
        self.prev_clubhead_dist = self._clubhead_to_ball_distance()
        plane_terms = self._club_plane_terms()
        self.prev_plane_position_error = plane_terms["position_error"]
        self.prev_plane_velocity_error = plane_terms["velocity_error"]

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
        phase_features = np.array(
            [
                self.step_count / max(1, self.max_steps),
                self.max_clubhead_x_velocity / 25.0,
                self.max_ball_x_distance / 12.0,
                self.max_ball_height / 2.0,
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
        depth = max(0.0, float(self.address_clubhead_pos[0] - clubhead_pos[0]))
        height = max(0.0, float(clubhead_pos[2] - self.address_clubhead_pos[2]))
        if not self.impact_happened:
            self.backswing_arc += step_arc
            self.max_backswing_away = max(self.max_backswing_away, away)
            self.max_backswing_depth = max(self.max_backswing_depth, depth)
            self.max_backswing_height = max(self.max_backswing_height, height)

    def _compute_reward(self, action, previous_action):
        cfg = self.reward_config
        reward_cfg = cfg["rewards"]
        plane_cfg = cfg["plane"]
        flight_cfg = cfg["flight"]
        distance_to_ball = self._clubhead_to_ball_distance()
        self.prev_clubhead_dist = distance_to_ball
        clubhead_speed = float(np.linalg.norm(self.clubhead_velocity))
        ball_contact = self._detect_ball_contact()

        if ball_contact and not self.impact_happened:
            self.impact_happened = True
            self.impact_step = self.step_count
            self.valid_impact = True

        current_clubhead_x_velocity = max(0.0, float(self.clubhead_velocity[0]))
        current_ball_x_distance = max(0.0, float(self._ball_pos()[0] - self.initial_ball_pos[0]))
        current_ball_height = max(0.0, float(self._ball_pos()[2] - self.initial_ball_pos[2]))

        previous_max_clubhead_x_velocity = self.max_clubhead_x_velocity
        previous_max_ball_x_distance = self.max_ball_x_distance
        previous_max_ball_height = self.max_ball_height
        self.max_clubhead_x_velocity = max(self.max_clubhead_x_velocity, current_clubhead_x_velocity)
        self.max_ball_x_distance = max(self.max_ball_x_distance, current_ball_x_distance)
        self.max_ball_height = max(self.max_ball_height, current_ball_height)

        clubhead_x_velocity_reward = reward_cfg["clubhead_x_velocity_weight"] * (
            self.max_clubhead_x_velocity - previous_max_clubhead_x_velocity
        )
        ball_distance_reward = reward_cfg["ball_distance_weight"] * (
            self.max_ball_x_distance - previous_max_ball_x_distance
        )
        ball_height_reward = reward_cfg["ball_height_weight"] * (
            self.max_ball_height - previous_max_ball_height
        )

        plane_terms = self._club_plane_terms(plane_cfg)
        plane_phase_multiplier = self._plane_phase_multiplier()
        plane_position_improvement_reward = (
            plane_phase_multiplier
            * plane_cfg["position_improvement_weight"]
            * max(0.0, self.prev_plane_position_error - plane_terms["position_error"])
        )
        plane_velocity_improvement_reward = (
            plane_phase_multiplier
            * plane_cfg["velocity_improvement_weight"]
            * max(0.0, self.prev_plane_velocity_error - plane_terms["velocity_error"])
        )
        self.prev_plane_position_error = plane_terms["position_error"]
        self.prev_plane_velocity_error = plane_terms["velocity_error"]
        plane_position_penalty = plane_phase_multiplier * plane_cfg["position_weight"] * plane_terms["position_loss"]
        plane_velocity_penalty = plane_phase_multiplier * plane_cfg["velocity_weight"] * plane_terms["velocity_loss"]

        line_terms = self._target_line_terms(flight_cfg)
        target_line_position_penalty = flight_cfg["target_line_position_weight"] * line_terms["position_loss"]
        target_line_velocity_penalty = flight_cfg["target_line_velocity_weight"] * line_terms["velocity_loss"]

        reward = (
            clubhead_x_velocity_reward
            + ball_distance_reward
            + ball_height_reward
            + plane_position_improvement_reward
            + plane_velocity_improvement_reward
            - plane_position_penalty
            - plane_velocity_penalty
            - target_line_position_penalty
            - target_line_velocity_penalty
        )
        return reward, {
            "clubhead_x_velocity_reward": clubhead_x_velocity_reward,
            "ball_distance_reward": ball_distance_reward,
            "ball_height_reward": ball_height_reward,
            "plane_position_improvement_reward": plane_position_improvement_reward,
            "plane_velocity_improvement_reward": plane_velocity_improvement_reward,
            "plane_position_penalty": plane_position_penalty,
            "plane_velocity_penalty": plane_velocity_penalty,
            "plane_penalty": plane_position_penalty + plane_velocity_penalty,
            "target_line_position_penalty": target_line_position_penalty,
            "target_line_velocity_penalty": target_line_velocity_penalty,
            "target_line_penalty": target_line_position_penalty + target_line_velocity_penalty,
            "clubhead_speed": clubhead_speed,
            "clubhead_x_velocity": current_clubhead_x_velocity,
            "max_clubhead_x_velocity": self.max_clubhead_x_velocity,
            "ball_x_distance": current_ball_x_distance,
            "max_ball_x_distance": self.max_ball_x_distance,
            "ball_height": current_ball_height,
            "max_ball_height": self.max_ball_height,
            "ball_lateral_error": line_terms["position_error"],
            "ball_lateral_velocity": line_terms["velocity_error"],
            "plane_phase_multiplier": plane_phase_multiplier,
            "plane_position_error": plane_terms["position_error"],
            "plane_velocity_error": plane_terms["velocity_error"],
            "plane_error": plane_terms["combined_error"],
            "clubhead_plane_distance": plane_terms["clubhead_position_error"],
            "shaft_mid_plane_distance": plane_terms["shaft_mid_position_error"],
            "wrist_plane_distance": plane_terms["wrist_position_error"],
            "clubhead_plane_velocity": plane_terms["clubhead_velocity_error"],
            "shaft_mid_plane_velocity": plane_terms["shaft_mid_velocity_error"],
            "wrist_plane_velocity": plane_terms["wrist_velocity_error"],
            "distance_to_ball": distance_to_ball,
            "valid_impact": self.valid_impact,
            "ball_contact": ball_contact,
        }

    def _is_terminal(self):
        if self.impact_happened and self.impact_step is not None:
            return self.step_count >= self.impact_step + self.terminate_after_impact_steps
        return False

    def _get_info(self):
        return {
            "step": self.step_count,
            "impact_happened": self.impact_happened,
            "valid_impact": self.valid_impact,
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

    def _clubhead_forward_velocity(self):
        return float(self.clubhead_velocity[0])

    def _point_plane_signed_distance(self, point):
        return float(np.dot(point - self.plane_point, self.plane_normal))

    def _club_plane_points(self):
        clubhead = self._clubhead_pos()
        wrist = self._wrist_pos()
        shaft_mid = 0.5 * (clubhead + wrist)
        return {
            "wrist": wrist,
            "shaft_mid": shaft_mid,
            "clubhead": clubhead,
        }

    def _negative_log_plane_loss(self, value, tolerance, barrier_limit, overflow_weight, max_loss=None):
        normalized = abs(float(value)) / max(float(tolerance), 1e-9)
        barrier_position = min(normalized / max(float(barrier_limit), 1e-9), 0.999999)
        barrier_loss = -math.log(1.0 - barrier_position)
        overflow = max(0.0, normalized - float(barrier_limit))
        loss = float(barrier_loss + float(overflow_weight) * overflow * overflow)
        if max_loss is not None:
            loss = min(loss, float(max_loss))
        return loss

    def _club_plane_terms(self, plane_cfg=None):
        plane_cfg = plane_cfg or self.reward_config["plane"]
        position_tolerance = plane_cfg["position_tolerance"]
        velocity_tolerance = plane_cfg["velocity_tolerance"]
        barrier_limit = plane_cfg["barrier_limit"]
        overflow_weight = plane_cfg["overflow_weight"]
        max_loss = plane_cfg.get("max_loss")

        position_errors = {}
        velocity_errors = {}
        position_losses = []
        velocity_losses = []
        for name, point in self._club_plane_points().items():
            signed_distance = self._point_plane_signed_distance(point)
            point_normal = self.plane_normal if signed_distance >= 0.0 else -self.plane_normal
            normal_velocity = abs(float(np.dot(self.clubhead_velocity, point_normal)))
            position_error = abs(signed_distance)
            position_errors[name] = position_error
            velocity_errors[name] = normal_velocity
            position_losses.append(
                self._negative_log_plane_loss(
                    position_error,
                    position_tolerance,
                    barrier_limit,
                    overflow_weight,
                    max_loss,
                )
            )
            velocity_losses.append(
                self._negative_log_plane_loss(
                    normal_velocity,
                    velocity_tolerance,
                    barrier_limit,
                    overflow_weight,
                    max_loss,
                )
            )

        position_error = float(np.mean(list(position_errors.values())))
        velocity_error = float(np.mean(list(velocity_errors.values())))
        return {
            "position_loss": float(np.mean(position_losses)),
            "velocity_loss": float(np.mean(velocity_losses)),
            "position_error": position_error,
            "velocity_error": velocity_error,
            "combined_error": position_error + velocity_error,
            "wrist_position_error": position_errors["wrist"],
            "shaft_mid_position_error": position_errors["shaft_mid"],
            "clubhead_position_error": position_errors["clubhead"],
            "wrist_velocity_error": velocity_errors["wrist"],
            "shaft_mid_velocity_error": velocity_errors["shaft_mid"],
            "clubhead_velocity_error": velocity_errors["clubhead"],
        }

    def _target_line_terms(self, flight_cfg=None):
        flight_cfg = flight_cfg or self.reward_config["flight"]
        ball_pos = self._ball_pos()
        if not self.impact_happened and ball_pos[0] <= self.initial_ball_pos[0] + 0.01:
            return {
                "position_loss": 0.0,
                "velocity_loss": 0.0,
                "position_error": 0.0,
                "velocity_error": 0.0,
            }

        position_error = abs(float(ball_pos[1] - self.initial_ball_pos[1]))
        velocity_error = abs(float(self.ball_velocity[1]))
        position_loss = self._negative_log_plane_loss(
            position_error,
            flight_cfg["target_line_position_tolerance"],
            flight_cfg["target_line_barrier_limit"],
            flight_cfg["target_line_overflow_weight"],
            flight_cfg["target_line_max_loss"],
        )
        velocity_loss = self._negative_log_plane_loss(
            velocity_error,
            flight_cfg["target_line_velocity_tolerance"],
            flight_cfg["target_line_barrier_limit"],
            flight_cfg["target_line_overflow_weight"],
            flight_cfg["target_line_max_loss"],
        )
        return {
            "position_loss": position_loss,
            "velocity_loss": velocity_loss,
            "position_error": position_error,
            "velocity_error": velocity_error,
        }

    def _plane_phase_multiplier(self):
        return 1.0

    def _swing_plane_error(self):
        terms = self._club_plane_terms()
        return float(terms["combined_error"])

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

    def _impact_is_valid(self, clubhead_speed, forward_velocity, path_error, early_contact):
        return self._detect_ball_contact()

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
