"""Gymnasium environment for residual learning on the current two-arm swing."""

import mujoco
import numpy as np

from golf_core.two_arm_joint_physics import (
    CONTROL_JOINTS,
    DEFAULT_IK_TRAJECTORY,
    FINISH_ROLLOUT_STEPS,
    SWING_STEPS,
    TORQUE_LIMITS,
    align_ball_to_address_clubface,
    interp_qpos_trajectory,
    joint_qpos_indices,
    joint_qvel_indices,
    load_qpos_trajectory,
    make_joint_model,
    prepare_control_trajectory,
    trajectory_pd_control,
)


class TwoArmJointSwingEnv:
    """Residual SAC environment around the current joint-accurate swing.

    The policy does not invent the swing from scratch. It adds small residual
    torques on top of the tuned two-arm baseline, so training can search for
    cleaner contact, more forward ball speed, and straighter direction.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        club_type="7iron",
        hand="right",
        residual_scale=0.08,
        max_steps=None,
        tracking_weight=0.06,
        action_weight=0.002,
        smoothness_weight=0.006,
        contact_reward=120.0,
        target_path=DEFAULT_IK_TRAJECTORY,
    ):
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TwoArmJointSwingEnv requires gymnasium. Install with: "
                "../.venv/bin/python -m pip install gymnasium"
            ) from exc

        self._gym = gym
        self.club_type = club_type
        self.hand = hand
        self.model = make_joint_model(club_type, hand)
        self.data = mujoco.MjData(self.model)
        self.qpos_indices = joint_qpos_indices(self.model)
        self.qvel_indices = joint_qvel_indices(self.model)
        self.torque_limits = TORQUE_LIMITS.astype(np.float32)
        self.residual_limits = self.torque_limits * float(residual_scale)
        self.max_steps = int(max_steps or (SWING_STEPS + FINISH_ROLLOUT_STEPS))
        self.tracking_weight = float(tracking_weight)
        self.action_weight = float(action_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.contact_reward_value = float(contact_reward)

        _, qpos_trajectory = load_qpos_trajectory(target_path)
        self.target_qpos = prepare_control_trajectory(qpos_trajectory)

        self.step_count = 0
        self.previous_action = np.zeros(len(CONTROL_JOINTS), dtype=np.float32)
        self.previous_ctrl = None
        self.ball_start_pos = np.zeros(3, dtype=np.float64)
        self.prev_club_ball_distance = 0.0
        self.max_clubhead_speed = 0.0
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_x_velocity = 0.0
        self.max_ball_height = 0.0
        self.max_ball_lateral_abs = 0.0
        self.min_club_ball_distance = np.inf
        self.impact_happened = False
        self.first_contact_step = None
        self.impact_clubhead_speed = 0.0
        self.impact_clubhead_x_velocity = 0.0
        self.impact_ball_x_velocity = 0.0
        self.impact_ball_lateral_velocity = 0.0
        self.impact_ball_lateral_error = 0.0
        self.last_reward_terms = {}

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(CONTROL_JOINTS),),
            dtype=np.float32,
        )
        self._reset_to_target_address()
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super_cls = getattr(super(), "reset", None)
        if callable(super_cls):
            try:
                super_cls(seed=seed)
            except TypeError:
                pass
        self._reset_to_target_address()
        self.step_count = 0
        self.previous_action[:] = 0.0
        self.previous_ctrl = None
        self.ball_start_pos = self._body_pos("ball")
        self.prev_club_ball_distance = self._club_ball_distance()
        self.max_clubhead_speed = 0.0
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_x_velocity = 0.0
        self.max_ball_height = 0.0
        self.max_ball_lateral_abs = 0.0
        self.min_club_ball_distance = np.inf
        self.impact_happened = False
        self.first_contact_step = None
        self.impact_clubhead_speed = 0.0
        self.impact_clubhead_x_velocity = 0.0
        self.impact_ball_x_velocity = 0.0
        self.impact_ball_lateral_velocity = 0.0
        self.impact_ball_lateral_error = 0.0
        self.last_reward_terms = {}
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        residual = action.astype(np.float64) * self.residual_limits.astype(np.float64)
        swing_active = self.step_count < SWING_STEPS
        progress = min(1.0, self.step_count / max(SWING_STEPS - 1, 1))
        active_residual = residual if swing_active else np.zeros_like(residual)
        baseline_target, ctrl = trajectory_pd_control(
            self.model,
            self.data,
            self.qpos_indices,
            self.qvel_indices,
            self.target_qpos,
            progress=progress,
            previous_ctrl=self.previous_ctrl,
            residual=active_residual,
            velocity_scale=1.0 if swing_active else 0.0,
            hold_target=not swing_active,
        )
        self.previous_ctrl = ctrl.copy()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        reward, terms = self._compute_reward(action, baseline_target)
        self.previous_action = action.copy()
        self.last_reward_terms = terms

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        info.update(terms)
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    def _get_obs(self):
        qpos = self.data.qpos[self.qpos_indices]
        qvel = self.data.qvel[self.qvel_indices]
        progress = min(1.0, self.step_count / max(SWING_STEPS - 1, 1))
        target = interp_qpos_trajectory(self.target_qpos, progress)
        clubhead_pos = self._site_pos("clubhead_site")
        clubhead_vel = self._site_vel("clubhead_site")
        grip_pos = self._site_pos("club_grip_site")
        mid_pos = self._site_pos("club_mid_site")
        ball_pos = self._body_pos("ball")
        ball_vel = self._body_vel("ball")
        obs = np.concatenate(
            [
                qpos,
                qvel,
                target,
                target - qpos,
                clubhead_pos,
                clubhead_vel,
                grip_pos,
                mid_pos,
                ball_pos,
                ball_vel,
                clubhead_pos - ball_pos,
                np.array([progress], dtype=np.float64),
                self.previous_action.astype(np.float64),
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action, baseline_target):
        clubhead_pos = self._site_pos("clubhead_site")
        clubhead_vel = self._site_vel("clubhead_site")
        ball_pos = self._body_pos("ball")
        ball_vel = self._body_vel("ball")
        ball_start_x = self.ball_start_pos[0]
        ball_start_z = self.ball_start_pos[2]

        clubhead_speed = float(np.linalg.norm(clubhead_vel))
        clubhead_x_velocity = float(clubhead_vel[0])
        ball_x_velocity = float(ball_vel[0])
        ball_x_distance = float(ball_pos[0] - ball_start_x)
        ball_height = float(ball_pos[2] - ball_start_z)
        club_ball_distance = self._club_ball_distance()

        previous_max_distance = self.max_ball_x_distance
        previous_max_height = self.max_ball_height
        previous_max_lateral = self.max_ball_lateral_abs
        lateral_error = float(ball_pos[1] - self.ball_start_pos[1])
        self.max_clubhead_speed = max(self.max_clubhead_speed, clubhead_speed)
        self.max_clubhead_x_velocity = max(self.max_clubhead_x_velocity, clubhead_x_velocity)
        self.max_ball_x_distance = max(self.max_ball_x_distance, ball_x_distance)
        self.max_ball_x_velocity = max(self.max_ball_x_velocity, ball_x_velocity)
        self.max_ball_height = max(self.max_ball_height, ball_height)
        self.max_ball_lateral_abs = max(self.max_ball_lateral_abs, abs(lateral_error))
        self.min_club_ball_distance = min(self.min_club_ball_distance, club_ball_distance)

        tracking_error = float(np.mean((baseline_target - self.data.qpos[self.qpos_indices]) ** 2))
        action_cost = float(np.mean(action * action))
        smoothness = float(np.mean((action - self.previous_action) ** 2))
        contact = self._detect_ball_contact()
        first_contact = bool(contact and not self.impact_happened)
        if first_contact:
            self.impact_happened = True
            self.first_contact_step = self.step_count
            self.impact_clubhead_speed = clubhead_speed
            self.impact_clubhead_x_velocity = clubhead_x_velocity
            self.impact_ball_x_velocity = ball_x_velocity
            self.impact_ball_lateral_velocity = float(ball_vel[1])
            self.impact_ball_lateral_error = lateral_error

        near_ball = float(np.exp(-((club_ball_distance / 0.08) ** 2)))
        approach_reward = 8.0 * max(0.0, self.prev_club_ball_distance - club_ball_distance)
        self.prev_club_ball_distance = club_ball_distance
        proximity_reward = 0.08 * near_ball
        speed_reward = 0.04 * near_ball * max(0.0, clubhead_x_velocity)
        peak_speed_reward = 0.0
        ball_velocity_reward = 12.0 * max(0.0, ball_x_velocity) if first_contact else 0.0
        distance_gain = max(0.0, self.max_ball_x_distance - previous_max_distance)
        lateral_ratio = self.max_ball_lateral_abs / max(self.max_ball_x_distance, 1.0)
        target_line_factor = float(np.exp(-((lateral_ratio / 0.18) ** 2)))
        distance_reward = 220.0 * distance_gain * target_line_factor
        height_reward = 22.0 * max(0.0, self.max_ball_height - previous_max_height)
        contact_reward = self.contact_reward_value if first_contact else 0.0
        impact_speed_reward = 8.0 * max(0.0, clubhead_x_velocity) if first_contact else 0.0
        forward_contact_reward = (
            90.0 * max(0.0, ball_x_velocity - 1.25 * abs(float(ball_vel[1])))
            if first_contact
            else 0.0
        )
        lateral_penalty = 900.0 * max(0.0, self.max_ball_lateral_abs - previous_max_lateral)
        lateral_velocity_penalty = 85.0 * abs(float(ball_vel[1])) if first_contact else 0.0
        backward_penalty = 120.0 * max(0.0, -ball_x_distance)
        miss_penalty = 260.0 if self.step_count >= self.max_steps - 1 and not self.impact_happened else 0.0
        tracking_penalty = self.tracking_weight * tracking_error
        action_penalty = self.action_weight * action_cost
        smoothness_penalty = self.smoothness_weight * smoothness
        ground_penalty = 8.0 if self._detect_ground_contact() else 0.0

        reward = (
            approach_reward
            + proximity_reward
            + speed_reward
            + peak_speed_reward
            + ball_velocity_reward
            + distance_reward
            + height_reward
            + contact_reward
            + impact_speed_reward
            + forward_contact_reward
            - lateral_penalty
            - lateral_velocity_penalty
            - backward_penalty
            - miss_penalty
            - tracking_penalty
            - action_penalty
            - smoothness_penalty
            - ground_penalty
        )
        return reward, {
            "clubhead_speed": clubhead_speed,
            "max_clubhead_speed": self.max_clubhead_speed,
            "clubhead_x_velocity": clubhead_x_velocity,
            "max_clubhead_x_velocity": self.max_clubhead_x_velocity,
            "ball_x_distance": ball_x_distance,
            "max_ball_x_distance": self.max_ball_x_distance,
            "ball_x_velocity": ball_x_velocity,
            "max_ball_x_velocity": self.max_ball_x_velocity,
            "ball_height": ball_height,
            "max_ball_height": self.max_ball_height,
            "ball_lateral_error": lateral_error,
            "max_ball_lateral_abs": self.max_ball_lateral_abs,
            "ball_lateral_velocity": float(ball_vel[1]),
            "ball_contact": float(contact),
            "first_contact": float(first_contact),
            "impact_happened": float(self.impact_happened),
            "first_contact_step": float(self.first_contact_step if self.first_contact_step is not None else -1),
            "impact_clubhead_speed": self.impact_clubhead_speed,
            "impact_clubhead_x_velocity": self.impact_clubhead_x_velocity,
            "impact_ball_x_velocity": self.impact_ball_x_velocity,
            "impact_ball_lateral_velocity": self.impact_ball_lateral_velocity,
            "impact_ball_lateral_error": self.impact_ball_lateral_error,
            "ball_contact_reward": contact_reward,
            "club_ball_distance": club_ball_distance,
            "min_club_ball_distance": float(self.min_club_ball_distance),
            "approach_reward": approach_reward,
            "proximity_reward": proximity_reward,
            "near_ball": near_ball,
            "target_line_factor": target_line_factor,
            "lateral_ratio": lateral_ratio,
            "speed_reward": speed_reward,
            "peak_speed_reward": peak_speed_reward,
            "ball_velocity_reward": ball_velocity_reward,
            "distance_reward": distance_reward,
            "height_reward": height_reward,
            "contact_reward": contact_reward,
            "impact_speed_reward": impact_speed_reward,
            "forward_contact_reward": forward_contact_reward,
            "lateral_penalty": lateral_penalty,
            "lateral_velocity_penalty": lateral_velocity_penalty,
            "backward_penalty": backward_penalty,
            "miss_penalty": miss_penalty,
            "tracking_penalty": tracking_penalty,
            "action_penalty": action_penalty,
            "smoothness_penalty": smoothness_penalty,
            "ground_penalty": ground_penalty,
            "tracking_error": tracking_error,
        }

    def _get_info(self):
        return {
            "step": self.step_count,
            "impact": self.impact_happened,
            "impact_happened": self.impact_happened,
            "club": self.club_type,
            "hand": self.hand,
        }

    def _reset_to_target_address(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.target_qpos[0]
        self.data.qvel[self.qvel_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)
        align_ball_to_address_clubface(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _club_ball_distance(self):
        return float(np.linalg.norm(self._site_pos("clubhead_site") - self._body_pos("ball")))

    def _site_pos(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[site_id].copy()

    def _body_pos(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body_id].copy()

    def _site_vel(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            vel,
            0,
        )
        return vel[3:].copy()

    def _body_vel(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel,
            0,
        )
        return vel[3:].copy()

    def _detect_ball_contact(self):
        return self._detect_contact_with("golf_ball", ("club_head_geom", "club_shaft_geom"))

    def _detect_ground_contact(self):
        return self._detect_contact_with("floor", ("club_head_geom", "club_shaft_geom"))

    def _detect_contact_with(self, geom_name, other_geom_names):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        other_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in other_geom_names
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if geom_id in pair and pair.intersection(other_ids):
                return True
        return False


try:
    import gymnasium as gym

    class TwoArmJointSwingGymEnv(TwoArmJointSwingEnv, gym.Env):
        pass

except ModuleNotFoundError:
    TwoArmJointSwingGymEnv = TwoArmJointSwingEnv
