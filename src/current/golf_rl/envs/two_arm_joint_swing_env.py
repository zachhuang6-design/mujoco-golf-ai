"""Gymnasium environment for the joint-accurate two-arm golf model."""

import math

import mujoco
import numpy as np

from golf_core.two_arm_joint_physics import (
    CONTROL_JOINTS,
    SWING_STEPS,
    TORQUE_LIMITS,
    baseline_pd_control,
    baseline_qpos,
    joint_qpos_indices,
    joint_qvel_indices,
    make_joint_model,
    reset_to_baseline,
)


class TwoArmJointSwingEnv:
    """Residual SAC environment around the joint-accurate baseline swing.

    The baseline PD controller creates the rough full swing. The policy action
    adds bounded residual torques, which is much easier to learn than asking RL
    to invent a legal two-arm swing from zero.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        club_type="7iron",
        hand="right",
        residual_scale=0.18,
        max_steps=SWING_STEPS + 160,
        tracking_weight=0.025,
        action_weight=0.002,
        smoothness_weight=0.004,
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
        self.max_steps = int(max_steps)
        self.tracking_weight = float(tracking_weight)
        self.action_weight = float(action_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.step_count = 0
        self.previous_action = np.zeros(len(CONTROL_JOINTS), dtype=np.float32)
        self.ball_start_pos = np.zeros(3, dtype=np.float64)
        self.prev_clubhead_x = 0.0
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_height = 0.0
        self.impact_happened = False
        self.last_reward_terms = {}

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(CONTROL_JOINTS),),
            dtype=np.float32,
        )
        reset_to_baseline(self.model, self.data)
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
        reset_to_baseline(self.model, self.data)
        self.step_count = 0
        self.previous_action[:] = 0.0
        self.ball_start_pos = self._body_pos("ball")
        self.prev_clubhead_x = float(self._site_pos("clubhead_site")[0])
        self.max_clubhead_x_velocity = 0.0
        self.max_ball_x_distance = 0.0
        self.max_ball_height = 0.0
        self.impact_happened = False
        self.last_reward_terms = {}
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        residual = action.astype(np.float64) * self.residual_limits.astype(np.float64)
        baseline_target, _ = baseline_pd_control(
            self.model,
            self.data,
            self.qpos_indices,
            self.qvel_indices,
            self.step_count,
            residual=residual,
        )
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        reward, terms = self._compute_reward(action, baseline_target)
        self.previous_action = action.copy()
        self.last_reward_terms = terms

        terminated = bool(self.impact_happened and self.step_count > SWING_STEPS + 50)
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
        target = baseline_qpos(progress)
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

        clubhead_x_velocity = float(clubhead_vel[0])
        self.max_clubhead_x_velocity = max(self.max_clubhead_x_velocity, clubhead_x_velocity)
        self.max_ball_x_distance = max(self.max_ball_x_distance, float(ball_pos[0] - ball_start_x))
        self.max_ball_height = max(self.max_ball_height, float(ball_pos[2] - ball_start_z))

        tracking_error = float(np.mean((baseline_target - self.data.qpos[self.qpos_indices]) ** 2))
        action_cost = float(np.mean(action * action))
        smoothness = float(np.mean((action - self.previous_action) ** 2))
        contact = self._detect_ball_contact()
        if contact:
            self.impact_happened = True

        speed_reward = 0.12 * max(0.0, clubhead_x_velocity)
        peak_speed_reward = 0.05 * max(0.0, self.max_clubhead_x_velocity)
        distance_reward = 80.0 * max(0.0, self.max_ball_x_distance)
        height_reward = 18.0 * max(0.0, self.max_ball_height)
        contact_reward = 25.0 if contact else 0.0
        lateral_penalty = 10.0 * abs(float(ball_pos[1] - self.ball_start_pos[1]))
        tracking_penalty = self.tracking_weight * tracking_error
        action_penalty = self.action_weight * action_cost
        smoothness_penalty = self.smoothness_weight * smoothness
        ground_penalty = 8.0 if self._detect_ground_contact() else 0.0

        reward = (
            speed_reward
            + peak_speed_reward
            + distance_reward
            + height_reward
            + contact_reward
            - lateral_penalty
            - tracking_penalty
            - action_penalty
            - smoothness_penalty
            - ground_penalty
        )
        return reward, {
            "clubhead_x_velocity": clubhead_x_velocity,
            "max_clubhead_x_velocity": self.max_clubhead_x_velocity,
            "ball_x_distance": float(ball_pos[0] - ball_start_x),
            "max_ball_x_distance": self.max_ball_x_distance,
            "ball_height": float(ball_pos[2] - ball_start_z),
            "max_ball_height": self.max_ball_height,
            "ball_lateral_error": float(ball_pos[1] - self.ball_start_pos[1]),
            "ball_lateral_velocity": float(ball_vel[1]),
            "ball_contact": float(contact),
            "speed_reward": speed_reward,
            "peak_speed_reward": peak_speed_reward,
            "distance_reward": distance_reward,
            "height_reward": height_reward,
            "contact_reward": contact_reward,
            "lateral_penalty": lateral_penalty,
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
            "club": self.club_type,
            "hand": self.hand,
        }

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
