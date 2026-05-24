import numpy as np

from golf_core.right_arm_cem import apply_pd_controls, target_angles
from golf_core.right_arm_swing import BIOMECH_SWING_CANDIDATE

from golf_rl.envs.golf_swing_env import GolfSwingEnv


class GolfResidualSwingEnv(GolfSwingEnv):
    """SAC environment that learns torque corrections on top of the CEM/PD swing.

    The raw torque environment asks RL to discover the whole swing from address,
    which has proven too sparse. This environment starts with the working CEM
    swing controller and lets the policy add bounded residual torques.
    """

    def __init__(
        self,
        *args,
        candidate=None,
        residual_scale=0.12,
        target_tracking_weight=0.015,
        residual_action_weight=0.02,
        early_turn_reward_weight=1.5,
        early_lift_penalty_weight=2.0,
        early_takeaway_fraction=0.38,
        early_turn_target_fraction=0.55,
        early_lift_tolerance_deg=6.0,
        **kwargs,
    ):
        self.candidate = dict(candidate or BIOMECH_SWING_CANDIDATE)
        self.residual_scale = float(residual_scale)
        self.target_tracking_weight = float(target_tracking_weight)
        self.residual_action_weight = float(residual_action_weight)
        self.early_turn_reward_weight = float(early_turn_reward_weight)
        self.early_lift_penalty_weight = float(early_lift_penalty_weight)
        self.early_takeaway_fraction = float(early_takeaway_fraction)
        self.early_turn_target_fraction = float(early_turn_target_fraction)
        self.early_lift_tolerance = np.deg2rad(float(early_lift_tolerance_deg))
        super().__init__(*args, **kwargs)
        self.residual_limits = self.torque_limits * self.residual_scale
        self.observation_space = self.observation_space.__class__(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_obs().shape[0],),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.candidate["hand"] = self.hand
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        apply_pd_controls(self.data, self.candidate, self.step_count)
        base_ctrl = self.data.ctrl.copy()
        residual_ctrl = action * self.residual_limits
        self.data.ctrl[:] = np.clip(base_ctrl + residual_ctrl, -self.torque_limits, self.torque_limits)

        prev_action = self.previous_action.copy()
        import mujoco

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        self._update_velocity_cache()
        self._update_backswing_progress()
        reward, reward_terms = self._compute_reward(action, prev_action)
        tracking_penalty = 0.0
        if self.target_tracking_weight > 0.0:
            tracking_penalty = self.target_tracking_weight * self._target_tracking_error()
            reward -= tracking_penalty
        residual_action_penalty = self.residual_action_weight * float(np.mean(action * action))
        reward -= residual_action_penalty
        early_terms = self._early_takeaway_terms()
        reward += early_terms["early_shoulder_turn_reward"]
        reward -= early_terms["early_shoulder_lift_penalty"]
        reward_terms["target_tracking_penalty"] = tracking_penalty
        reward_terms["residual_action_penalty"] = residual_action_penalty
        reward_terms.update(early_terms)
        reward_terms["residual_norm"] = float(np.linalg.norm(action))
        self.previous_action = action.copy()

        terminated = self._is_terminal()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        info.update(reward_terms)
        self.last_reward_terms = reward_terms
        self._store_previous_positions()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        target = np.asarray(target_angles(self.candidate, self.step_count), dtype=np.float64)
        qpos = self.data.qpos[self.joint_qpos_indices]
        tracking_error = target - qpos
        return np.concatenate([base_obs, target, tracking_error]).astype(np.float32)

    def _target_tracking_error(self):
        target = np.asarray(target_angles(self.candidate, self.step_count), dtype=np.float64)
        qpos = self.data.qpos[self.joint_qpos_indices]
        return float(np.mean((target - qpos) ** 2))

    def _early_takeaway_terms(self):
        top_step = int(self.candidate["top_step"])
        window_end = max(1, int(top_step * self.early_takeaway_fraction))
        address = np.asarray(self.candidate.get("address_pose"), dtype=np.float64)
        top = np.asarray(self.candidate["top_pose"], dtype=np.float64)
        qpos = self.data.qpos[self.joint_qpos_indices]

        if self.step_count > window_end:
            return {
                "early_shoulder_turn_reward": 0.0,
                "early_shoulder_lift_penalty": 0.0,
                "early_shoulder_turn_progress": 1.0,
                "early_shoulder_lift_error_deg": 0.0,
            }

        turn_delta = float(top[0] - address[0])
        turn_direction = 1.0 if turn_delta >= 0.0 else -1.0
        target_turn = max(abs(turn_delta) * self.early_turn_target_fraction, 1e-6)
        current_turn = max(0.0, float((qpos[0] - address[0]) * turn_direction))
        turn_progress = min(1.0, current_turn / target_turn)
        early_shoulder_turn_reward = self.early_turn_reward_weight * turn_progress

        lift_error = abs(float(qpos[1] - address[1]))
        normalized_lift_error = max(0.0, lift_error / max(self.early_lift_tolerance, 1e-9) - 1.0)
        early_shoulder_lift_penalty = self.early_lift_penalty_weight * normalized_lift_error * normalized_lift_error
        return {
            "early_shoulder_turn_reward": early_shoulder_turn_reward,
            "early_shoulder_lift_penalty": early_shoulder_lift_penalty,
            "early_shoulder_turn_progress": turn_progress,
            "early_shoulder_lift_error_deg": float(np.rad2deg(lift_error)),
        }

    def _plane_phase_multiplier(self):
        plane_cfg = self.reward_config["plane"]
        top_step = int(self.candidate["top_step"])
        down_start_step = int(self.candidate.get("down_start_step", top_step + self.candidate.get("top_hold", 0)))
        impact_step = int(self.candidate["impact_step"])
        takeaway_end = max(1, int(top_step * 0.35))

        if self.step_count < takeaway_end:
            return float(plane_cfg["takeaway_multiplier"])
        if self.step_count < down_start_step:
            return float(plane_cfg["backswing_multiplier"])
        if self.step_count < impact_step:
            return float(plane_cfg["downswing_multiplier"])
        return float(plane_cfg["followthrough_multiplier"])
