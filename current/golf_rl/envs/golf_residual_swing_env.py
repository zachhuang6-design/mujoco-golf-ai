import numpy as np

from human_right_arm_biomech_cem import apply_pd_controls, target_angles
from human_right_arm_biomech_swing import BIOMECH_SWING_CANDIDATE

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
        residual_scale=0.35,
        target_tracking_weight=0.03,
        **kwargs,
    ):
        self.candidate = dict(candidate or BIOMECH_SWING_CANDIDATE)
        self.residual_scale = float(residual_scale)
        self.target_tracking_weight = float(target_tracking_weight)
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
        tracking_penalty = self.target_tracking_weight * self._target_tracking_error()
        reward -= tracking_penalty
        reward_terms["target_tracking_penalty"] = tracking_penalty
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
