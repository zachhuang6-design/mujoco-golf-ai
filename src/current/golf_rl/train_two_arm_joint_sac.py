import argparse
from pathlib import Path

from golf_rl.envs import TwoArmJointSwingGymEnv


def require_sb3():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Joint SAC training needs Stable-Baselines3 and Gymnasium. Install with: "
            "../.venv/bin/python -m pip install gymnasium stable-baselines3"
        ) from exc
    return SAC, BaseCallback, CheckpointCallback, EvalCallback, check_env, Monitor


def make_reward_logger(BaseCallback):
    class RewardTermCallback(BaseCallback):
        def _on_step(self):
            infos = self.locals.get("infos", [])
            if infos:
                for key, value in infos[0].items():
                    if key.endswith("_reward") or key.endswith("_penalty") or key in (
                        "clubhead_speed",
                        "max_clubhead_speed",
                        "clubhead_x_velocity",
                        "max_clubhead_x_velocity",
                        "ball_x_distance",
                        "max_ball_x_distance",
                        "ball_x_velocity",
                        "max_ball_x_velocity",
                        "ball_height",
                        "max_ball_height",
                        "ball_lateral_error",
                        "max_ball_lateral_abs",
                        "ball_lateral_velocity",
                        "club_ball_distance",
                        "min_club_ball_distance",
                        "tracking_error",
                        "near_ball",
                        "target_line_factor",
                        "lateral_ratio",
                        "ball_contact",
                        "first_contact",
                        "impact_happened",
                        "first_contact_step",
                        "impact_clubhead_speed",
                        "impact_clubhead_x_velocity",
                        "impact_ball_x_velocity",
                        "impact_ball_lateral_velocity",
                        "impact_ball_lateral_error",
                    ):
                        try:
                            self.logger.record(f"joint_reward_terms/{key}", float(value))
                        except (TypeError, ValueError):
                            pass
            return True

    return RewardTermCallback


def make_env(args):
    return TwoArmJointSwingGymEnv(
        club_type=args.club,
        hand=args.hand,
        residual_scale=args.residual_scale,
        tracking_weight=args.tracking_weight,
        action_weight=args.action_weight,
        smoothness_weight=args.smoothness_weight,
        contact_reward=args.contact_reward,
    )


def main():
    SAC, BaseCallback, CheckpointCallback, EvalCallback, check_env, Monitor = require_sb3()
    RewardTermCallback = make_reward_logger(BaseCallback)

    parser = argparse.ArgumentParser(description="Train SAC residual torques on the joint-accurate two-arm model.")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right", choices=("right", "left"))
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument("--tracking-weight", type=float, default=0.06)
    parser.add_argument("--action-weight", type=float, default=0.002)
    parser.add_argument("--smoothness-weight", type=float, default=0.006)
    parser.add_argument("--contact-reward", type=float, default=120.0)
    parser.add_argument("--log-dir", default="artifacts/runs/sac_two_arm_joint")
    parser.add_argument("--model-dir", default="artifacts/trained_models_two_arm_joint")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    if args.check_env:
        check_env(env, warn=True)
    env = Monitor(env)
    eval_env = Monitor(make_env(args))

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.985,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
    )
    callbacks = [
        RewardTermCallback(),
        CheckpointCallback(
            save_freq=25_000,
            save_path=args.model_dir,
            name_prefix=f"sac_two_arm_joint_{args.club}_{args.hand}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.model_dir,
            log_path=args.log_dir,
            eval_freq=25_000,
            deterministic=True,
        ),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(str(Path(args.model_dir) / f"sac_two_arm_joint_{args.club}_{args.hand}_final"))


if __name__ == "__main__":
    main()
