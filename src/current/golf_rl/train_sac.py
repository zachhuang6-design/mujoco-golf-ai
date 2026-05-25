import argparse
from pathlib import Path

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv


def make_env(
    env_name,
    club,
    hand,
    residual_scale=0.12,
    target_tracking_weight=0.015,
    early_turn_reward_weight=1.5,
    early_lift_penalty_weight=2.0,
):
    if env_name == "raw":
        return GolfSwingEnv(club_type=club, hand=hand)
    if env_name == "residual":
        return GolfResidualSwingEnv(
            club_type=club,
            hand=hand,
            residual_scale=residual_scale,
            target_tracking_weight=target_tracking_weight,
            early_turn_reward_weight=early_turn_reward_weight,
            early_lift_penalty_weight=early_lift_penalty_weight,
        )
    raise ValueError(f"Unknown env '{env_name}'. Use raw or residual.")


def require_sb3():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAC training needs Stable-Baselines3 and Gymnasium. Install with: "
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
                        "clubhead_x_velocity",
                        "max_clubhead_x_velocity",
                        "ball_x_distance",
                        "max_ball_x_distance",
                        "ball_height",
                        "max_ball_height",
                        "plane_error",
                        "plane_position_error",
                        "plane_velocity_error",
                        "plane_phase_multiplier",
                        "ball_lateral_error",
                        "ball_lateral_velocity",
                        "early_shoulder_turn_progress",
                        "early_shoulder_lift_error_deg",
                        "distance_to_ball",
                        "valid_impact",
                        "residual_norm",
                    ):
                        try:
                            self.logger.record(f"reward_terms/{key}", float(value))
                        except (TypeError, ValueError):
                            pass
            return True

    return RewardTermCallback


def main():
    SAC, BaseCallback, CheckpointCallback, EvalCallback, check_env, Monitor = require_sb3()
    RewardTermCallback = make_reward_logger(BaseCallback)

    parser = argparse.ArgumentParser(description="Train SAC on the biomechanical golf swing environment.")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="residual")
    parser.add_argument("--residual-scale", type=float, default=0.12)
    parser.add_argument("--target-tracking-weight", type=float, default=0.015)
    parser.add_argument("--early-turn-reward-weight", type=float, default=1.5)
    parser.add_argument("--early-lift-penalty-weight", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="artifacts/runs/sac_golf")
    parser.add_argument("--model-dir", default="artifacts/trained_models")
    parser.add_argument("--device", default="auto", help="PyTorch device for policy training: auto, cpu, cuda, or mps.")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(
        args.env,
        args.club,
        args.hand,
        args.residual_scale,
        args.target_tracking_weight,
        args.early_turn_reward_weight,
        args.early_lift_penalty_weight,
    )
    if args.check_env:
        check_env(env, warn=True)
    env = Monitor(env)
    eval_env = Monitor(
        make_env(
            args.env,
            args.club,
            args.hand,
            args.residual_scale,
            args.target_tracking_weight,
            args.early_turn_reward_weight,
            args.early_lift_penalty_weight,
        )
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
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
            name_prefix=f"sac_{args.env}_{args.club}_{args.hand}",
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
    model.save(str(Path(args.model_dir) / f"sac_{args.env}_{args.club}_{args.hand}_final"))


if __name__ == "__main__":
    main()
