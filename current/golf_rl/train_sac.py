import argparse
from pathlib import Path

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv


def make_env(env_name, club, hand):
    if env_name == "raw":
        return GolfSwingEnv(club_type=club, hand=hand)
    if env_name == "residual":
        return GolfResidualSwingEnv(club_type=club, hand=hand)
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
                        "forward_velocity",
                        "path_error",
                        "plane_error",
                        "distance_to_ball",
                        "backswing_arc",
                        "max_backswing_depth",
                        "max_backswing_height",
                        "valid_impact",
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
    parser.add_argument("--env", choices=("raw", "residual"), default="raw")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="runs/sac_golf")
    parser.add_argument("--model-dir", default="trained_models")
    parser.add_argument("--device", default="auto", help="PyTorch device for policy training: auto, cpu, cuda, or mps.")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(args.env, args.club, args.hand)
    if args.check_env:
        check_env(env, warn=True)
    env = Monitor(env)
    eval_env = Monitor(make_env(args.env, args.club, args.hand))

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
