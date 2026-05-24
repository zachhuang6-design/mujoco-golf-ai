import argparse
from pathlib import Path

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv
from golf_rl.train_sac import make_reward_logger


def make_env(env_name, club, hand):
    if env_name == "raw":
        return GolfSwingEnv(club_type=club, hand=hand)
    if env_name == "residual":
        return GolfResidualSwingEnv(club_type=club, hand=hand)
    raise ValueError(f"Unknown env '{env_name}'. Use raw or residual.")


def require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PPO training needs Stable-Baselines3 and Gymnasium. Install with: "
            "../.venv/bin/python -m pip install gymnasium stable-baselines3"
        ) from exc
    return PPO, BaseCallback, CheckpointCallback, EvalCallback, check_env, Monitor


def main():
    PPO, BaseCallback, CheckpointCallback, EvalCallback, check_env, Monitor = require_sb3()
    RewardTermCallback = make_reward_logger(BaseCallback)

    parser = argparse.ArgumentParser(description="Train PPO on the biomechanical golf swing environment.")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="raw")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="runs/ppo_golf")
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
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
            name_prefix=f"ppo_{args.env}_{args.club}_{args.hand}",
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
    model.save(str(Path(args.model_dir) / f"ppo_{args.env}_{args.club}_{args.hand}_final"))


if __name__ == "__main__":
    main()
