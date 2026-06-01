import argparse
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np

from golf_rl.envs import TwoArmJointSwingGymEnv

REWARD_VERSION = "post_impact_summary_v3"


LOGGED_INFO_KEYS = (
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
    "post_impact_peak_ball_x_velocity",
    "post_impact_peak_ball_height",
    "post_impact_max_lateral_velocity",
    "post_impact_max_lateral_error",
    "post_impact_target_line_factor",
    "post_impact_window_active",
    "post_impact_done",
    "no_contact_timeout",
)


def should_log_metric(key):
    return key.endswith("_reward") or key.endswith("_penalty") or key in LOGGED_INFO_KEYS


def require_sb3():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Joint SAC training needs Stable-Baselines3 and Gymnasium. Install with: "
            "../.venv/bin/python -m pip install gymnasium stable-baselines3"
        ) from exc
    return SAC, BaseCallback, CheckpointCallback, EvalCallback, check_env, evaluate_policy, Monitor


def make_reward_logger(BaseCallback):
    class RewardTermCallback(BaseCallback):
        def _on_step(self):
            infos = self.locals.get("infos", [])
            if infos:
                for key, value in infos[0].items():
                    if should_log_metric(key):
                        try:
                            self.logger.record(f"joint_reward_terms/{key}", float(value))
                        except (TypeError, ValueError):
                            pass
            return True

    return RewardTermCallback


def make_eval_info_logger(BaseCallback, eval_env, eval_freq):
    class EvalInfoCallback(BaseCallback):
        def _on_step(self):
            if self.n_calls % eval_freq != 0:
                return True

            obs, info = eval_env.reset()
            total_reward = 0.0
            episode_length = 0
            terminal_info = info
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += float(reward)
                episode_length += 1
                terminal_info = info
                if terminated or truncated:
                    break

            self.logger.record("eval_reward_terms/manual_episode_reward", total_reward)
            self.logger.record("eval_reward_terms/manual_episode_length", episode_length)
            for key, value in terminal_info.items():
                if should_log_metric(key):
                    try:
                        self.logger.record(f"eval_reward_terms/{key}", float(value))
                    except (TypeError, ValueError):
                        pass
            return True

    return EvalInfoCallback


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


def parse_ent_coef(value):
    try:
        return float(value)
    except ValueError:
        return value


def resolve_model_path(path):
    path = Path(path)
    if path.exists():
        return path
    if path.suffix != ".zip":
        zip_path = Path(f"{path}.zip")
        if zip_path.exists():
            return zip_path
    return None


def read_best_score(model_root):
    version_path = Path(model_root) / "best_score_version.txt"
    if not version_path.exists() or version_path.read_text().strip() != REWARD_VERSION:
        return None
    score_path = Path(model_root) / "best_score.txt"
    if not score_path.exists():
        return None
    try:
        return float(score_path.read_text().strip())
    except ValueError:
        return None


def write_best_score(model_root, score):
    model_root = Path(model_root)
    (model_root / "best_score.txt").write_text(f"{float(score):.8f}\n")
    (model_root / "best_score_version.txt").write_text(f"{REWARD_VERSION}\n")


def read_run_best_score(run_log_dir):
    evaluations_path = Path(run_log_dir) / "evaluations.npz"
    if not evaluations_path.exists():
        return None
    data = np.load(evaluations_path)
    if "results" not in data:
        return None
    results = np.asarray(data["results"], dtype=np.float64)
    if results.size == 0:
        return None
    means = results.mean(axis=1)
    return float(np.max(means))


def main():
    SAC, BaseCallback, CheckpointCallback, EvalCallback, check_env, evaluate_policy, Monitor = require_sb3()
    RewardTermCallback = make_reward_logger(BaseCallback)

    parser = argparse.ArgumentParser(description="Train SAC residual torques on the joint-accurate two-arm model.")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right", choices=("right", "left"))
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", default="0.03")
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument("--tracking-weight", type=float, default=0.06)
    parser.add_argument("--action-weight", type=float, default=0.002)
    parser.add_argument("--smoothness-weight", type=float, default=0.006)
    parser.add_argument("--contact-reward", type=float, default=120.0)
    parser.add_argument("--log-dir", default="artifacts/runs/sac_two_arm_joint")
    parser.add_argument("--model-dir", default="artifacts/trained_models_two_arm_joint")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", default="auto")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.model_dir)
    log_root = Path(args.log_dir)
    run_name = args.run_name or datetime.now().strftime(f"{args.club}_{args.hand}_%Y%m%d_%H%M%S")
    run_model_dir = model_root / run_name
    run_log_dir = log_root / run_name

    model_root.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    if args.check_env:
        check_env(env, warn=True)
    env = Monitor(env)
    eval_env = Monitor(make_env(args))
    eval_info_env = Monitor(make_env(args))
    ent_coef = parse_ent_coef(args.ent_coef)

    resume_path = None
    if not args.fresh:
        if args.resume_from == "auto":
            resume_path = resolve_model_path(model_root / "best_model")
        else:
            resume_path = resolve_model_path(args.resume_from)

    if resume_path is not None:
        print(f"Resuming SAC policy from: {resume_path}")
        if args.ent_coef != "checkpoint":
            print(
                "Keeping the entropy setting stored in the checkpoint while resuming. "
                "Use --fresh to start a new fixed-entropy policy."
            )
        model = SAC.load(
            resume_path,
            env=env,
            device=args.device,
            tensorboard_log=str(run_log_dir),
            learning_rate=args.learning_rate,
        )
        previous_best_score = read_best_score(model_root)
        if previous_best_score is None:
            print("Scoring the previous best once so this run must beat it before promotion.")
            previous_best_score, _ = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=1,
                deterministic=True,
                warn=False,
            )
            write_best_score(model_root, previous_best_score)
        print(f"Previous best score: {previous_best_score:.4f}")
    else:
        previous_best_score = None
        if args.fresh:
            print("Starting fresh SAC policy because --fresh was provided.")
        elif args.resume_from == "auto":
            print(f"No previous best model found at {model_root / 'best_model'}; starting fresh.")
        else:
            print(f"Could not find resume model '{args.resume_from}'; starting fresh.")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.985,
            train_freq=1,
            gradient_steps=1,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=str(run_log_dir),
            seed=args.seed,
            device=args.device,
        )

    print(f"Saving this run to: {run_model_dir}")
    print(f"Logging this run to: {run_log_dir}")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir),
        log_path=str(run_log_dir),
        eval_freq=25_000,
        deterministic=True,
    )
    if previous_best_score is not None:
        eval_callback.best_mean_reward = previous_best_score
    EvalInfoCallback = make_eval_info_logger(BaseCallback, eval_info_env, eval_freq=25_000)

    callbacks = [
        RewardTermCallback(),
        CheckpointCallback(
            save_freq=25_000,
            save_path=str(run_model_dir),
            name_prefix=f"sac_two_arm_joint_{args.club}_{args.hand}",
        ),
        eval_callback,
        EvalInfoCallback(),
    ]
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_path is None,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving the current policy before exiting.")

    final_path = run_model_dir / f"sac_two_arm_joint_{args.club}_{args.hand}_final"
    model.save(str(final_path))

    run_best_score = read_run_best_score(run_log_dir)
    run_best = resolve_model_path(run_model_dir / "best_model")
    source_model = run_best
    should_promote = source_model is not None and (
        previous_best_score is None
        or (run_best_score is not None and run_best_score > previous_best_score)
    )
    if should_promote:
        latest_best = model_root / "best_model.zip"
        shutil.copy2(source_model, latest_best)
        if run_best_score is not None:
            write_best_score(model_root, run_best_score)
        print(f"Updated next-run resume model: {latest_best}")
        if run_best_score is not None:
            print(f"New best score: {run_best_score:.4f}")
    elif previous_best_score is None:
        source_model = resolve_model_path(final_path)
        if source_model is not None:
            latest_best = model_root / "best_model.zip"
            shutil.copy2(source_model, latest_best)
            print(f"Initialized next-run resume model from final policy: {latest_best}")
        else:
            print("No best/final model file found to promote for the next run.")
    else:
        print("This run did not beat the previous best; keeping the existing resume model.")
        if run_best_score is not None:
            print(f"Run best score: {run_best_score:.4f}")
        print(f"Previous best score: {previous_best_score:.4f}")


if __name__ == "__main__":
    main()
