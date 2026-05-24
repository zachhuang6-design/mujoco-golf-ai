import argparse
from pathlib import Path

import numpy as np

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


def resolve_model_path(path):
    candidate = Path(path)
    if candidate.is_dir():
        for name in ("best_model.zip", "best_model", "model.zip", "model"):
            nested = candidate / name
            if nested.exists() and not nested.is_dir():
                return str(nested)
    if candidate.exists() and not candidate.is_dir():
        return str(candidate)
    if candidate.suffix == ".zip":
        without_zip = candidate.with_suffix("")
        if without_zip.exists():
            return str(without_zip)
    else:
        with_zip = candidate.with_suffix(candidate.suffix + ".zip") if candidate.suffix else Path(str(candidate) + ".zip")
        if with_zip.exists():
            return str(with_zip)
    raise FileNotFoundError(
        f"Could not find model '{path}'. Tried '{candidate}'"
        + (f" and '{with_zip}'." if "with_zip" in locals() else ".")
    )


def load_model(algo, path):
    try:
        from stable_baselines3 import PPO, SAC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy evaluation needs Stable-Baselines3. Install with: "
            "../.venv/bin/python -m pip install gymnasium stable-baselines3"
        ) from exc
    resolved_path = resolve_model_path(path)
    if algo == "sac":
        return SAC.load(resolved_path)
    if algo == "ppo":
        return PPO.load(resolved_path)
    raise ValueError(f"Unknown algo '{algo}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved golf swing policy.")
    parser.add_argument("model_path")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="residual")
    parser.add_argument("--residual-scale", type=float, default=0.12)
    parser.add_argument("--target-tracking-weight", type=float, default=0.015)
    parser.add_argument("--early-turn-reward-weight", type=float, default=1.5)
    parser.add_argument("--early-lift-penalty-weight", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    env = make_env(
        args.env,
        args.club,
        args.hand,
        args.residual_scale,
        args.target_tracking_weight,
        args.early_turn_reward_weight,
        args.early_lift_penalty_weight,
    )
    rewards = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        done = False
        total = 0.0
        final_info = {}
        impact_info = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            final_info = info
            if info.get("ball_contact") and impact_info is None:
                impact_info = dict(info)
            done = terminated or truncated
        rewards.append(total)
        report_info = impact_info or final_info
        print(
            "episode",
            episode + 1,
            "reward",
            round(total, 4),
            "impact",
            final_info.get("impact_happened"),
            "valid",
            report_info.get("valid_impact"),
        )
        for key in (
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
            "clubhead_plane_distance",
            "shaft_mid_plane_distance",
            "wrist_plane_distance",
            "clubhead_plane_velocity",
            "shaft_mid_plane_velocity",
            "wrist_plane_velocity",
            "ball_lateral_error",
            "ball_lateral_velocity",
            "distance_to_ball",
            "clubhead_x_velocity_reward",
            "ball_distance_reward",
            "ball_height_reward",
            "plane_position_improvement_reward",
            "plane_velocity_improvement_reward",
            "plane_position_penalty",
            "plane_velocity_penalty",
            "target_line_position_penalty",
            "target_line_velocity_penalty",
            "early_shoulder_turn_reward",
            "early_shoulder_lift_penalty",
            "early_shoulder_turn_progress",
            "early_shoulder_lift_error_deg",
            "target_tracking_penalty",
            "residual_action_penalty",
            "residual_norm",
        ):
            if key in report_info:
                value = report_info[key]
                if isinstance(value, (bool, np.bool_)):
                    print(" ", key, bool(value))
                else:
                    print(" ", key, round(float(value), 4))
    print("mean_reward", round(float(np.mean(rewards)), 4))


if __name__ == "__main__":
    main()
