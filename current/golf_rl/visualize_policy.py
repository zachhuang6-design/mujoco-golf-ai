import argparse
import time

import mujoco.viewer

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv
from golf_rl.evaluate_policy import load_model


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


def main():
    parser = argparse.ArgumentParser(description="Run a saved policy in the MuJoCo viewer.")
    parser.add_argument("model_path")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="residual")
    parser.add_argument("--residual-scale", type=float, default=0.12)
    parser.add_argument("--target-tracking-weight", type=float, default=0.015)
    parser.add_argument("--early-turn-reward-weight", type=float, default=1.5)
    parser.add_argument("--early-lift-penalty-weight", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=12.0)
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
    obs, info = env.reset()
    printed_impact = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("ball_contact") and not printed_impact:
                printed_impact = True
                print("ball contact")
                print("valid_impact:", bool(info.get("valid_impact", False)))
                print("clubhead_speed:", round(float(info.get("clubhead_speed", 0.0)), 4))
                print("clubhead_x_velocity:", round(float(info.get("clubhead_x_velocity", 0.0)), 4))
                print("max_clubhead_x_velocity:", round(float(info.get("max_clubhead_x_velocity", 0.0)), 4))
                print("max_ball_x_distance:", round(float(info.get("max_ball_x_distance", 0.0)), 4))
                print("max_ball_height:", round(float(info.get("max_ball_height", 0.0)), 4))
                print("plane_error:", round(float(info.get("plane_error", 0.0)), 4))
                print("plane_position_error:", round(float(info.get("plane_position_error", 0.0)), 4))
                print("plane_velocity_error:", round(float(info.get("plane_velocity_error", 0.0)), 4))
                print("ball_lateral_error:", round(float(info.get("ball_lateral_error", 0.0)), 4))
                print("ball_lateral_velocity:", round(float(info.get("ball_lateral_velocity", 0.0)), 4))
                print("early_shoulder_turn_progress:", round(float(info.get("early_shoulder_turn_progress", 0.0)), 4))
                print("early_shoulder_lift_error_deg:", round(float(info.get("early_shoulder_lift_error_deg", 0.0)), 4))
                print("reward:", round(float(reward), 4))
            viewer.sync()
            time.sleep(env.model.opt.timestep * args.speed)
            if terminated or truncated:
                obs, info = env.reset()
                printed_impact = False


if __name__ == "__main__":
    main()
