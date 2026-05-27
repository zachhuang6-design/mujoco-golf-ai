import argparse
import time

import mujoco.viewer

from golf_rl.envs import TwoArmJointSwingGymEnv
from golf_rl.evaluate_policy import load_model


def main():
    parser = argparse.ArgumentParser(description="Visualize a saved SAC policy on the joint-accurate two-arm model.")
    parser.add_argument("model_path")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right", choices=("right", "left"))
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument("--speed", type=float, default=8.0)
    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    env = TwoArmJointSwingGymEnv(
        club_type=args.club,
        hand=args.hand,
        residual_scale=args.residual_scale,
    )
    obs, info = env.reset()
    printed_contact = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("ball_contact") and not printed_contact:
                printed_contact = True
                print("ball contact")
                print("clubhead_speed:", round(float(info.get("clubhead_speed", 0.0)), 4))
                print("clubhead_x_velocity:", round(float(info.get("clubhead_x_velocity", 0.0)), 4))
                print("impact_clubhead_speed:", round(float(info.get("impact_clubhead_speed", 0.0)), 4))
                print("impact_ball_x_velocity:", round(float(info.get("impact_ball_x_velocity", 0.0)), 4))
                print("impact_ball_lateral_velocity:", round(float(info.get("impact_ball_lateral_velocity", 0.0)), 4))
                print("max_clubhead_x_velocity:", round(float(info.get("max_clubhead_x_velocity", 0.0)), 4))
                print("ball_x_velocity:", round(float(info.get("ball_x_velocity", 0.0)), 4))
                print("max_ball_x_velocity:", round(float(info.get("max_ball_x_velocity", 0.0)), 4))
                print("max_ball_x_distance:", round(float(info.get("max_ball_x_distance", 0.0)), 4))
                print("max_ball_height:", round(float(info.get("max_ball_height", 0.0)), 4))
                print("ball_lateral_error:", round(float(info.get("ball_lateral_error", 0.0)), 4))
                print("target_line_factor:", round(float(info.get("target_line_factor", 0.0)), 4))
                print("min_club_ball_distance:", round(float(info.get("min_club_ball_distance", 0.0)), 4))
                print("reward:", round(float(reward), 4))
            viewer.sync()
            time.sleep(env.model.opt.timestep * args.speed)
            if terminated or truncated:
                print("episode complete")
                print("impact_happened:", bool(info.get("impact_happened")))
                print("max_ball_x_distance:", round(float(info.get("max_ball_x_distance", 0.0)), 4))
                print("max_ball_lateral_abs:", round(float(info.get("max_ball_lateral_abs", 0.0)), 4))
                print("target_line_factor:", round(float(info.get("target_line_factor", 0.0)), 4))
                obs, info = env.reset()
                printed_contact = False


if __name__ == "__main__":
    main()
