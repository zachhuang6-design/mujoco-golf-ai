import argparse
import time

import mujoco.viewer

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv
from golf_rl.evaluate_policy import load_model


def make_env(env_name, club, hand):
    if env_name == "raw":
        return GolfSwingEnv(club_type=club, hand=hand)
    if env_name == "residual":
        return GolfResidualSwingEnv(club_type=club, hand=hand)
    raise ValueError(f"Unknown env '{env_name}'. Use raw or residual.")


def main():
    parser = argparse.ArgumentParser(description="Run a saved policy in the MuJoCo viewer.")
    parser.add_argument("model_path")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="raw")
    parser.add_argument("--speed", type=float, default=12.0)
    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    env = make_env(args.env, args.club, args.hand)
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
                print("forward_velocity:", round(float(info.get("forward_velocity", 0.0)), 4))
                print("path_error:", round(float(info.get("path_error", 0.0)), 4))
                print("plane_error:", round(float(info.get("plane_error", 0.0)), 4))
                print("backswing_completed:", bool(info.get("backswing_completed", False)))
                print("backswing_arc:", round(float(info.get("backswing_arc", 0.0)), 4))
                print("max_backswing_depth:", round(float(info.get("max_backswing_depth", 0.0)), 4))
                print("max_backswing_height:", round(float(info.get("max_backswing_height", 0.0)), 4))
                print("reward:", round(float(reward), 4))
            viewer.sync()
            time.sleep(env.model.opt.timestep * args.speed)
            if terminated or truncated:
                obs, info = env.reset()
                printed_impact = False


if __name__ == "__main__":
    main()
