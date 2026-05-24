import argparse
import time

import mujoco.viewer

from golf_rl.envs import GolfSwingEnv
from golf_rl.evaluate_policy import load_model


def main():
    parser = argparse.ArgumentParser(description="Run a saved policy in the MuJoCo viewer.")
    parser.add_argument("model_path")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--speed", type=float, default=12.0)
    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    env = GolfSwingEnv(club_type=args.club, hand=args.hand)
    obs, info = env.reset()
    printed_impact = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("ball_contact") and not printed_impact:
                printed_impact = True
                print("ball contact")
                print("clubhead_speed:", round(float(info.get("clubhead_speed", 0.0)), 4))
                print("plane_error:", round(float(info.get("plane_error", 0.0)), 4))
                print("reward:", round(float(reward), 4))
            viewer.sync()
            time.sleep(env.model.opt.timestep * args.speed)
            if terminated or truncated:
                obs, info = env.reset()
                printed_impact = False


if __name__ == "__main__":
    main()
