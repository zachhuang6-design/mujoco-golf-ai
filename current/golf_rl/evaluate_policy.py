import argparse
from pathlib import Path

import numpy as np

from golf_rl.envs import GolfSwingEnv


def resolve_model_path(path):
    candidate = Path(path)
    if candidate.exists():
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
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model = load_model(args.algo, args.model_path)
    env = GolfSwingEnv(club_type=args.club, hand=args.hand)
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
            "forward_velocity",
            "path_error",
            "plane_error",
            "distance_to_ball",
            "backswing_completed",
            "max_backswing_depth",
            "max_backswing_height",
            "backswing_arc",
            "ball_contact_reward",
            "valid_impact_bonus",
            "weak_contact_penalty",
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
