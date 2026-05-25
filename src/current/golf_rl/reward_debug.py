import argparse

import numpy as np

from golf_rl.envs import GolfResidualSwingEnv, GolfSwingEnv


def make_env(env_name, club, hand):
    if env_name == "raw":
        return GolfSwingEnv(club_type=club, hand=hand)
    if env_name == "residual":
        return GolfResidualSwingEnv(club_type=club, hand=hand)
    raise ValueError(f"Unknown env '{env_name}'. Use raw or residual.")


def run_policy(env, policy_name, action_fn, episodes):
    totals = []
    term_sums = {}
    for episode in range(episodes):
        obs, info = env.reset(seed=episode)
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = action_fn(env, obs, steps)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
            for key, value in env.last_reward_terms.items():
                if isinstance(value, (int, float, np.floating, bool)):
                    term_sums[key] = term_sums.get(key, 0.0) + float(value)
        totals.append(total)
    print(policy_name)
    print("  episodes:", episodes)
    print("  avg_total_reward:", round(float(np.mean(totals)), 4))
    for key in sorted(term_sums):
        print(" ", key + ":", round(term_sums[key] / max(1, episodes), 4))


def main():
    parser = argparse.ArgumentParser(description="Inspect reward term magnitudes before training.")
    parser.add_argument("--club", default="7iron")
    parser.add_argument("--hand", default="right")
    parser.add_argument("--env", choices=("raw", "residual"), default="raw")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_env(args.env, args.club, args.hand)
    run_policy(
        env,
        "zero_action",
        lambda env, obs, step: np.zeros(env.action_space.shape, dtype=np.float32),
        args.episodes,
    )
    run_policy(
        env,
        "random_action",
        lambda env, obs, step: env.action_space.sample(),
        args.episodes,
    )


if __name__ == "__main__":
    main()
