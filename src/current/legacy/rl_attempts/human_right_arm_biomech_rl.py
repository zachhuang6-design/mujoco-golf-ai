"""Compatibility launcher for the new Gymnasium/SB3 RL pipeline.

The previous hand-rolled policy-gradient experiment has been retired. The
current RL path is:

    golf_rl/envs/golf_swing_env.py
    golf_rl/train_sac.py
    golf_rl/train_ppo.py
    golf_rl/evaluate_policy.py
    golf_rl/visualize_policy.py

Run this file if you want the old filename to start SAC training.
"""

from golf_rl.train_sac import main


if __name__ == "__main__":
    main()

