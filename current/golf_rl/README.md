# Golf RL Pipeline

This folder replaces the retired hand-rolled RL experiment with a standard
Gymnasium + Stable-Baselines3 workflow.

## Install

```bash
../.venv/bin/python -m pip install -r golf_rl/requirements.txt
```

## Quick Checks

```bash
../.venv/bin/python -m golf_rl.reward_debug --club 7iron --hand right --episodes 3
../.venv/bin/python -m golf_rl.train_sac --club 7iron --hand right --check-env --timesteps 10000
```

## Main Training

```bash
../.venv/bin/python -m golf_rl.train_sac --club 7iron --hand right --timesteps 1000000
```

PPO comparison:

```bash
../.venv/bin/python -m golf_rl.train_ppo --club 7iron --hand right --timesteps 1000000
```

## Evaluate And Watch

```bash
../.venv/bin/python -m golf_rl.evaluate_policy trained_models/sac_7iron_right_final.zip --algo sac
../.venv/bin/python -m golf_rl.visualize_policy trained_models/sac_7iron_right_final.zip --algo sac
```

## Current Curriculum Stage

This is Stage A: reliable contact.

The reward intentionally focuses on:

- progress toward the ball
- clubhead speed
- first ball contact bonus
- ground-contact penalty
- action and action-smoothness penalties
- light swing-plane shaping

Face/path/center/attack-angle terms are present in the environment and reward
decomposition, but their weights are currently zero in the club config files.
Turn them on after a policy can reliably make contact.

