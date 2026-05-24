# Golf RL Pipeline

This folder replaces the retired hand-rolled RL experiment with a standard
Gymnasium + Stable-Baselines3 workflow.

## Install

```bash
../.venv/bin/python -m pip install -r golf_rl/requirements.txt
```

For DGX Spark / CUDA setup, see `golf_rl/DGX_SETUP.md` and run:

```bash
bash setup_dgx_venv.sh
```

## Quick Checks

```bash
../.venv/bin/python -m golf_rl.reward_debug --club 7iron --hand right --episodes 3
../.venv/bin/python -m golf_rl.train_sac --club 7iron --hand right --check-env --timesteps 10000
```

## Main Training

Recommended next run: residual SAC. This starts from the working CEM/PD swing
and learns torque corrections instead of trying to invent a full golf swing
from raw torque exploration.

```bash
../.venv/bin/python -m golf_rl.train_sac --env residual --club 7iron --hand right --timesteps 1000000 --model-dir artifacts/trained_models/residual_stage1 --log-dir artifacts/runs/sac_residual_stage1
```

Raw torque SAC is still available, but it is currently much harder and should
not be the main path until the residual learner is working:

```bash
../.venv/bin/python -m golf_rl.train_sac --club 7iron --hand right --timesteps 1000000
```

PPO comparison:

```bash
../.venv/bin/python -m golf_rl.train_ppo --club 7iron --hand right --timesteps 1000000
```

## Evaluate And Watch

```bash
../.venv/bin/python -m golf_rl.evaluate_policy artifacts/trained_models/residual_stage1/best_model.zip --env residual --algo sac
../.venv/bin/python -m golf_rl.visualize_policy artifacts/trained_models/residual_stage1/best_model.zip --env residual --algo sac
```

## Current Curriculum Stage

This is Stage B/C: real backswing before contact, then valid impact.

The reward intentionally requires:

- backswing arc, depth behind the ball, and height before impact can count
- no reward for tiny early taps
- valid impact speed and positive target-line clubhead velocity
- path error penalty at impact
- ground-contact penalty
- action and action-smoothness penalties
- swing-plane shaping

Face/path/center/attack-angle terms are present in the environment and reward
decomposition. Path error is now active; face, center, and attack angle can be
turned up after the policy reliably makes powerful valid contact.
