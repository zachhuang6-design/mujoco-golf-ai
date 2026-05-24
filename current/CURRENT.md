# Current Working Project Map

This folder has been reorganized around the current working path:

1. `golf_core/`
   - Current biomechanical right-arm MuJoCo model.
   - Current CEM/PD swing controller.
   - Current swing viewer and club presets.

2. `golf_rl/`
   - Gymnasium + Stable-Baselines3 reinforcement-learning tools.
   - Current recommendation: use `--env residual`, which starts from the CEM/PD swing and lets RL learn corrections.

3. `legacy/`
   - Older 2D three-joint scripts and swing-plane bridge experiments.
   - Preserved for reference, not the current path.

4. `artifacts/`
   - Training runs and saved models.
   - Generated outputs live here instead of cluttering the source root.

## Current Viewer

```bash
mjpython current_swing.py --club 7iron --hand right
```

## Current Residual RL Training

```bash
../.venv/bin/python -m golf_rl.train_sac \
  --env residual \
  --club 7iron \
  --hand right \
  --timesteps 100000 \
  --device cpu \
  --model-dir artifacts/trained_models/residual_stage1_local \
  --log-dir artifacts/runs/sac_residual_stage1_local
```

## Evaluate

```bash
../.venv/bin/python -m golf_rl.evaluate_policy \
  artifacts/trained_models/residual_stage1_local/best_model \
  --env residual \
  --algo sac \
  --club 7iron \
  --hand right \
  --episodes 5
```

## Visualize A Saved RL Policy

```bash
mjpython -m golf_rl.visualize_policy \
  artifacts/trained_models/residual_stage1_local/best_model \
  --env residual \
  --algo sac \
  --club 7iron \
  --hand right \
  --speed 16
```
