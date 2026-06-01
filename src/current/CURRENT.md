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

## Completed Kinematics Swing

This is the hand-built two-arm + chest swing animation. It is the clean visual
reference for the motion sequence.

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/mjpython two_arm_chest_full_swing.py --club 7iron --hand right --speed 1
```

## Current Best Physics-Based Swing

This replays the saved best SAC policy on the joint-accurate two-arm physics
model.

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/mjpython -m golf_rl.visualize_two_arm_joint_policy \
  artifacts/trained_models_two_arm_joint/best_model \
  --club 7iron \
  --hand right \
  --speed 6
```

If the saved policy is missing, view the physics baseline directly:

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/mjpython two_arm_joint_baseline.py --club 7iron --hand right --speed 6
```

## Known-Best Historical Version

The best-performing older commit has been restored outside this folder:

```bash
cd ../best_model
/Users/zachhuang/mujoco-test/.venv/bin/mjpython human_right_arm_biomech_swing.py --club 7iron --hand right
```

Use `../best_model` as the known-good reference if the newer experiments get
too noisy.

## Current Two-Arm Physics RL Training

Restart training from the current physics-based two-arm environment:

```bash
cd /Users/zachhuang/mujoco-test/src/current
PYTHONDONTWRITEBYTECODE=1 /Users/zachhuang/mujoco-test/.venv/bin/python -m golf_rl.train_two_arm_joint_sac \
  --club 7iron \
  --hand right \
  --timesteps 100000 \
  --device cpu
```

Each training run now saves into its own timestamped folder under
`artifacts/trained_models_two_arm_joint/`. By default, a new run resumes from
`artifacts/trained_models_two_arm_joint/best_model`, then promotes the best
model from the new run back to that same path for the next run.

The current reward judges the strike with one short post-impact launch report
card instead of relying on full-rollout distance. After impact, training watches
about 90 simulator steps, scores forward ball speed, straightness, and useful
height, penalizes sideways velocity/drift, then ends the episode. A run only
replaces the shared `best_model` if its evaluation score beats the previous
best score under the current reward version.

SAC now uses lower default exploration for this residual-control stage
(`--ent-coef 0.03`) and a lower learning rate (`--learning-rate 0.0001`), since
the policy is meant to make small corrections to an existing swing rather than
invent a new motion from scratch. No-contact deterministic evaluations are
penalized explicitly and end near the expected contact window.

To force a brand-new policy instead of continuing from the previous best:

```bash
cd /Users/zachhuang/mujoco-test/src/current
PYTHONDONTWRITEBYTECODE=1 /Users/zachhuang/mujoco-test/.venv/bin/python -m golf_rl.train_two_arm_joint_sac \
  --club 7iron \
  --hand right \
  --timesteps 100000 \
  --device cpu \
  --fresh
```

To resume from a specific older run:

```bash
cd /Users/zachhuang/mujoco-test/src/current
PYTHONDONTWRITEBYTECODE=1 /Users/zachhuang/mujoco-test/.venv/bin/python -m golf_rl.train_two_arm_joint_sac \
  --club 7iron \
  --hand right \
  --timesteps 100000 \
  --device cpu \
  --resume-from artifacts/trained_models_two_arm_joint/name_of_run/best_model
```

## Older Residual RL Training

This was the earlier one-arm residual RL path and is kept for reference:

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/python -m golf_rl.train_sac \
  --env residual \
  --club 7iron \
  --hand right \
  --timesteps 100000 \
  --device cpu \
  --model-dir artifacts/trained_models/residual_stage1_local \
  --log-dir artifacts/runs/sac_residual_stage1_local
```

## DGX Setup

DGX setup files are grouped in `dgx/`:

```bash
bash dgx/setup_dgx_venv.sh
```

## Evaluate

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/python -m golf_rl.evaluate_policy \
  artifacts/trained_models/residual_stage1_local/best_model \
  --env residual \
  --algo sac \
  --club 7iron \
  --hand right \
  --episodes 5
```

## Visualize A Saved RL Policy

```bash
cd /Users/zachhuang/mujoco-test/src/current
/Users/zachhuang/mujoco-test/.venv/bin/mjpython -m golf_rl.visualize_policy \
  artifacts/trained_models/residual_stage1_local/best_model \
  --env residual \
  --algo sac \
  --club 7iron \
  --hand right \
  --speed 16
```
