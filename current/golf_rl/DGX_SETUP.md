# DGX Spark Setup

Create the venv on the DGX itself, not on the Mac. Python virtual environments are not portable across macOS and Linux/CUDA.

## 1. Copy Or Pull The Project Onto The DGX

From the project root on the DGX:

```bash
cd /path/to/mujoco-test/current
```

## 2. Create The DGX Venv

```bash
bash setup_dgx_venv.sh
```

Then activate it:

```bash
source .venv-dgx/bin/activate
```

## 3. Confirm CUDA Works

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

You want `cuda: True`.

If it says `False`, the venv is using a CPU-only PyTorch build. On DGX hardware, the best fix is usually to install the NVIDIA/PyTorch build provided by the DGX software stack, or run the project inside an NVIDIA PyTorch container and create the venv there.

## 4. Train With CUDA

```bash
python -m golf_rl.train_sac \
  --club 7iron \
  --hand right \
  --timesteps 1000000 \
  --device cuda \
  --model-dir trained_models/stageb_dgx \
  --log-dir runs/sac_stageb_dgx
```

## 5. Evaluate

```bash
python -m golf_rl.evaluate_policy trained_models/stageb_dgx/sac_7iron_right_final \
  --algo sac \
  --club 7iron \
  --hand right \
  --episodes 5
```

## 6. Visualize

If the DGX has a display:

```bash
python -m golf_rl.visualize_policy trained_models/stageb_dgx/sac_7iron_right_final \
  --algo sac \
  --club 7iron \
  --hand right \
  --speed 16
```

If the DGX is headless, train/evaluate there and copy the saved `.zip` model back to the Mac for visualization.

## Note On Speed

The neural network training can use CUDA. MuJoCo stepping is still CPU-side in this setup, so the next major speed upgrade is running multiple environments in parallel.
