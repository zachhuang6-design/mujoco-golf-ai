#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv-dgx"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

echo "Creating DGX virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing golf RL dependencies"
python -m pip install -r dgx/requirements-dgx.txt

echo
echo "Checking PyTorch / CUDA visibility"
python - <<'PY'
import sys

import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
else:
    print("WARNING: PyTorch cannot see CUDA from this virtual environment.")
    print("If this is a DGX machine, install/use the NVIDIA-provided PyTorch build,")
    print("or recreate the venv inside the DGX software stack/container.")
PY

echo
echo "DGX venv setup complete."
echo "Activate it with:"
echo "  source .venv-dgx/bin/activate"
echo
echo "Train SAC on GPU with:"
echo "  python -m golf_rl.train_sac --env residual --club 7iron --hand right --timesteps 1000000 --device cuda --model-dir artifacts/trained_models/residual_stage1_dgx --log-dir artifacts/runs/sac_residual_stage1_dgx"
