#!/bin/bash
# =============================================================================
# Circuit Breakers Fir Setup Script (Alliance)
# =============================================================================
#
# Fir conventions (per Alliance docs):
# - SCRATCH is typically available at: $HOME/scratch
#   (often a symlink to /scratch/$USER/...) and is NOT backed up.
#
# This script:
# - Creates a Python venv in $HOME/scratch/.venvs/cb_env (by default)
# - Moves HF / datasets / wandb / torch caches to $HOME/scratch
# - Creates a repo-local logs/ directory used by sbatch scripts
#
# Usage (on Fir login node):
#   ssh fir.alliancecan.ca
#   cd $HOME/scratch
#   git clone <your-repo-url> harmful-agents-meta-dataset
#   cd harmful-agents-meta-dataset
#   bash scripts/hpc_setup_fir.sh
#
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  CB Fir Setup (Alliance)"
echo "=============================================="

SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$SCRATCH_DIR/harmful-agents-meta-dataset"
VENV_DIR="$SCRATCH_DIR/.venvs/cb_env"
CACHE_ROOT="$SCRATCH_DIR/cb_cache"

mkdir -p "$SCRATCH_DIR/.venvs" "$REPO_DIR/logs" "$CACHE_ROOT"/{hf,wandb,torch,xdg}
cd "$REPO_DIR"

echo "Repo:   $REPO_DIR"
echo "Venv:   $VENV_DIR"
echo "Cache:  $CACHE_ROOT"

# Modules (Fir already has `module`; no extra sourcing needed)
module purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

echo "Modules loaded:"
module list

# uv install (user-space)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"

# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python 3.11
fi

source "$VENV_DIR/bin/activate"
python -V
which python

export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo "Cache root: $CACHE_ROOT"

uv pip install --upgrade pip setuptools wheel

echo "Installing Python deps (requirements.txt)..."
uv pip install -r requirements.txt

echo "Ensuring CUDA PyTorch wheels (cu121)..."
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("num gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}:", torch.cuda.get_device_name(i))
PY

echo "=============================================="
echo "✅ Fir setup complete"
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Logs directory:"
echo "  $REPO_DIR/logs"
echo "Next: submit the Fir smoke test:"
echo "  cd $REPO_DIR && sbatch slurm/fir_cb_llama4_1xh100_debug.sbatch"
echo "=============================================="
