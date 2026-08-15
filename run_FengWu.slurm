#!/bin/bash
#SBATCH --job-name=FengWu_2020
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --tmp=300G
#SBATCH --exclude=fc10507,fc10509,fc10510,fc10511,fc10417
#SBATCH --output=FengWu_2020_%j.out

set -e

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load arrow
module load eccodes
module load openmpi/4.1.5
module load mpi4py

cd "$SLURM_SUBMIT_DIR"

source "$HOME/model_env/bin/activate"

export TMPDIR=$SLURM_TMPDIR
export XDG_CACHE_HOME=$SLURM_TMPDIR/.cache
export TORCH_HOME=$SCRATCH/.cache/torch
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "HOST=$(hostname)"
echo "JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_TMPDIR=$SLURM_TMPDIR"
echo "PYTHON=$(which python)"

nvidia-smi || true

python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; stop before running FengWu")

x = torch.randn(10, device="cuda")
print("CUDA tensor test OK:", x.device)
print("GPU:", torch.cuda.get_device_name(0))
PY

python run_FengWu.py
