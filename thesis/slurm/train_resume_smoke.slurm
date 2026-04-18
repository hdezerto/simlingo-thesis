#!/bin/bash
#SBATCH -p berzelius
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=simlingo_resume_smoke
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/training/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/training/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hugoad@kth.se

set -euo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
conda activate simlingo
source thesis/env.sh

cd "$REPO_DIR"

export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/tmp/$USER/triton-cache-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "LD_LIBRARY_PATH before unset: ${LD_LIBRARY_PATH:-<unset>}"
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after unset: ${LD_LIBRARY_PATH:-<unset>}"

RESUME_CKPT="${REPO_DIR}/outputs/2026-04-18/14-26-58/checkpoints/last.ckpt"

test -d database/simlingo/data/simlingo || { echo "dataset path missing: database/simlingo/data/simlingo"; exit 1; }
test -f database/bucketsv2_simlingo/buckets_paths.pkl || { echo "bucket file missing: database/bucketsv2_simlingo/buckets_paths.pkl"; exit 1; }
test -d "$RESUME_CKPT" || { echo "resume checkpoint missing: $RESUME_CKPT"; exit 1; }

echo "JOB ID $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Resume checkpoint: $RESUME_CKPT"
nvidia-smi

python simlingo_training/train.py \
  experiment=debug \
  debug=true \
  gpus=1 \
  data_module.batch_size=1 \
  data_module.num_workers=0 \
  overfit=1 \
  max_epochs=2 \
  val_every_n_epochs=1 \
  resume=true \
  "resume_path='$RESUME_CKPT'" \
  name=baseline_resume_smoke_ckpt
