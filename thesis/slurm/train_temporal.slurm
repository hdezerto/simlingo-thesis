#!/bin/bash
#SBATCH -p berzelius
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=simlingo_temporal
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/training/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/training/%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=hugo.afonso.dezerto@gmail.com

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "LD_LIBRARY_PATH before unset: ${LD_LIBRARY_PATH:-<unset>}"
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after unset: ${LD_LIBRARY_PATH:-<unset>}"

test -d database/simlingo/data/simlingo || { echo "dataset path missing: database/simlingo/data/simlingo"; exit 1; }
test -f database/bucketsv2_simlingo/buckets_paths.pkl || { echo "bucket file missing: database/bucketsv2_simlingo/buckets_paths.pkl"; exit 1; }
test -f "$MODEL_CKPT" || { echo "checkpoint missing: $MODEL_CKPT"; exit 1; }

echo "JOB ID $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Checkpoint: $MODEL_CKPT"
nvidia-smi

python simlingo_training/train.py \
  experiment=temporal_qformer_v2 \
  gpus=8 \
  data_module.num_workers=8 \
  "checkpoint='$MODEL_CKPT'" \
  name=temporal_qformer_v2_8gpu
