#!/bin/bash
#SBATCH -p berzelius
#SBATCH -N 1
#SBATCH -C fat
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=simlingo_qformer_v8_freshlora
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/training/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/training/%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=hugo.afonso.dezerto@gmail.com

set -euo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
conda activate simlingo

export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"
source "${REPO_DIR}/thesis/env.sh"

cd "$REPO_DIR"

source "${REPO_DIR}/thesis/scripts/stage_simlingo_to_scratch.sh"
TRAIN_DATA_ROOT="${STAGED_DATA_ROOT}"
TRAIN_BUCKET_ROOT="${STAGED_BUCKET_ROOT}"

export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/tmp/$USER/triton-cache-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "LD_LIBRARY_PATH before unset: ${LD_LIBRARY_PATH:-<unset>}"
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH after unset: ${LD_LIBRARY_PATH:-<unset>}"

test -d "${TRAIN_DATA_ROOT}/data/simlingo" || { echo "dataset path missing: ${TRAIN_DATA_ROOT}/data/simlingo"; exit 1; }
test -f "${TRAIN_BUCKET_ROOT}/buckets_paths.pkl" || { echo "bucket file missing: ${TRAIN_BUCKET_ROOT}/buckets_paths.pkl"; exit 1; }
test -f "$MODEL_CKPT" || { echo "checkpoint missing: $MODEL_CKPT"; exit 1; }

echo "JOB ID $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Checkpoint: $MODEL_CKPT"
echo "Loading the SimLingo checkpoint but reinitializing LLM LoRA weights; training Q-former v8 with scene-fact supervision and dynamic sample weight 2.0."
echo "Training data root: ${TRAIN_DATA_ROOT}"
echo "Training bucket root: ${TRAIN_BUCKET_ROOT}"
nvidia-smi

python_args=(
  simlingo_training/train.py
  experiment=temporal_qformer_v8_fresh_lora
  gpus=8
  data_module.num_workers=8
  "data_module.base_dataset.data_path='${TRAIN_DATA_ROOT}'"
  "data_module.base_dataset.bucket_path='${TRAIN_BUCKET_ROOT}'"
  "checkpoint='${MODEL_CKPT}'"
  name=temporal_qformer_v8_fresh_lora_8gpu
)

python "${python_args[@]}"
