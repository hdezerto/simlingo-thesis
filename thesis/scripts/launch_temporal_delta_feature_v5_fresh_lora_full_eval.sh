#!/bin/bash
set -euo pipefail

# Launch a full Bench2Drive evaluation for the completed Delta feature v5
# fresh-LoRA run. Run this from an interactive tmux session, not with sbatch.
# The script starts start_eval_simlingo.py, which submits per-route Slurm jobs.

export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"

export EVAL_RUN_NAME="simlingo_temporal_delta_feature_v5_fresh_lora_full"
export EVAL_AGENT_NAME="${EVAL_RUN_NAME}"

RUN_DIR="${REPO_DIR}/outputs/2026_05_29_02_22_08_temporal_delta_feature_v5_fresh_lora_8gpu"
CKPT_NAME="final"
CKPT_DIR="${RUN_DIR}/checkpoints/${CKPT_NAME}.ckpt"
FP32_DIR="${RUN_DIR}/checkpoints/${CKPT_NAME}_fp32"
export MODEL_CKPT="${FP32_DIR}/pytorch_model.bin"

source "${REPO_DIR}/thesis/env.sh"
export EVAL_RUN_NAME="simlingo_temporal_delta_feature_v5_fresh_lora_full"
export EVAL_AGENT_NAME="${EVAL_RUN_NAME}"
export BENCH2DRIVE_ROOT="${EVAL_OUT_ROOT}/${EVAL_AGENT_NAME}/bench2drive"
cd "${REPO_DIR}"

test -d "${CKPT_DIR}" || { echo "Missing checkpoint directory: ${CKPT_DIR}" >&2; exit 1; }

export CUDA_HOME="${CUDA_HOME:-/software/sse/manual/CUDA/12.1.1_530.30.02}"
export PATH="${CUDA_HOME}/bin:${PATH}"
test -d "${CUDA_HOME}" || { echo "Missing CUDA_HOME: ${CUDA_HOME}" >&2; exit 1; }

mkdir -p "/tmp/${USER}_triton"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}_triton}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ ! -f "${MODEL_CKPT}" ]; then
  echo "Converting ${CKPT_DIR} to fp32 checkpoint..."
  /home/x_hugaf/.conda/envs/simlingo/bin/python     "${CKPT_DIR}/zero_to_fp32.py"     "${CKPT_DIR}"     "${FP32_DIR}"
fi

test -f "${MODEL_CKPT}" || { echo "Missing model checkpoint: ${MODEL_CKPT}" >&2; exit 1; }

echo "${MAX_EVAL_JOBS:-8}" > max_num_jobs.txt

echo "Starting full Bench2Drive evaluation"
echo "  EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "  MODEL_CKPT=${MODEL_CKPT}"
echo "  BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "  max jobs=$(cat max_num_jobs.txt)"

python start_eval_simlingo.py
