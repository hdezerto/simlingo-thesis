#!/bin/bash
set -euo pipefail

# Launch a clean full Bench2Drive evaluation for the released SimLingo baseline
# checkpoint using the current eval code. Run this from an
# interactive tmux session, not with sbatch. The script starts
# start_eval_simlingo.py, which then submits per-route Slurm jobs.

export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"

export EVAL_RUN_NAME="simlingo_baseline_evalfix_full"
export EVAL_AGENT_NAME="${EVAL_RUN_NAME}"
export MODEL_CKPT="${MODEL_CKPT:-${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"

source "${REPO_DIR}/thesis/env.sh"
export EVAL_RUN_NAME="simlingo_baseline_evalfix_full"
export EVAL_AGENT_NAME="${EVAL_RUN_NAME}"
export BENCH2DRIVE_ROOT="${EVAL_OUT_ROOT}/${EVAL_AGENT_NAME}/bench2drive"
cd "${REPO_DIR}"

test -f "${MODEL_CKPT}" || { echo "Missing model checkpoint: ${MODEL_CKPT}" >&2; exit 1; }
test -f "${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/.hydra/config.yaml" || { echo "Missing released baseline Hydra config" >&2; exit 1; }

mkdir -p "/tmp/${USER}_triton"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}_triton}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "${MAX_EVAL_JOBS:-8}" > max_num_jobs.txt

echo "Starting full Bench2Drive evaluation"
echo "  EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "  MODEL_CKPT=${MODEL_CKPT}"
echo "  BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT}"
echo "  max jobs=$(cat max_num_jobs.txt)"

python start_eval_simlingo.py
