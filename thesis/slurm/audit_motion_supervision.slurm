#!/bin/bash
#SBATCH -p berzelius-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=simlingo_supervision_audit
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/supervision/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/supervision/%j.err
#SBATCH --mail-type=END,FAIL
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

cd "${REPO_DIR}"
mkdir -p "${BASE_DIR}/logs/supervision"

source "${REPO_DIR}/thesis/scripts/stage_simlingo_to_scratch.sh"

audit_args=(
  thesis/scripts/audit_motion_supervision.py
  --data-root "${STAGED_DATA_ROOT}"
  --output-dir thesis/results/supervision_audit
  --max-routes "${AUDIT_MAX_ROUTES:-200}"
  --max-samples "${AUDIT_MAX_SAMPLES:-100}"
  --frame-step "${AUDIT_FRAME_STEP:-5}"
  --seed "${AUDIT_SEED:-9876}"
  --pred-len "${AUDIT_PRED_LEN:-10}"
)

if [ -n "${AUDIT_ROUTE_CONTAINS:-}" ]; then
  audit_args+=(--route-contains "${AUDIT_ROUTE_CONTAINS}")
fi
if [ "${AUDIT_INCLUDE_VALIDATION:-0}" = "1" ]; then
  audit_args+=(--include-validation)
fi
if [ -n "${AUDIT_FRAME_START:-}" ]; then
  audit_args+=(--frame-start "${AUDIT_FRAME_START}")
fi
if [ -n "${AUDIT_FRAME_END:-}" ]; then
  audit_args+=(--frame-end "${AUDIT_FRAME_END}")
fi
if [ "${AUDIT_ONLY_MOTION_TEXT:-0}" = "1" ]; then
  audit_args+=(--only-motion-text)
fi
if [ "${AUDIT_SHUFFLE_FRAMES:-1}" = "1" ]; then
  audit_args+=(--shuffle-frames)
fi
if [ "${AUDIT_COPY_RGB:-1}" = "1" ]; then
  audit_args+=(--copy-rgb)
fi
if [ "${AUDIT_COPY_ALL_RGB:-0}" = "1" ]; then
  audit_args+=(--copy-all-rgb)
fi

python "${audit_args[@]}"
