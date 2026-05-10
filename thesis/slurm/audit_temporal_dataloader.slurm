#!/bin/bash
#SBATCH -p berzelius-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=simlingo_temporal_data_audit
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/audit/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/audit/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hugo.afonso.dezerto@gmail.com

set -euo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo

export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"
source "${REPO_DIR}/thesis/env.sh"

cd "${REPO_DIR}"

mkdir -p "${LOG_ROOT}/audit"

source "${REPO_DIR}/thesis/scripts/stage_simlingo_to_scratch.sh"

load_item_args=()
if [ "${AUDIT_LOAD_ITEM:-1}" = "1" ]; then
  load_item_args=(--load-item)
fi

python thesis/scripts/audit_temporal_dataloader.py \
  --experiment "${AUDIT_EXPERIMENT:-temporal_qformer_v4_nogate}" \
  --data-root "${STAGED_DATA_ROOT}" \
  --bucket-root "${STAGED_BUCKET_ROOT}" \
  --dataset "${AUDIT_DATASET:-all}" \
  --split "${AUDIT_SPLIT:-train}" \
  --bucket "${AUDIT_BUCKET:-all}" \
  --max-routes "${AUDIT_MAX_ROUTES:-100}" \
  --max-samples "${AUDIT_MAX_SAMPLES:-40}" \
  --seed "${AUDIT_SEED:-9876}" \
  "${load_item_args[@]}"
