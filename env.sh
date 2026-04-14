# Shared environment defaults for Berzelius thesis workflow.
# Single source of truth for paths/accounts used by scripts and docs.

# Core identity/path variables
export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
if [ -z "${USERNAME}" ]; then
	echo "env.sh: USERNAME is not set. Export USERNAME before sourcing env.sh." >&2
	return 1 2>/dev/null || exit 1
fi
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"

# Repository and runtime layout
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"
export CARLA_VERSION="${CARLA_VERSION:-0.9.15}"
export CARLA_ROOT="${CARLA_ROOT:-${BASE_DIR}/carla/CARLA_${CARLA_VERSION}}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${BASE_DIR}/checkpoints}"
export BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive}"
export LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/logs}"
export MODEL_CKPT="${MODEL_CKPT:-${CHECKPOINT_ROOT}/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"

# Slurm defaults
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-berzelius-2025-435}"
export SLURM_PARTITION="${SLURM_PARTITION:-berzelius}"

# Backward-compatible aliases used by existing scripts
export PROJECT_ROOT="${PROJECT_ROOT:-${BASE_DIR}}"
export WORK_DIR="${WORK_DIR:-${REPO_DIR}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}}"
export LOG_DIR="${LOG_DIR:-${LOG_ROOT}}"
export RESULT_DIR="${RESULT_DIR:-${BASE_DIR}/results}"

# CARLA evaluation infrastructure
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${WORK_DIR}/scenario_runner}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${WORK_DIR}/leaderboard}"

# Python paths
export PYTHONPATH="$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}"

