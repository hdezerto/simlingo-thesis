#!/bin/bash
set -euo pipefail

# Profile model-step inference cost on Bench2Drive route 3373 only. This is a
# direct one-route launcher, separate from the full-evaluation manager.

export PROJECT_ID="${PROJECT_ID:-berzelius-2023-154}"
export USERNAME="${USERNAME:-${USER:-}}"
export BASE_DIR="${BASE_DIR:-/proj/${PROJECT_ID}/users/${USERNAME}}"
export REPO_DIR="${REPO_DIR:-${BASE_DIR}/simlingo-thesis}"

source "${REPO_DIR}/thesis/env.sh"
cd "${REPO_DIR}"

export PROFILE_ROUTE_ID="3373"
export PROFILE_ROUTE_INDEX="067"
export PROFILE_ROUTE_FILE="${REPO_DIR}/leaderboard/data/bench2drive_split/bench2drive_67.xml"
export PROFILE_SEED="${PROFILE_SEED:-1}"
export PROFILE_RUN_TAG="${PROFILE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export PROFILE_ROOT="${PROFILE_ROOT:-${BASE_DIR}/eval_results/Bench2Drive/inference_profiles}"
export PROFILE_EVAL_ROOT="${PROFILE_ROOT}/eval_runs/${PROFILE_RUN_TAG}"
export PROFILE_JSON_ROOT="${PROFILE_ROOT}/profile_json/${PROFILE_RUN_TAG}"
export PROFILE_SLURM_ROOT="${PROFILE_ROOT}/slurm/${PROFILE_RUN_TAG}"
export SIMLINGO_PROFILE_WARMUP="${SIMLINGO_PROFILE_WARMUP:-20}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_HOME="${CUDA_HOME:-/software/sse/manual/CUDA/12.1.1_530.30.02}"
export PATH="${CUDA_HOME}/bin:${PATH}"
test -d "${CUDA_HOME}" || { echo "Missing CUDA_HOME: ${CUDA_HOME}" >&2; exit 1; }
test -f "${PROFILE_ROUTE_FILE}" || { echo "Missing route file: ${PROFILE_ROUTE_FILE}" >&2; exit 1; }

mkdir -p "/tmp/${USER}_triton" "${PROFILE_JSON_ROOT}" "${PROFILE_SLURM_ROOT}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}_triton}"

convert_fp32_if_needed() {
  local ckpt_dir="$1"
  local fp32_dir="$2"
  local model_ckpt="${fp32_dir}/pytorch_model.bin"

  test -d "${ckpt_dir}" || { echo "Missing checkpoint directory: ${ckpt_dir}" >&2; exit 1; }
  if [ ! -f "${model_ckpt}" ]; then
    echo "Converting ${ckpt_dir} to fp32 checkpoint..."
    /home/x_hugaf/.conda/envs/simlingo/bin/python \
      "${ckpt_dir}/zero_to_fp32.py" \
      "${ckpt_dir}" \
      "${fp32_dir}"
  fi
  test -f "${model_ckpt}" || { echo "Missing model checkpoint: ${model_ckpt}" >&2; exit 1; }
}

submit_profile_job() {
  local variant_label="$1"
  local run_slug="$2"
  local checkpoint="$3"
  local history_mode="${4:-real}"
  local world_port="$5"
  local tm_port="$6"

  local eval_run_name="${run_slug}_${PROFILE_RUN_TAG}"
  local base_dir="${PROFILE_EVAL_ROOT}/${eval_run_name}/bench2drive/${PROFILE_SEED}"
  local viz_path="${base_dir}/viz/${PROFILE_ROUTE_INDEX}"
  local result_file="${base_dir}/res/${PROFILE_ROUTE_INDEX}_res.json"
  local log_file="${base_dir}/out/${PROFILE_ROUTE_INDEX}_out.log"
  local err_file="${base_dir}/err/${PROFILE_ROUTE_INDEX}_err.log"
  local job_file="${base_dir}/run/profile_${PROFILE_ROUTE_INDEX}.sh"
  local profile_dir="${PROFILE_JSON_ROOT}/${eval_run_name}"

  mkdir -p "${base_dir}/run" "${base_dir}/res" "${base_dir}/out" "${base_dir}/err" "${viz_path}" "${profile_dir}"

  cat > "${job_file}" <<SLURM
#!/bin/bash
#SBATCH --job-name=${eval_run_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH -o ${log_file}
#SBATCH -e ${err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

echo JOB ID \$SLURM_JOB_ID

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh

cd ${REPO_DIR}

export TEMPORAL_HISTORY_MODE="${history_mode}"
export EVAL_RUN_NAME="${eval_run_name}"
export EVAL_AGENT_NAME="${eval_run_name}"
export EVAL_ROUTE_INDEX="${PROFILE_ROUTE_INDEX}"
export EVAL_ROUTE_B2D_ID="${PROFILE_ROUTE_ID}"
export EVAL_ROUTE_FILE="${PROFILE_ROUTE_FILE}"
export EVAL_SEED="${PROFILE_SEED}"
export SIMLINGO_PROFILE_INFERENCE=1
export SIMLINGO_PROFILE_WARMUP="${SIMLINGO_PROFILE_WARMUP}"
export SIMLINGO_PROFILE_DIR="${profile_dir}"
export SIMLINGO_PROFILE_VARIANT="${variant_label}"
export CARLA_ROOT="${CARLA_ROOT}"
export PYTHONPATH=\$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=\$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=${REPO_DIR}:${REPO_DIR}/Bench2Drive/leaderboard:${REPO_DIR}/Bench2Drive/scenario_runner:\$PYTHONPATH
export SCENARIO_RUNNER_ROOT=${REPO_DIR}/Bench2Drive/scenario_runner
export SAVE_PATH="${viz_path}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export WANDB_MODE="${WANDB_MODE}"

mkdir -p /tmp/\$SLURM_JOB_ID
echo '#!/bin/bash' > /tmp/\$SLURM_JOB_ID/xdg-user-dir
echo 'echo /scratch/local' >> /tmp/\$SLURM_JOB_ID/xdg-user-dir
chmod +x /tmp/\$SLURM_JOB_ID/xdg-user-dir
export PATH=/tmp/\$SLURM_JOB_ID:\$PATH

python -u ${REPO_DIR}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="${PROFILE_ROUTE_FILE}" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="${result_file}" \
  --timeout=600 \
  --agent=${REPO_DIR}/team_code/agent_simlingo.py \
  --agent-config="${checkpoint}" \
  --traffic-manager-seed="${PROFILE_SEED}" \
  --port=${world_port} \
  --traffic-manager-port=${tm_port}
SLURM

  echo "Submitting ${variant_label} profiling job"
  echo "  route=${PROFILE_ROUTE_ID} seed=${PROFILE_SEED}"
  echo "  result=${result_file}"
  echo "  profile_dir=${profile_dir}"
  sbatch "${job_file}"
}

baseline_ckpt="${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
test -f "${baseline_ckpt}" || { echo "Missing baseline checkpoint: ${baseline_ckpt}" >&2; exit 1; }
test -f "${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/.hydra/config.yaml" || { echo "Missing released baseline Hydra config" >&2; exit 1; }

qformer_run="${REPO_DIR}/outputs/2026_05_24_17_47_21_temporal_qformer_v8_fresh_lora_8gpu"
qformer_ckpt="${qformer_run}/checkpoints/final_fp32/pytorch_model.bin"
convert_fp32_if_needed "${qformer_run}/checkpoints/final.ckpt" "${qformer_run}/checkpoints/final_fp32"

delta_run="${REPO_DIR}/outputs/2026_05_29_02_22_08_temporal_delta_feature_v5_fresh_lora_8gpu"
delta_ckpt="${delta_run}/checkpoints/final_fp32/pytorch_model.bin"
convert_fp32_if_needed "${delta_run}/checkpoints/final.ckpt" "${delta_run}/checkpoints/final_fp32"

submit_profile_job "Baseline" "profile_baseline_evalfix_route3373_seed${PROFILE_SEED}" "${baseline_ckpt}" "real" 11067 31067
submit_profile_job "Q-former v8 fresh-LoRA" "profile_qformer_v8_fresh_lora_route3373_seed${PROFILE_SEED}" "${qformer_ckpt}" "real" 11167 31167
submit_profile_job "Delta v5 fresh-LoRA" "profile_delta_v5_fresh_lora_route3373_seed${PROFILE_SEED}" "${delta_ckpt}" "real" 11267 31267

echo "Submitted route-3373 profiling jobs. Check with:"
echo "  squeue -u \"\${USER}\""
echo "Summarize after all jobs finish with:"
echo "  python thesis/analysis/summarize_inference_profiles.py --root ${PROFILE_JSON_ROOT} --out thesis/results/inference_profile_route3373_${PROFILE_RUN_TAG}.txt"
