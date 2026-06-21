# SimLingo Temporal Training on Berzelius

This guide is linked from the repository `README.md` and gives the
Berzelius-specific workflow for reproducing the temporal thesis experiments. It
keeps operational details here; method descriptions and result interpretation
belong to the thesis report.

## Index

- [1. Scope and Branch](#1-scope-and-branch)
- [2. Environment](#2-environment)
- [3. Dataset Archive Workflow](#3-dataset-archive-workflow)
- [4. Training From Archives](#4-training-from-archives)
- [5. FlashAttention2](#5-flashattention2)
- [6. Training Jobs](#6-training-jobs)
- [7. Checkpoints](#7-checkpoints)
- [8. Full Bench2Drive Evaluation](#8-full-bench2drive-evaluation)
- [9. Selected Renders](#9-selected-renders)
- [10. Final Artifacts](#10-final-artifacts)

## 1. Scope and Branch

Use the default thesis branch, `temporal-module`. It contains the final Q-former
and feature-delta experiments, dataset staging, render/profiling fixes, and
committed evidence used by the report.

## 2. Environment

Use one persistent tmux session for interactive work:

```bash
tmux new -As temporal
```

If you disconnect from Berzelius, attach to the same session again:

```bash
tmux attach -t temporal
```

Check existing tmux sessions:

```bash
tmux ls
```

Detach without stopping the session with `Ctrl-b`, then `d`.

Load the environment inside tmux:

```bash
export USERNAME="${USERNAME:-${USER}}"
cd /proj/berzelius-2023-154/users/${USERNAME}/simlingo-thesis

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh
```

Useful variables from `thesis/env.sh`:

- `BASE_DIR=/proj/${PROJECT_ID}/users/${USERNAME}`
- `REPO_DIR=${BASE_DIR}/simlingo-thesis`
- `LOG_ROOT=${BASE_DIR}/logs`
- `EVAL_OUT_ROOT=${BASE_DIR}/eval_results/Bench2Drive`
- `EVAL_RUN_NAME=simlingo`
- `EVAL_AGENT_NAME=${EVAL_RUN_NAME}`
- `BENCH2DRIVE_ROOT=${EVAL_OUT_ROOT}/${EVAL_AGENT_NAME}/bench2drive`
- `MODEL_CKPT=${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- `SLURM_ACCOUNT=berzelius-2025-435`

Dedicated evaluation launchers override the default `EVAL_RUN_NAME` when needed.

Note: the storage project directory is `/proj/berzelius-2023-154`, but the
current Slurm allocation account is `berzelius-2025-435`.

Quick check:

```bash
echo "$REPO_DIR"
echo "$MODEL_CKPT"
echo "$SLURM_ACCOUNT"
test -f "$MODEL_CKPT" && echo "checkpoint ok"
```

## 3. Dataset Archive Workflow

The recommended Berzelius dataset format is an archive set on `/proj`, staged to
node-local scratch at job start. This avoids keeping millions of extracted files
under the project directory.

Final archive location:

```bash
${SIMLINGO_ARCHIVE_ROOT}
```

Expected archive files:

- `simlingo_part_*.tar`
- `dataset_archives.txt`
- `dataset_metadata.env`
- `stage_parent_dirs.txt`
- `bucketsv2_simlingo.tar`

The temporary extracted dataset is used only while building the archive set. Do
not keep the full extracted dataset in `${BASE_DIR}/database/simlingo`.

### Build Archives

Use a node with enough scratch space:

```bash
cd "${REPO_DIR}"
source thesis/env.sh

export HF_DATASET_ROOT="${BASE_DIR}/dataset_sources/simlingo_hf"
export SIMLINGO_BUILD_ROOT="${SNIC_TMP:-/scratch/local}/${USERNAME}/simlingo_archive_build"
export SIMLINGO_DATA_ROOT="${SIMLINGO_BUILD_ROOT}/simlingo"
export SIMLINGO_BUCKET_ROOT="${SIMLINGO_BUILD_ROOT}/bucketsv2_simlingo"
export SIMLINGO_ARCHIVE_ROOT="${BASE_DIR}/database_archives/simlingo"
```

Download from Hugging Face:

```bash
mkdir -p "$(dirname "${HF_DATASET_ROOT}")"

if [ ! -d "${HF_DATASET_ROOT}/.git" ]; then
  git lfs install
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/RenzKa/simlingo "${HF_DATASET_ROOT}"
fi

cd "${HF_DATASET_ROOT}"
git lfs pull
```

Create the temporary scratch source tree:

```bash
rm -rf "${SIMLINGO_BUILD_ROOT}"
mkdir -p "${SIMLINGO_DATA_ROOT}" "${SIMLINGO_BUCKET_ROOT}"
cp "${HF_DATASET_ROOT}/buckets_paths.pkl" "${SIMLINGO_BUCKET_ROOT}/"

for f in "${HF_DATASET_ROOT}"/*.tar.gz; do
  echo "Extracting $f"
  tar -xzf "$f" -C "${SIMLINGO_DATA_ROOT}"
done
```

Pack the archive set:

```bash
cd "${REPO_DIR}"
bash thesis/scripts/pack_simlingo_dataset.sh
```

Verify:

```bash
test -s "${SIMLINGO_ARCHIVE_ROOT}/dataset_archives.txt" && echo "dataset archive manifest ok"
test -s "${SIMLINGO_ARCHIVE_ROOT}/dataset_metadata.env" && cat "${SIMLINGO_ARCHIVE_ROOT}/dataset_metadata.env"
test -s "${SIMLINGO_ARCHIVE_ROOT}/stage_parent_dirs.txt" && echo "parent dirs manifest ok"
test -f "${SIMLINGO_ARCHIVE_ROOT}/bucketsv2_simlingo.tar" && echo "bucket archive ok"
```

After verification:

```bash
rm -rf "${SIMLINGO_BUILD_ROOT}"
```

Knobs:

- `SIMLINGO_ARCHIVE_PARTS`, default `8`: number of dataset tar parts.
- `SIMLINGO_STAGE_JOBS`, default `8`: number of parallel unpack jobs.

## 4. Training From Archives

Archive-aware Slurm launchers:

- source `thesis/scripts/stage_simlingo_to_scratch.sh`
- read `${SIMLINGO_ARCHIVE_ROOT}/dataset_archives.txt`
- unpack archives to `${SIMLINGO_SCRATCH_BASE}/simlingo/${SLURM_JOB_ID}`
- unpack `bucketsv2_simlingo.tar`
- validate route count against `dataset_metadata.env`
- train from `TRAIN_DATA_ROOT="${STAGED_DATA_ROOT}"`
- train from `TRAIN_BUCKET_ROOT="${STAGED_BUCKET_ROOT}"`

Important:

- no repo-local dataset symlinks are needed
- jobs do not read from Hugging Face `.tar.gz` files
- jobs fail immediately if the archive set is missing or invalid
- do not delete or move `${SIMLINGO_ARCHIVE_ROOT}` while a job may still need it

## 5. FlashAttention2

Install only if the environment does not already have it:

```bash
export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=8
python -m pip install flash-attn --no-build-isolation
```

## 6. Training Jobs

Training jobs use archive-aware Slurm launchers under `thesis/slurm/`.
Version-specific experiment settings live in
`simlingo_training/config/experiment/`.

Shared hardware request:

- `8` A100 80GB GPUs
- `64` CPU cores
- `900G` RAM
- `3` days walltime

Create logs:

```bash
mkdir -p "${LOG_ROOT}/training"
```

Launch a training job:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch thesis/slurm/<training_launcher>.slurm
```

Final thesis training launchers:

```bash
thesis/slurm/train_temporal_qformer_v8_fresh_lora.slurm              # Q-former, fresh LoRA
thesis/slurm/train_temporal_delta_feature_v5_fresh_lora.slurm        # feature-delta adapter, fresh LoRA
thesis/slurm/train_temporal_qformer_v8_fresh_lora_no_temporal.slurm  # no-temporal control, fresh LoRA
thesis/slurm/train_temporal_qformer_v8_loaded_lora.slurm             # Q-former loaded-LoRA diagnostic
```

Older Q-former, gate, and delta launchers are kept for provenance. They
should not be treated as final thesis comparisons because they used earlier
supervision code or exploratory settings.

Each launcher:

- sources `thesis/env.sh`
- stages the dataset archives to node-local scratch
- passes staged data and bucket roots to Hydra
- loads the SimLingo `epoch=013` checkpoint from `MODEL_CKPT`
- writes logs to `${LOG_ROOT}/training/<jobid>.out` and `.err`

Monitor jobs:

```bash
squeue -u "${USER}"
tail -f "${LOG_ROOT}/training/<jobid>.out"
tail -f "${LOG_ROOT}/training/<jobid>.err"
```

Check expected start time for pending jobs:

```bash
squeue --start -j <jobid>
```

Check current GPU allocation:

```bash
scontrol show job <jobid>
```

## 7. Checkpoints

DeepSpeed checkpoints are directories:

- `outputs/.../checkpoints/epoch=007.ckpt`
- `outputs/.../checkpoints/final.ckpt`
- `outputs/.../checkpoints/last.ckpt`, in some older/resumed runs

These are training-resume checkpoints, not plain `torch.load(...)` files.

Convert to fp32 for many inference/render jobs:

```bash
cd "${REPO_DIR}"
mkdir -p /tmp/${USER}_triton

RUN_DIR="outputs/<run_name>"
EPOCH="final"        # or epoch=013, etc.

PYTHONPATH="${REPO_DIR}" \
TRITON_CACHE_DIR="/tmp/${USER}_triton" \
/home/x_hugaf/.conda/envs/simlingo/bin/python \
"${RUN_DIR}/checkpoints/${EPOCH}.ckpt/zero_to_fp32.py" \
"${RUN_DIR}/checkpoints/${EPOCH}.ckpt" \
"${RUN_DIR}/checkpoints/${EPOCH}_fp32"
```

Result:

- `${RUN_DIR}/checkpoints/${EPOCH}_fp32/pytorch_model.bin`

Rendering can load either the DeepSpeed checkpoint directory or the converted
`.bin`. The `.bin` usually starts faster across many jobs.

## 8. Full Bench2Drive Evaluation

Run full evaluations from tmux with the dedicated launchers. They set the
checkpoint, unique eval name, output root, fp32 conversion when needed, and
`max_num_jobs.txt`.

Main thesis evaluations:

```bash
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_baseline_current_full_eval.sh
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_temporal_qformer_v8_fresh_lora_final_full_eval.sh
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_temporal_qformer_v8_fresh_lora_no_temporal_full_eval.sh
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_temporal_delta_feature_v5_fresh_lora_full_eval.sh
```

Loaded-LoRA full-evaluation diagnostic:

```bash
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_temporal_qformer_v8_loaded_lora_full_eval.sh
```

Repeat-current full-evaluation diagnostic for the qualitative temporal model:

```bash
MAX_EVAL_JOBS=24 bash thesis/scripts/launch_temporal_qformer_v8_fresh_lora_repeat_current_full_eval.sh
```

Inference profiling for the computational-cost table uses route `3373` only
and runs the released baseline, Q-former v8 fresh-LoRA, and Delta v5 fresh-LoRA
under matched route/seed conditions. It is separate from the full
benchmark manager and enables profiling only for these jobs.

```bash
bash thesis/scripts/launch_inference_profiling_route3373.sh
```

After the three jobs finish, run the summarizer command printed by the launcher,
for example:

```bash
python thesis/analysis/summarize_inference_profiles.py \
  --root "${BASE_DIR}/eval_results/Bench2Drive/inference_profiles/profile_json/<run_tag>" \
  --out thesis/results/inference_profile_route3373_<run_tag>.txt
```

`start_eval_simlingo.py` skips routes with completed result files, so a crashed
controller can be resumed with the same launcher after active route jobs finish.
Do not merge/analyze before online evaluation finishes.

After all routes finish, merge metrics and use the scenario-comparison script
for thesis evidence:

```bash
python thesis/analysis/merge_bench2drive_results.py \
  -b /proj/berzelius-2023-154/users/x_hugaf/eval_results/Bench2Drive/<eval_name>/bench2drive \
  > thesis/results/metrics_<eval_name>.txt

python thesis/analysis/compare_bench2drive_scenarios.py \
  --baseline /proj/berzelius-2023-154/users/x_hugaf/eval_results/Bench2Drive/<baseline_eval_name>/bench2drive \
  --candidate /proj/berzelius-2023-154/users/x_hugaf/eval_results/Bench2Drive/<candidate_eval_name>/bench2drive \
  --baseline-name <baseline_label> \
  --candidate-name <candidate_label> \
  -o thesis/results/scenario_comparison_<baseline_label>_vs_<candidate_label>.txt
```

Analysis expects full benchmark result files, not selected-route rendering JSONs.

## 9. Selected Renders

Selected videos are generated through `thesis/rendering/submit_render_jobs.py`.
Use the wrapper for the final thesis render sets:

```bash
cd "${REPO_DIR}"
bash thesis/scripts/launch_selected_renders.sh --dry-run all
bash thesis/scripts/launch_selected_renders.sh all
```

Useful targets are `main`, `baseline`, `qformer-fresh`, `delta`, `diagnostics`,
`loaded`, `no-temporal`, `repeat-current`, and `all`. The manifests are under
`thesis/rendering/manifests/`, and render outputs are written under:

```bash
${BASE_DIR}/eval_results/Bench2Drive/renders/<agent_name>/bench2drive/<seed>/
```

The render submitter reuses completed result JSONs and multiview frames when
available. If frames are complete, the job only stitches the video on the CPU; if
frames are missing or incomplete, it reruns the simulator route.

## 10. Final Artifacts

Committed full-evaluation summaries:

| Experiment | Driving score | Success | Role |
| --- | --- | --- | --- |
| Corrected SimLingo baseline | `86.61 +/- 1.02` | `68.18% +/- 0.98%` | Final single-frame baseline |
| No-temporal v8 fresh LoRA | `86.16 +/- 0.61` | `67.58% +/- 0.86%` | Control: updated supervision without temporal tokens |
| Q-former v8 fresh LoRA | `87.06 +/- 0.14` | `69.85% +/- 0.21%` | Query-based temporal adapter |
| Q-former v8 repeated-current | `86.06 +/- 0.53` | `66.67% +/- 1.30%` | Diagnostic: temporal adapter without real history change |
| Q-former v8 loaded LoRA | `87.85 +/- 0.58` | `71.97% +/- 1.13%` | Diagnostic: released LoRA initialization plus stronger dynamic weighting |
| Delta feature v5 fresh LoRA | `87.84 +/- 0.73` | `71.67% +/- 0.57%` | Feature-delta temporal adapter |

The runbook intentionally does not duplicate the method discussion in the report.
For configuration details, use the Slurm launchers above and the Hydra experiment
files in `simlingo_training/config/experiment/`.

Committed result and audit files:

- `thesis/results/metrics_baseline_evalfix_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_fresh_lora_final_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_fresh_lora_no_temporal_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_fresh_lora_repeat_current_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_loaded_lora_full.txt`
- `thesis/results/metrics_temporal_delta_feature_v5_fresh_lora_full.txt`
- `thesis/results/scenario_comparison_baseline_evalfix_vs_qformer_v8_fresh.txt`
- `thesis/results/scenario_comparison_baseline_evalfix_vs_delta_v5_fresh.txt`
- `thesis/results/inference_profile_route3373_20260614_020221.txt`
- `thesis/results/temporal_dataloader_audit/temporal_dataloader_audit_20260509_210739_summary.txt`
- `thesis/results/interaction_supervision_audit_scene_facts_v3_yield_review/interaction_supervision_audit_20260524_155732_summary.txt`

The only incomplete corrected-baseline route records are route index `111`, route
`11715`, seeds `2` and `3`; these are documented as CARLA instability and are
counted conservatively as failed baseline attempts.

Check jobs and GPU hours when reproducing runs:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```
