# SimLingo Temporal Training on Berzelius

Concise runbook for the temporal SimLingo experiments on Berzelius.

This document is the main notes file for `temporal-module`.

## 1. Branches

- `temporal-module`: active temporal branch. It contains the Q-former experiments, integrated delta-feature experiments, scratch dataset staging, audits, rendering/inference fixes, and thesis training notes.

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
/home/x_hugaf/.conda/envs/simlingo/bin/python -m pip install flash-attn --no-build-isolation
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

Current relevant training launchers:

```bash
thesis/slurm/train_temporal_qformer_v8_fresh_lora.slurm              # completed main Q-former run: corrected supervision + fresh LLM LoRA
thesis/slurm/train_temporal_qformer_v8_loaded_lora.slurm             # completed diagnostic: corrected supervision + loaded SimLingo LoRA
thesis/slurm/train_temporal_qformer_v8_fresh_lora_no_temporal.slurm  # prepared main ablation: corrected supervision + fresh LLM LoRA + no temporal module
thesis/slurm/train_temporal_delta_feature_v5_fresh_lora.slurm        # prepared next run: corrected supervision + fresh LLM LoRA + delta adapter
```

Older Q-former, gate, and delta launchers are kept for provenance but should not be treated as final thesis comparisons because they used earlier supervision code or exploratory settings.

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

Repeat-current full-evaluation ablation for the qualitative temporal model:

```bash
MAX_EVAL_JOBS=24 bash thesis/scripts/launch_temporal_qformer_v8_fresh_lora_repeat_current_full_eval.sh
```

Inference profiling for the Chapter 6 computational-cost table uses route `3373`
only and runs the released baseline, Q-former v8 fresh-LoRA, and Delta v5
fresh-LoRA under matched route/seed conditions. It is separate from the full
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

## 9. Render Selected Failure Cases

Rendering uses `team_code/agent_simlingo.py` through `thesis/rendering/submit_render_jobs.py`.

Chapter 3 render groups:

- Main temporal set: `3936`, `4183`, `4468`, `4683`.
  Focused signalized-left-turn cases where visible motion changes the decision.
- Boundary dynamic case: `11755`.
  Actor-flow case; useful, but hard to fix with front-camera history alone.
- Limitation case: `3373`.
  Emergency-vehicle / limited-view case; not a main temporal-success target.

Current render manifests:

- `thesis/rendering/manifests/render_manifest_baseline_evalfix_diagnostic_and_result_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_diagnostic_and_result_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_no_temporal_diagnostic_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_delta_feature_v5_fresh_lora_diagnostic_and_result_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_loaded_lora_diagnostic_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_repeat_current_diagnostic_selected.json`

Old selected-render outputs are archived at
`${BASE_DIR}/eval_results/archive/Bench2Drive_selected_old_2026_05_31/`.

Use one launcher for selected renders:

```bash
cd "${REPO_DIR}"
bash thesis/scripts/launch_selected_renders.sh --dry-run all
bash thesis/scripts/launch_selected_renders.sh all
```

Useful targets: `main` for the baseline, Q-former v8 fresh, and Delta v5
diagnostic/result-selected renders; `diagnostics` for loaded-LoRA and
no-temporal diagnostic renders; and `repeat-current` for the
real-history diagnostic. Individual targets are also available: `baseline`,
`qformer-fresh`, `delta`, `loaded`, and `no-temporal`.

The selected-render outputs are written under
`${BASE_DIR}/eval_results/Bench2Drive/renders/<agent_name>/`. The matching
real-history outputs for `repeat-current` are produced by the Q-former
diagnostic/result-selected manifest under
`${BASE_DIR}/eval_results/Bench2Drive/renders/simlingo_temporal_qformer_v8_fresh_lora_diagnostic_and_result_selected/`.

The render submitter reuses completed result JSONs and multiview frames when
available. If a route has complete frames, the job only stitches the video on the
CPU; if frames are missing or incomplete, it reruns the simulator route.

Monitor:

```bash
squeue -u "$USERNAME"
```

Render outputs are written under `${BASE_DIR}/eval_results/Bench2Drive/renders/<agent_name>/bench2drive/<seed>/`.

------------


## 10. Temporal Implementation Summary

Temporal input and injection:

- Training history is old-to-new; the last frame is current. Main temporal runs use `hist_len=5` and `history_stride=1`, so frames are spaced by the dataset interval of `5` CARLA ticks. The no-temporal ablation uses `hist_len=1`.
- CARLA inference encodes each current frame once, stores it in `frame_feature_buffer`, and samples history at the training spacing. Override only with `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`.
- Token layout:

```text
<img>
<TEMP_CONTEXT> x num_temporal_tokens
<IMG_CONTEXT>  x num_image_tokens
</img>
text prompt...
```

`<IMG_CONTEXT>` receives current-frame InternVL tokens. `<TEMP_CONTEXT>` receives temporal encoder outputs.

Temporal methods:

| Method | Core idea | Status |
| --- | --- | --- |
| Q-former | Learned query tokens attend over temporal visual memory; final v8 runs use delta-enriched memory from past features, signed deltas, and absolute deltas | Main completed thesis method: v8 fresh LoRA. v6 remains useful provenance but is not the final method because it used older supervision |
| Delta feature | Encode current-vs-past feature deltas into compact motion tokens | Delta v5 is prepared as the fair corrected-supervision comparison; older delta results are provenance |

Common temporal training setup:

- Load SimLingo `epoch=013`; freeze vision backbone and waypoint input encoder; train temporal modules, adaptors, and LLM LoRA.
- Keep current-frame image tokens; temporal modules add history tokens instead of replacing the image.
- Put temporal tokens inside `<img>...</img>` and use the mild prompt: `Consider nearby traffic motion and whether the ego path is clear.`
- Add box-derived motion commentary, an auxiliary temporal motion head, and dynamic sample weighting.
- The no-temporal ablation keeps corrected commentary/motion-description supervision and dynamic sample weighting, but has no temporal tokens or auxiliary temporal motion head.

### Temporal Motion Head

Implemented in `simlingo_training/models/driving.py`. It is training-only and gives temporal tokens a direct motion signal:

```python
temporal_pooled = temporal_embeds.mean(dim=1)      # [B, D]
logits = temporal_motion_head(temporal_pooled)    # [B, 4]
```

`D` is the LLM hidden size. The head is:

```text
LayerNorm(D) -> Linear(D, 256) -> GELU -> Linear(256, 4)
```

Labels are multi-label:

| Label | Meaning |
| --- | --- |
| `0` | Moving actor near the ego future path, mostly front/path region |
| `1` | Moving side/cross actor near the ego future path |
| `2` | Stopped actor near or blocking the ego future path |
| `3` | Strict dynamic-yield interaction |

The loss is masked BCE-with-logits, weighted by `aux_loss_weight=0.5`. It does not replace the main language/route/waypoint losses. `dynamic_sample_weight` is separate and upweights main losses when `actor_motion_labels[3]=1`; v8 uses `3.0` for loaded LoRA and `2.0` for fresh LoRA. The no-temporal v8 ablation keeps `dynamic_sample_weight=2.0` but disables this auxiliary temporal motion loss.

### Supervision Versions

Completed v3/v6 results used strict interaction cleanup based mainly on boxes and future waypoints. It helped selected Q-former renders, but stale language still leaked through: false `black car` / `stay behind`, false traffic-light reasons, and `vehicles are stopped at the junction` / `junction is clear`.

Current v8 supervision uses shared scene facts from boxes, current measurements, future measurements, and future expert waypoints:

- `expert_slows_or_waits` is a behavior proxy from waypoints and planner metadata; it does not itself mean yielding.
- `actor_motion_labels[3]=1` only when there is a strict moving-actor conflict, the expert/planner slows or waits, and the stop is not explained by red/yellow light, stop sign, construction, or ordinary lead-vehicle following.
- Labels no longer depend on cleaned commentary text.
- Cleanup removes stale junction-clear/stopped claims, false traffic-light color claims, unsupported following/`black car` claims, and green-light go/speed-up instructions contradicted by interaction facts.
- Strict dynamic-yield commentary appends: `A moving actor is close to the ego path, so the ego should yield until the path is clear.`

Valid red-light stops and valid lead-following text are kept. Green-light interaction cases intentionally emphasize path clearance rather than reinforcing the old `green => accelerate` shortcut.

Inference fixes:

- `team_code/agent_simlingo.py` sets `self.model.eval()` after loading checkpoints.
- Language generation uses greedy decoding with `temperature=0.0`.
- CARLA can still vary due to actor spawning and simulator state, so renders with spawn warnings are weak evidence.

## 11. Audits

This section tracks checks that validate the data and supervision pipeline. It
should not contain model comparisons, ablations, or qualitative behavior
analysis.

| Audit | Evidence | Thesis use |
| --- | --- | --- |
| Temporal dataloader wiring | Audit: `40` driving + `40` dreamer samples, `0` failures | Confirms history frames and temporal inputs are loaded correctly |
| v8 scene-fact supervision | `scene_facts_v3_yield_review`: `200` candidates, `109` accepted yield, `68` rule/static rejects, `23` lead-following rejects, `0` other rejects; no final stale junction, green-light, or speed-up phrases | Supports using v8 supervision as the final corrected-supervision recipe |

Older prompt-only, no-CoT, v3/v4 temporal-signal, and old-supervision history
ablations are kept for reference only. They should not be used as final thesis
evidence because they were run on superseded supervision or exploratory configs.

## 12. Experiment Status And Thesis Evidence

Current full-evaluation results:

| Experiment | Driving score | Success | Thesis use |
| --- | --- | --- | --- |
| Corrected SimLingo baseline | `86.61 +/- 1.02` | `68.18% +/- 0.98%` | Final single-frame baseline |
| No-temporal v8 fresh LoRA | `86.16 +/- 0.61` | `67.58% +/- 0.86%` | Main ablation: corrected supervision without temporal tokens |
| Q-former v8 fresh LoRA | `87.06 +/- 0.14` | `69.85% +/- 0.21%` | Query-based temporal adapter |
| Delta feature v5 fresh LoRA | `87.84 +/- 0.73` | `71.67% +/- 0.57%` | Current quantitative leader |

Current interpretation:

- Delta feature v5 fresh LoRA is the current quantitative leader.
- Q-former v8 fresh has the clearest qualitative temporal-interaction behavior
  on the selected render set.
- Use Q-former v8 fresh for the repeat-current full-evaluation ablation.
- No-temporal v8 is the main temporal-module-off ablation.
- Q-former v8 loaded LoRA remains a qualitative diagnostic only.
- Delta v5 full evaluation is complete across all three seeds.

Current running work:

- No render jobs are currently queued.
- Selected renders are complete.
- No new training jobs are needed for the main thesis story.
- Q-former repeat-current full evaluation is prepared but not launched.
- The only incomplete full-evaluation records are the corrected-baseline route
  index `111`, route `11715`, seeds `2` and `3`; these are documented as CARLA
  instability and counted conservatively as failed baseline attempts.

Main result files for the thesis text:

- `thesis/results/metrics_baseline_evalfix_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_fresh_lora_final_full.txt`
- `thesis/results/metrics_temporal_qformer_v8_fresh_lora_no_temporal_full.txt`
- `thesis/results/metrics_temporal_delta_feature_v5_fresh_lora_full.txt`
- `thesis/results/scenario_comparison_baseline_evalfix_vs_qformer_v8_fresh.txt`
- `thesis/results/scenario_comparison_baseline_evalfix_vs_delta_v5_fresh.txt`
- `thesis/results/temporal_dataloader_audit/temporal_dataloader_audit_20260509_210739_summary.txt`
- `thesis/results/interaction_supervision_audit_scene_facts_v3_yield_review/interaction_supervision_audit_20260524_155732_summary.txt`


## 13. To Do

Immediate experiment tasks:

- Launch Q-former repeat-current full evaluation if benchmark-level temporal
  history ablation evidence is desired.
- Add Delta v5 to the thesis benchmark table and discuss why it outperforms the
  query-based adapter on aggregate metrics.
- Keep scenario-level summaries for both Q-former and Delta concise: Delta
  explains aggregate gains, while Q-former supports the qualitative temporal
  analysis.
Writing focus now:

- Finish Chapter 3 with corrected-baseline metrics and the Chapter 3 render set.
- Finish Chapters 4 and 5 using the completed selected-render set.
- Draft Chapter 6 with Delta v5 metrics and placeholders for final render
  frames.
- Report quantitative results first; select the final best temporal model after
  the qualitative review.
- Use no-temporal as the temporal-module-off ablation and repeat-current as the
  real-history usage diagnostic.

Check jobs and GPU hours:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```
