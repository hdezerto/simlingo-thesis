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
```

Historical/provenance evaluation:

```bash
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_temporal_qformer_v6_epoch013_full_eval.sh
```

`start_eval_simlingo.py` skips routes with completed result files, so a crashed
controller can be resumed with the same launcher after active route jobs finish.
Do not merge/analyze before online evaluation finishes.

After all routes finish:

```bash
export EVAL_RUN_NAME="<unique_eval_name>"
source thesis/env.sh

python thesis/analysis/merge_bench2drive_results.py -b "${BENCH2DRIVE_ROOT}"

python thesis/analysis/analyze_scenario_failures.py \
  -b "${BENCH2DRIVE_ROOT}" \
  -o thesis/results/scenario_failure_report_<unique_eval_name>.txt
```

Analysis expects full benchmark result files, not selected-route rendering JSONs.

## 9. Render Selected Failure Cases

Rendering uses:

- `team_code/agent_simlingo.py`
- `thesis/rendering/submit_render_jobs.py`

Current thesis-relevant selected-route manifests:

- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_loaded_lora_final_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v8_fresh_lora_final_selected.json`

Older provenance manifests:

- `thesis/rendering/manifests/render_manifest_temporal_delta_feature_v4_final_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v6_epoch013_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v7_final_selected.json`

For new runs, create a manifest with:

- a unique `agent` name, so outputs do not overwrite older renders
- the correct checkpoint directory or fp32 checkpoint in `checkpoint`
- the same selected route cases when comparing versions
- `debug_viz=true` and `debug_save_language=true` when videos/commentary are needed

Dry run:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py \
  --manifest thesis/rendering/manifests/<manifest>.json \
  --dry-run
```

Submit:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py \
  --manifest thesis/rendering/manifests/<manifest>.json
```

Monitor:

```bash
squeue -u "$USERNAME"
```

Render outputs are written under:

```bash
${EVAL_OUT_ROOT}/<agent_name>/bench2drive/<seed>/
```

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

| Experiment | Main evidence | Thesis status |
| --- | --- | --- |
| Corrected SimLingo baseline | Full eval: driving score `86.61 +/- 1.02`, success `68.18% +/- 0.98%` | Final baseline for metric comparison |
| Q-former v8 fresh LoRA | Full eval: driving score `87.06 +/- 0.14`, success `69.85% +/- 0.21%`; selected renders show cleaner commentary than older runs, with no `black car` hits and only rare stale junction-clear phrases; behavior remains cautious in dense left-turn traffic streams, with several minor rear-contact scratches except `11755` where side visibility is likely limiting | Main current thesis result |
| Q-former v8 loaded LoRA | Selected renders weaker than v8 fresh, including a frontal collision on `4183`; `dynamic_sample_weight=3.0` | Diagnostic for loaded-vs-fresh LoRA, not the main result |
| No-temporal v8 fresh-LoRA ablation | Training running as job `16682954`: corrected supervision, fresh LoRA, `hist_len=1`, no temporal tokens, no auxiliary temporal motion loss, `dynamic_sample_weight=2.0` | Main ablation; wait for training, then evaluate/render |
| Delta feature v5 fresh LoRA | Training running as job `16682900`: corrected v8 supervision, fresh LoRA, `hist_len=5`, delta adapter, `dynamic_sample_weight=2.0` | Fair Q-former-vs-delta comparison; wait for training, then run selected renders |
| Corrected history ablations | Pending: real history vs repeated-current history vs zero history on the corrected v8 model | Needed before making a final causal claim that the temporal module uses history effectively |
| Archived old-supervision/exploratory runs | Q-former v1-v7, gate/history diagnostics, and Delta v1-v4 were moved to archive or kept for reference | Do not use as final thesis comparisons, except to explain method evolution if needed |

Current interpretation:

- The corrected baseline and Q-former v8 fresh LoRA are the current main metric comparison.
- Q-former v8 fresh LoRA is preferred over Q-former v6 for the thesis because its supervision story is cleaner and easier to justify, even though v6 had a slightly higher exploratory aggregate score.
- The no-temporal ablation is needed to separate temporal-context effects from corrected supervision, fresh LoRA, and dynamic sample weighting.
- Delta v5 is needed for a fair Q-former-vs-delta comparison under the same corrected-supervision recipe.
- Final claims should wait for the no-temporal ablation, Delta v5 selected renders, and corrected history ablations if they finish in time.





----------------------------------------


## 13. To Do

Current status:

- Main completed comparison: corrected baseline (`86.61 +/- 1.02`, success `68.18% +/- 0.98%`) vs Q-former v8 fresh LoRA (`87.06 +/- 0.14`, success `69.85% +/- 0.21%`).
- Running jobs: no-temporal v8 fresh-LoRA ablation (`16682954`) and Delta feature v5 fresh-LoRA (`16682900`).
- Q-former v8 loaded-vs-fresh selected-render comparison is complete; old-supervision delta/gate/history runs are archived for reference only.

Next experiment steps:

- Let jobs `16682954` and `16682900` finish.
- For no-temporal v8: evaluate/render enough to decide whether v8 gains come from temporal context or mainly from corrected supervision/fresh LoRA.
- For Delta v5: render selected routes `11755`, `3936`, `4183`, `4468`, and `4683`; run full Bench2Drive only if it looks competitive or a complete Q-former-vs-delta table is required.
- Run corrected history ablations for Q-former v8 fresh: real history vs repeated-current history vs zero history.

Thesis-writing steps while jobs run:

- Finish/polish introduction and background.
- Draft baseline/motivation, method, and experimental setup.
- Prepare result-table templates for baseline, Q-former v8 fresh, no-temporal v8, and Delta v5.
- Draft discussion/limitations only after the ablation and Delta results are known.



Check jobs and GPU hours:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```
