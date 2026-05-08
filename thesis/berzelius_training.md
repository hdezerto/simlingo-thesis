# SimLingo Temporal Training on Berzelius

Concise runbook for the temporal SimLingo experiments on Berzelius.

This document is the canonical notes file on `temporal-module`.

## 1. Branches

- `temporal-module`: main temporal branch. It contains the Q-former experiments, the integrated delta-feature experiment, scratch dataset staging, and rendering/inference fixes.
- `temporal-delta-feature`: old development branch for the DeltaFlow-inspired feature-delta experiment. It can be kept as history until the integrated `temporal-module` version is verified.

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
- `BENCH2DRIVE_ROOT=${EVAL_OUT_ROOT}/${EVAL_RUN_NAME}/bench2drive`
- `MODEL_CKPT=${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- `SLURM_ACCOUNT=berzelius-2025-435`

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

Shared hardware request for full temporal runs:

- `8` A100 80GB GPUs
- `64` CPU cores
- `900G` RAM
- `3` days walltime

Create logs:

```bash
mkdir -p "${LOG_ROOT}/training"
```

### Q-Former v2

- Config: `simlingo_training/config/experiment/temporal_qformer_v2.yaml`
- Launcher: `thesis/slurm/train_temporal.slurm`
- Output: `outputs/2026_04_25_17_38_40_temporal_qformer_v2_8gpu`
- Status: completed `14` epochs.

Submit:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_temporal.slurm
```

### Q-Former v3 Gate 0

- Config: `simlingo_training/config/experiment/temporal_qformer_v3_gate0.yaml`
- Launcher: `thesis/slurm/train_temporal_v3_gate0.slurm`
- Output: `outputs/2026_04_27_02_25_36_temporal_qformer_v3_gate0_8gpu`
- Difference from v2: `temporal_model.gate_init=0.0`.

Submit:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_temporal_v3_gate0.slurm
```

Monitor any training job:

```bash
squeue -u "$USERNAME"
tail -f "${LOG_ROOT}/training/<jobid>.out"
tail -f "${LOG_ROOT}/training/<jobid>.err"
```

## 7. Checkpoints

DeepSpeed checkpoints are directories:

- `outputs/.../checkpoints/epoch=007.ckpt`
- `outputs/.../checkpoints/last.ckpt`

These are training-resume checkpoints, not plain `torch.load(...)` files.

Convert to fp32 for many inference/render jobs:

```bash
cd "${REPO_DIR}"
mkdir -p /tmp/${USER}_triton

RUN_DIR="outputs/<run_name>"
EPOCH="epoch=013"

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

Workflow:

1. run the model online on all Bench2Drive routes
2. merge the per-route JSON files
3. run scenario-failure analysis
4. render selected cases if needed

Do not merge/analyze before online evaluation finishes.

Run full evaluation from tmux:

```bash
export MODEL_CKPT="${REPO_DIR}/outputs/<run_name>/checkpoints/<checkpoint>"
export EVAL_RUN_NAME="<unique_eval_name>"
source thesis/env.sh
echo "8" > max_num_jobs.txt
python start_eval_simlingo.py
```

Important:

- always use a unique `EVAL_RUN_NAME`
- default `simlingo` can collide with baseline results
- `start_eval_simlingo.py` skips routes with completed result files

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

Useful manifests:

- `thesis/rendering/manifests/render_manifest_temporal_v2_8gpu_epoch013_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_v3_gate0_8gpu_epoch011_selected.json`

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

## 10. Temporal Implementation Summary

Temporal input:

- Training frames are ordered old-to-new; the last frame is current.
- Dataset frames are saved every `5` CARLA ticks.
- Online CARLA inference calls the agent every simulator tick, so the model predicts control every tick.
- The temporal window is sampled from the feature buffer at the training spacing.
- With `history_stride=1`, the history spacing is `1 * 5 = 5` simulator ticks.
- Example: at simulator tick `100`, `hist_len=5` uses ticks `80, 85, 90, 95, 100`.
- Override temporal history spacing only if needed with `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`; this does not change how often CARLA calls the agent.
- Online inference stores encoded InternVL features in `frame_feature_buffer`, so each current frame is encoded once and reused as history later.

Temporal methods:

| Method | Config target | Temporal information |
| --- | --- | --- |
| Q-former | `simlingo_training.models.temporal.qformer.TemporalQFormer` | Past-frame InternVL features compressed into `<TEMP_CONTEXT>` tokens |
| Delta feature | `simlingo_training.models.temporal.delta_feature.TemporalDeltaFeatureEncoder` | Current-vs-past InternVL feature differences pooled into `<TEMP_CONTEXT>` tokens |

Delta feature flow:

```text
current InternVL tokens: [B, 512, 896]
past InternVL tokens:    [B, 4, 512, 896]
signed_delta = current - past                       -> [B, 4, 512, 896]
absolute_delta = abs(current - past)                -> [B, 4, 512, 896]
concat signed + absolute delta                      -> [B, 4, 512, 1792]
weighted average over time with delta_decay=0.9     -> [B, 512, 1792]
delta projection                                    -> [B, 512, 896]
pool 512 visual tokens -> 64 temporal tokens        -> [B, 64, 896]
token projection + output norm + optional gate      -> [B, 64, 896]
```

Evaluation behavior:

- Normal image tokens receive the current frame.
- During training, driving heads condition on ground-truth assistant text.
- During evaluation, the model first generates assistant text, then predicts driving from that generated text.
- Wrong generated commentary can therefore hurt waypoint/control prediction.
- Assistant text is commentary plus `Waypoints:`, direct `Waypoints:`, or a QA answer depending on the sample type.

Inference fixes:

- `team_code/agent_simlingo.py` now calls `self.model.eval()` after loading the checkpoint, so dropout is disabled during rendering/evaluation.
- `torch.no_grad()` disables gradients but does not disable dropout.
- Language generation uses greedy decoding with `temperature=0.0`, so the model text generation itself is deterministic.
- CARLA closed-loop runs can still vary because actor spawning, simulator state, and small trajectory changes can alter the scenario.
- Treat old renders made before the `eval()` fix, or renders with actor-spawn warnings, as weak evidence.

## 11. Experiment Evidence

| Experiment | Setup | Main Evidence | Interpretation |
| --- | --- | --- | --- |
| v1 Q-former | `hist_len=3`, `queries=8`, `gate_init=-2.0`, `batch=6`, `epochs=5`; only temporal Q-former and InternVL `mlp1` trainable | Wiring test completed | Too small/short to fix motion failures |
| v2 Q-former | `hist_len=5`, `queries=16`, `gate_init=-2.0`, `batch=12`, `epochs=14`; temporal Q-former, LLM LoRA, InternVL `mlp1` trainable | Final gate weak, sigmoid about `0.167`; selected final renders still collided on `3936`, `4183`, `4468`, `4683` | Temporal influence likely too weak |
| v3 Q-former gate 0 | Same as v2, but `gate_init=0.0` | Full `epoch=011` eval: driving score `86.09 +/- 0.70`, success rate `67.42% +/- 0.77%`; gate stayed about `0.503`; current route `4683` diagnostic did not show robust improvement | Stronger initial gate helped some cases, but not consistently |
| Delta feature v1 | `hist_len=5`, `queries=64`, `gate_init=0.0`, `include_absolute_delta=true`, `delta_decay=0.9`, `batch=12`, `epochs=14` | Trained through `epoch=013`; selected renders not clearly better than v3; still produced problematic stopped-vehicle commentary on `4683` | Useful thesis ablation, but not a clear fix |

## 12. Diagnostic Tests And Results

These tests are small controlled renders, not full benchmark results.

### Diagnostic Questions

1. Does temporal signal exist in the encoded visual features?
2. Do temporal tokens change the closed-loop rollout?
3. Is generated commentary conditioning the main bottleneck?
4. Can a motion-aware prompt make the model use temporal context better?

### 1. Temporal Signal Exists

Purpose: check whether past-frame and delta features contain visible motion
information, especially around moving vehicles.

Current tests:

- Q-former v3 `epoch=011`, route `4683`: real history vs repeated-current history vs zero temporal tokens
- Delta feature `epoch=013`, route `4683`: same test, plus feature-delta heatmaps

Representative result at frame `185`, near the dynamic interaction:

| Method | Input feature delta mean | Real vs repeat-current L2 | Real vs repeat-current cosine | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Q-former v3 | `11.93` | `2.00` | `0.986` | input changes exist, but Q-former tokens remain very similar to fake-history tokens |
| Delta feature v1 | `21.90` | `19.81` | `-0.283` | delta tokens change strongly when real history is used |

Delta heatmaps saved for frame `185`:

- `00185_feature_delta_mean.jpg`: unweighted mean of current-vs-past feature-delta magnitudes across the history window
- `00185_feature_delta_latest.jpg`: current-vs-closest-past feature-delta magnitude

Interpretation:

- Q-former v3 appears to compress real history into tokens close to repeated-current history.
- Delta feature v1 produces motion-sensitive temporal tokens, but route `4683` still collided.
- This suggests the remaining bottleneck is likely downstream of the temporal encoder, or in the supervision/labels, not simply absence of temporal signal.

### 2. Temporal History Changes The Rollout

Route `4683`, v3 `epoch=011`:

| Mode | Result |
| --- | --- |
| `real` | score `42`, `1` collision, scenario timeout |
| `repeat_current` | score `36`, `2` collisions |
| `zero` | score `60`, `1` collision |

Conclusion:

- v3 does not robustly fix route `4683`.
- Temporal history affects the rollout, but real history was not better than ablated history.
- The old selected-render success for route `4683` should not be treated as strong evidence because it used older inference code and had actor-spawn warnings.

### 3. Commentary Conditioning Bottleneck

Purpose: check whether generated commentary is the main reason driving fails.
The model still generates commentary for the rendered video, but the driving
heads do not condition on that generated text.

v3 `epoch=011`, selected routes:

| Route | Normal CoT action | No-CoT action | Interpretation |
| --- | --- | --- | --- |
| `11755` | score `60`, vehicle collision | score `60`, vehicle collision | no improvement |
| `3936` | score `60`, vehicle collision | score `49`, scenario timeout | worse |
| `4183` | score `60`, vehicle collision | score `60`, vehicle collision | no improvement |
| `4468` | score `60`, vehicle collision | score `60`, vehicle collision | no improvement |
| `4683` | score `100`, no collision | score `60`, vehicle collision | worse |

Conclusion:

- no-CoT action did not improve the selected failure cases
- generated commentary conditioning is probably not the main easy bottleneck
- keep normal commentary-conditioned driving for main evaluation
- use no-CoT only as a diagnostic, because it differs from training

### 4. Prompting Diagnostic

Purpose: test whether the temporal signal exists but the LLM needs clearer
instruction to reason about motion.

Status: not run yet.

Possible test:

- render the same selected routes with a motion-aware prompt
- compare generated commentary and collisions against the normal prompt
- treat this as a diagnostic, not as main evidence, because changing the prompt changes the evaluation distribution

### Remaining Hypotheses

- temporal tokens are noisy or not used in the right way
- motion supervision may be too weak or diluted
- labels may overuse stopped-vehicle wording
- Q-former compression may lose small moving vehicles
- waypoints/control may fail even when commentary improves

Best follow-ups:

- After v4 early checkpoints: repeat the real/repeat-current/zero diagnostic on route `4683`.
- If time allows, train a no-history control: same temporal module, but train with repeated current frames instead of real past frames.

## 13. Current Training Run (v4)

v4 Q-former is running:

- config: `simlingo_training/config/experiment/temporal_qformer_v4_nogate.yaml`
- launcher: `thesis/slurm/train_temporal_v4_nogate.slurm`
- keep `num_queries=16`
- set `temporal_model.gate_enabled=false`
- set `freeze_adaptors=false`
- keep `hist_len=5`, `history_stride=1`, `batch_size=12`, `max_epochs=14`

Why:

- ORION is the closest reference point and found `16` history queries better than `32`.
- v3 used `16` queries but the learned gate stayed around `0.503`, so disabling the gate directly tests whether it limited temporal influence.
- Unfreezing adaptors lets the language/driving bridges adapt to `<TEMP_CONTEXT>`.
- The main vision encoder, base LLM weights, and waypoint encoder remain frozen, so the run is stronger than v3 but still controlled.

Evaluate early:

- render selected failures from `epoch=005` or `epoch=007`
- include route `4683`
- compare real history against repeated-current and zero history if the route remains unstable
- keep normal commentary-conditioned driving for the main comparison
- use no-CoT/direct-driving only as a diagnostic, because the v3 selected no-CoT renders were not better

## 14. Temporary Resource Snapshot

Checked on `2026-05-07`:

```bash
projinfo -m berzelius-2025-435
```

- Monthly allocation: `5000 h/month`
- Project consumption since `2026-05-01`: `3625.55 h`
- User consumption since `2026-05-01`: `1741.07 h`
- Approximate project hours remaining: `1374.45 h`
- Active job at check: `16498473`, `simlingo_temporal_v4_nogate`, running on `8` GPUs
- Accounting may lag while jobs are running
