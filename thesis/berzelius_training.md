# SimLingo Temporal Training on Berzelius

Concise runbook for the temporal SimLingo experiments on Berzelius.

This document is the canonical notes file on `temporal-module`.

## 1. Branches

- `temporal-module`: main temporal branch. It contains the Q-former experiments, scratch dataset staging, and rendering/inference fixes.
- `temporal-delta-feature`: separate branch for the DeltaFlow-inspired feature-delta experiment. Keep the delta implementation there unless deliberately merging it later.

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

## 10. Temporal Implementation Facts

- Training frame order is old-to-new; the last frame is current.
- Dataset frames are saved every `5` CARLA ticks.
- With `history_stride=1`, inference samples every `1 * 5` simulator ticks.
- Override inference spacing only if needed with `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`.
- Normal image tokens receive the current frame.
- Q-former temporal tokens receive only past frames.
- The delta-feature temporal implementation is kept on the separate `temporal-delta-feature` branch.
- During training, driving heads condition on ground-truth assistant text.
- During evaluation, the model first generates assistant text, then predicts driving from that generated text.
- Therefore wrong generated commentary can hurt driving prediction.

Assistant text cases:

- commentary-generation samples: ground-truth driving commentary followed by `Waypoints:`
- direct-driving samples: only `Waypoints:`
- QA samples: ground-truth QA answer

## 11. Experiment Tracker

### v1 Q-Former

- Config: `simlingo_training/config/experiment/temporal_qformer_v1.yaml`
- Setup: `hist_len=3`, `history_stride=1`, `num_queries=8`, `gate_init=-2.0`, `batch_size=6`, `max_epochs=5`
- Trainable: temporal Q-former, InternVL `mlp1`
- Frozen: main vision encoder, full LLM, adaptors, waypoint-token encoder
- Outcome: useful wiring test, but too small/short to fix motion-reasoning failures.

### v2 Q-Former

- Config: `simlingo_training/config/experiment/temporal_qformer_v2.yaml`
- Job: `16437515`
- Output: `outputs/2026_04_25_17_38_40_temporal_qformer_v2_8gpu`
- Setup: `hist_len=5`, `history_stride=1`, `num_queries=16`, `num_layers=2`, `gate_init=-2.0`, `batch_size=12`, `max_epochs=14`, `val_every_n_epochs=2`
- Trainable: temporal Q-former, LLM LoRA adapters, InternVL `mlp1`
- Frozen: main vision encoder, base LLM weights, driving/language adaptors, waypoint-token encoder

Evidence:

- Final selected renders still showed wrong/stale moving-vehicle commentary in selected failures.
- v2 final gate stayed weak: sigmoid about `0.167` at `epoch=013`.
- v2 selected final renders still had collisions on routes including `3936`, `4183`, `4468`, and `4683`.

### v3 Q-Former Gate 0

- Config: `simlingo_training/config/experiment/temporal_qformer_v3_gate0.yaml`
- Job: `16443291`
- Output: `outputs/2026_04_27_02_25_36_temporal_qformer_v3_gate0_8gpu`
- Difference from v2: gate starts at sigmoid `0.5` instead of `0.119`.

Evidence so far:

- `epoch=011` improved route `4683` in selected rendering: score `100`, no collision.
- `epoch=013` did not keep that improvement on route `4683`; the selected render again collided.
- Full `epoch=011` evaluation:
  - metrics file: `thesis/results/metrics_temporal_v3_gate0_epoch011_eval.txt`
  - driving score: `86.09 +/- 0.70`
  - success rate: `67.42% +/- 0.77%`
- Gate values stayed almost unchanged:
  - `epoch=005`: raw `0.006931`, sigmoid `0.501733`
  - `epoch=011`: raw `0.012207`, sigmoid `0.503052`
  - `epoch=013`: raw `0.012276`, sigmoid `0.503069`
- Interpretation: stronger initial temporal influence helped some cases, but the model did not learn to open the gate much further and the improvement was not consistent.

### Delta Feature v1

- Branch: `temporal-delta-feature`
- Config: `simlingo_training/config/experiment/temporal_delta_feature_v1.yaml`
- Launcher: `thesis/slurm/train_temporal_delta_feature.slurm`
- Setup: `hist_len=5`, `history_stride=1`, `num_queries=64`, `gate_init=0.0`, `include_absolute_delta=true`, `delta_decay=0.9`, `batch_size=12`, `max_epochs=14`
- Rationale: inspired by DeltaFlow. Instead of summarizing past frames, it computes feature-space motion trails using current-vs-past InternVL deltas.

Delta feature flow:

```text
current InternVL tokens: [B, 256, D]
past InternVL tokens:    [B, 4, 256, D]

signed_delta = current - past
absolute_delta = abs(current - past)
weighted average over time with delta_decay=0.9
shared motion map: [B, 256, D]
pool 16x16 -> 8x8
64 temporal tokens
```

Why `64` tokens:

- InternVL2-1B gives `256` visual tokens, roughly a `16 x 16` grid.
- `64` temporal tokens preserve an `8 x 8` coarse motion grid.
- This is less compressed than `16` tokens and should preserve more small-vehicle motion detail.

Evidence:

- Training completed successfully through `epoch=013`.
- Selected render output: `simlingo_temporal_delta_feature_v1_8gpu_epoch013_render_selected`
- Rendered videos were produced for routes `3936`, `4183`, `4468`, and `4683`.
- Route `11755` failed technically because CARLA crashed with `Signal 11` and then timed out.
- Delta was not clearly better than Q-former v3:
  - matched v3 on `3936`
  - matched or improved over v3 `epoch=013` on `4468`
  - worse than v3 `epoch=011` on `4183` and `4683`
  - still generated the problematic "other vehicles are stopped" commentary on route `4683`

Interpretation:

- The delta representation is useful as an alternative thesis ablation, but the selected renders do not show a significant improvement over the Q-former branch.
- The remaining failure mode is likely not only temporal-token architecture. It is probably also related to supervision, data balance, and generated-commentary conditioning.

## 12. Current Resource Snapshot

Checked on `2026-05-04` with:

```bash
projinfo -m berzelius-2025-435
```

- Monthly allocation: `5000 h/month`
- Project consumption since `2026-05-01`: `1329.01 h`
- User `x_hugaf` consumption since `2026-05-01`: `1102.25 h`
- Approximate project hours remaining this month: `3670.99 h`

## 13. Recommended Next Run

Recommended next Q-former experiment:

- keep the v3 architecture and training recipe
- keep `num_queries=16`
- increase `temporal_model.gate_init` from `0.0` to `1.0`
- keep `hist_len=5`, `history_stride=1`, `batch_size=12`, `max_epochs=14`
- render the same selected failure routes early, for example at `epoch=005` or `epoch=007`

Why this run:

- ORION is the closest paper to this setting and found `16` history queries better than `32`.
- v3 already used `16` queries and showed one real improvement, but the gate stayed at about `0.503`, so the temporal branch may still be too weak.
- A stronger gate start, sigmoid about `0.731`, tests temporal influence without removing the stabilizing gate completely.
- Disabling the gate is a useful ablation later, but is riskier because it lets randomly initialized temporal tokens enter at full strength from the first step.

If this run still fails:

Do not only keep changing the temporal architecture. Diagnose the source:

- supervision/data: oversample junctions with moving vehicles, DriveLM moving-status QA, or explicit moving/stopped labels
- labels: improve commentary labels that describe crossing/moving vehicles as stopped
- conditioning: test direct waypoint prediction without generated commentary
- temporal usage: compare real history with repeated-current or shuffled history

Useful thesis ablations:

- LoRA ablation: train temporal Q-former with and without trainable LLM LoRA adapters.
- Parameter-matched no-temporal control: keep temporal module capacity but replace past frames with current-frame copies.
- Temporal-history diagnostic: compare real history with repeated-current or shuffled history during rendering.
- No-COT diagnostic: evaluate selected routes with direct waypoint prediction instead of generated commentary first.
