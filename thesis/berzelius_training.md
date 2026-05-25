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

Full temporal training jobs use archive-aware Slurm launchers under
`thesis/slurm/`. Version-specific experiment settings live in
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

Current temporal launchers:

```bash
thesis/slurm/train_temporal_qformer_v6.slurm              # strongest completed selected-route result
thesis/slurm/train_temporal_qformer_v8_loaded_lora.slurm  # next main run: new supervision + loaded SimLingo LoRA
thesis/slurm/train_temporal_qformer_v8_fresh_lora.slurm   # ablation: new supervision + fresh LLM LoRA
```

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

Current selected-route manifests:

- `thesis/rendering/manifests/render_manifest_temporal_delta_feature_v2_epoch013_selected.json`
- `thesis/rendering/manifests/render_manifest_temporal_qformer_v5_epoch013_selected.json`

For new runs, create a manifest with:

- a unique `agent` name, so outputs do not overwrite older renders
- the correct DeepSpeed checkpoint directory in `checkpoint`
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

## 10. Temporal Implementation Summary

Temporal input and injection:

- Training history is old-to-new; the last frame is current. With `hist_len=5` and `history_stride=1`, frames are spaced by the dataset interval of `5` CARLA ticks.
- CARLA inference encodes each current frame once, stores it in `frame_feature_buffer`, and samples history at the training spacing. Override only with `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`.
- Token layout:

```text
<img>
<TEMP_CONTEXT> x num_temporal_tokens
<IMG_CONTEXT>  x num_image_tokens
</img>
text prompt...
```

`<IMG_CONTEXT>` receives current-frame InternVL tokens. `<TEMP_CONTEXT>` receives temporal encoder outputs. Driving heads train with ground-truth assistant text and evaluate from generated assistant text.

Temporal methods:

| Method | Core idea | Status |
| --- | --- | --- |
| Q-former | Learned query tokens attend over temporal visual memory | Best completed method so far: v6. Next runs: v8 loaded/fresh LoRA with the same architecture and revised supervision |
| Delta feature | Encode current-vs-past feature deltas into compact motion tokens | Useful ablation, but less promising than Q-former on selected renders |

Common temporal training setup:

- Load SimLingo `epoch=013`; freeze vision backbone and waypoint input encoder; train temporal modules, adaptors, and LLM LoRA.
- Keep current-frame image tokens; temporal modules add history tokens instead of replacing the image.
- Put temporal tokens inside `<img>...</img>` and use the mild prompt: `Consider nearby traffic motion and whether the ego path is clear.`
- Add box-derived motion commentary, an auxiliary temporal motion head, and dynamic sample weighting.

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

The loss is masked BCE-with-logits, weighted by `aux_loss_weight=0.5`. It does not replace the main language/route/waypoint losses. `dynamic_sample_weight` is separate and upweights main losses when `actor_motion_labels[3]=1`; v8 uses `3.0` for loaded LoRA and `2.0` for fresh LoRA.

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

## 11. Diagnostic Tests And Results

These are controlled renders/audits, not full benchmark results.

| Question | Result | Interpretation |
| --- | --- | --- |
| Does temporal signal exist? | Route `4683`, frame `185`: Q-former v3 real-vs-repeat L2 `2.00`, cosine `0.986`; Delta v1 L2 `19.81`, cosine `-0.283` | Delta tokens were much more motion-sensitive than Q-former v3 tokens |
| Does temporal history change rollout? | v3 route `4683`: real score `42`, repeat-current `36`, zero `60` | Temporal history affects rollout, but real history was not reliably better |
| Is generated commentary the main bottleneck? | No-CoT action did not improve selected v3 failures; route `4683` got worse | Keep normal commentary-conditioned driving; no-CoT is diagnostic only |
| Can prompting alone fix it? | Strong motion prompt on v4 did not improve selected renders and produced OOD text | Prompt-only changes are risky unless trained with matching wording |
| Is the data pipeline sane? | Temporal dataloader audit: `40` driving + `40` dreamer samples, `0` failures | Dataloader wiring is not the main problem |
| Does v8 scene-fact supervision select useful cases? | `scene_facts_v3_yield_review`: `200` candidates, `109` accepted yield, `68` rule/static rejects, `23` lead-following rejects, `0` other rejects; no final stale junction, green-light, or speed-up phrases | Supervision is clean enough to launch Q-former v8 |

## 12. Experiment Evidence

| Experiment | Main evidence | Interpretation |
| --- | --- | --- |
| v1 Q-former | `hist_len=3`, `queries=8`, `epochs=5`; wiring test completed | Too small/short to fix motion failures |
| v2 Q-former | `hist_len=5`, `queries=16`; final gate sigmoid about `0.167`; selected renders still collided on `3936`, `4183`, `4468`, `4683` | Temporal influence likely too weak |
| v3 Q-former gate 0 | Full `epoch=011` eval: driving score `86.09 +/- 0.70`, success `67.42% +/- 0.77%`; route `4683` diagnostic not robust | Stronger gate helped some cases, not consistently |
| v4 Q-former no gate | `epoch=011` selected renders still collided on all five routes | No gate + trainable adaptors did not fix failures |
| Delta feature v1 | Trained through `epoch=013`; selected renders not clearly better; stale stopped-vehicle commentary on `4683` | Useful ablation, not a clear fix |
| Delta feature v2 | Completed `epoch=013`; route `3936` scored `100`, but `11755`, `4183`, `4468`, `4683` still collided | Useful intermediate run before strict cleanup |
| Q-former v5 | Completed `epoch=013`; explicit delta-aware memory; selected renders still collided on all five routes | More logical architecture, but supervision still weak |
| Delta feature v3 | Completed `epoch=013`; strict interaction cleanup. `epoch=009` improved `3936`, but `epoch=013` regressed | Strict cleanup did not make delta robust; longer training hurt selected behavior |
| Q-former v6 | Completed `epoch=013`; strict interaction cleanup. Final selected renders fixed `3936` and `4683`; `11755`, `4183`, `4468` still collided | Best temporal result so far, but still fragile |
| Q-former v7 | Fresh LoRA + previous scene-fact cleanup. Selected renders still showed degraded/stale commentary behavior | Fresh LoRA alone is risky; keep v6 as strongest completed result |
| Q-former v8 loaded LoRA planned | Same Q-former architecture as v6/v7, revised v8 supervision, loaded SimLingo LoRA, `dynamic_sample_weight=3.0` | Main next run: best chance to preserve driving skill while correcting stale interaction language |
| Q-former v8 fresh LoRA planned | Same as v8 loaded LoRA, but resets LLM LoRA and lowers `dynamic_sample_weight` to `2.0` | Ablation to test whether stale language lives in old LoRA weights without over-weighting rare yield samples |

Selected-route render results:

| Route | Scenario | Delta v2 `e13` | Q-former v5 `e13` | Delta v3 `e09` | Delta v3 `e13` | Q-former v6 `e09` | Q-former v6 `e13` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `11755` | `EnterActorFlow_1` | `60`, collision | `60`, collision | `60`, collision | `60`, collision | `60`, collision | `60`, collision |
| `3936` | `SignalizedJunctionLeftTurn_1` | `100`, no collision | `60`, collision | `100`, no collision | `60`, collision | `100`, no collision | `100`, no collision |
| `4183` | `SignalizedJunctionLeftTurn_1` | `60`, collision | `36`, two collisions | `60`, collision | `60`, collision | `60`, collision | `60`, collision |
| `4468` | `SignalizedJunctionLeftTurn_1` | `60`, collision | `60`, collision | `60`, collision | `60`, collision | `42`, collision + timeout | `60`, collision |
| `4683` | `SignalizedJunctionLeftTurn_1` | `60`, collision | `60`, collision | `60`, collision | `42`, collision + timeout | `42`, collision + timeout | `100`, no collision |

Interpretation:

- Q-former v6 `epoch=013` is the strongest current selected-route result: it keeps `3936` successful and fixes `4683` on the selected render set.
- Results are not robust enough to claim the problem is solved. Remaining renders still show false traffic-light claims, unsupported `black car` / `stay behind`, and stale `vehicles are stopped` / `junction is clear` commentary.
- `11755` may require side information that a front-camera-only model does not have, but the commentary should still avoid confidently claiming the path is clear.

## 13. Next Actions

Current status: Q-former v8 training has been launched.

| Run | Slurm job | Purpose |
| --- | --- | --- |
| `temporal_qformer_v8_loaded_lora_8gpu` | `16636097` | Main run: revised supervision, loaded SimLingo LoRA, `dynamic_sample_weight=3.0` |
| `temporal_qformer_v8_fresh_lora_8gpu` | `16636098` | Ablation: revised supervision, fresh LLM LoRA, `dynamic_sample_weight=2.0` |

While training runs:

1. Monitor the first hour of logs to confirm dataset staging completes, the checkpoint loads, and training starts without missing/unexpected key issues.
2. When `epoch=009` is available for each run, render the selected failure routes first.
3. Render the final checkpoint for both runs when training finishes.
4. Keep Q-former v6 `epoch=013` as the strongest completed result until v8 renders are evaluated.
5. Use the waiting time to write thesis sections that do not depend on v8 metrics: introduction, background, dataset/model description, temporal method design, supervision design, and experimental protocol.

Recommended thesis-writing focus now:

- Background: VLA driving, SimLingo, temporal context, Bench2Drive.
- Method: Q-former temporal adapter, temporal token insertion, motion head, and interaction-aware supervision.
- Experiments: training setup, compute/resources, selected-route render protocol, full-evaluation protocol.
- Results so far: v6 as strongest completed temporal result; v8 described as ongoing follow-up.


Check jobs and GPU hours:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```




To do in the future:
- Ablation for the temporal module
- Try multi-view