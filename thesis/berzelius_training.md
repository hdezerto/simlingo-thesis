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
thesis/slurm/train_temporal_delta_feature_v3.slurm
thesis/slurm/train_temporal_qformer_v6.slurm
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

Temporal input:

- Training frames are ordered old-to-new; the last frame is current.
- Dataset frames are saved every `5` CARLA ticks. With `hist_len=5` and `history_stride=1`, the model uses ticks like `80, 85, 90, 95, 100`.
- Online CARLA inference still runs every simulator tick. It samples the history from `frame_feature_buffer` at the training spacing, so each current frame is encoded once and reused later.
- Override history spacing only with `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`.

Temporal injection:

```text
<img>
<TEMP_CONTEXT> x num_temporal_tokens
<IMG_CONTEXT>  x num_image_tokens
</img>
text prompt...
```

`<IMG_CONTEXT>` receives current-frame InternVL tokens. `<TEMP_CONTEXT>`
receives temporal encoder outputs. During training, driving heads condition on
ground-truth assistant text; during evaluation, the model first generates
assistant text and then predicts driving from that generated text.

Temporal methods:

| Method | Config target | Output |
| --- | --- | --- |
| Q-former | `simlingo_training.models.temporal.qformer.TemporalQFormer` | Learned query tokens attend over temporal visual memory |
| Delta feature | `simlingo_training.models.temporal.delta_feature.TemporalDeltaFeatureEncoder` | Current-vs-past feature deltas are pooled into compact motion tokens |

Q-former v5 flow:

```text
current tokens: [B, 512, 896]
past tokens:    [B, 4, 512, 896]
concat past + (current-past) + abs(current-past) -> [B, 4, 512, 2688]
input projection + temporal positions           -> [B, 4, 512, 896]
flatten memory                                  -> [B, 2048, 896]
32 learned queries attend to memory for 2 layers -> [B, 32, 896]
```

Each Q-former layer has separate weights and applies query self-attention,
query-to-memory cross-attention, and an MLP update. In v5, `dropout=0.1`,
`num_heads=8`, `num_layers=2`, and the gate is disabled.

Delta feature v2 flow:

```text
current tokens: [B, 512, 896]
past tokens:    [B, 4, 512, 896]
concat (current-past) + abs(current-past)        -> [B, 4, 512, 1792]
decay-weighted average over time, delta_decay=0.9 -> [B, 512, 1792]
delta projection                                 -> [B, 512, 896]
spatial pool 2 x 16 x 16 token blocks            -> [B, 64, 896]
token projection + output norm                   -> [B, 64, 896]
```

Delta spatial pooling preserves rough image locality. If the token layout
cannot be inferred, the encoder warns once and falls back to 1D pooling.

Common temporal training changes:

- Recent runs load the SimLingo `epoch=013` checkpoint; vision backbone and waypoint input encoder are frozen, while temporal modules, adaptors, and LLM LoRA remain trainable.
- Temporal tokens are placed inside `<img>...</img>` to make the LLM treat them as visual evidence.
- Mild prompt: `Consider nearby traffic motion.`
- Commentary is enriched with box-derived motion facts.
- Auxiliary motion head predicts four box-derived labels from temporal tokens: moving front/path actor, moving lateral/cross actor, stopped blocking actor, and strict dynamic-yield interaction.
- `aux_loss_weight=0.5`; `dynamic_sample_weight=3.0` upweights strict dynamic-yield interaction samples for the main losses.
- Current-frame image tokens are kept; no image-token dropout in these runs.

Strict interaction supervision added after v2/v5:

- Removes stale commentary claims such as "vehicles are stopped at the junction", "junction is clear", and green-light "accelerate/drive through" instructions when boxes and expert waypoints indicate a moving actor conflict.
- Sets `actor_motion_labels[3]=1` only when a moving actor is close to the ego future path, expert future waypoints slow/yield, commentary is available, and the case is not explained by a red light, stop sign, construction, or front-car following.
- Adds yield text only for those strict interaction samples: `A moving actor is close to the ego path, so the ego should yield until the path is clear.`

Inference fixes:

- `team_code/agent_simlingo.py` calls `self.model.eval()` after loading checkpoints, so dropout is disabled during rendering/evaluation.
- Language generation uses greedy decoding with `temperature=0.0`.
- Closed-loop CARLA can still vary due to actor spawning and simulator state. Treat old renders before the `eval()` fix, or with actor-spawn warnings, as weak evidence.

## 11. Experiment Evidence

| Experiment | Main Evidence | Interpretation |
| --- | --- | --- |
| v1 Q-former | `hist_len=3`, `queries=8`, `epochs=5`; wiring test completed | Too small/short to fix motion failures |
| v2 Q-former | `hist_len=5`, `queries=16`; final gate sigmoid about `0.167`; selected final renders still collided on `3936`, `4183`, `4468`, `4683` | Temporal influence likely too weak |
| v3 Q-former gate 0 | Full `epoch=011` eval: driving score `86.09 +/- 0.70`, success `67.42% +/- 0.77%`; route `4683` diagnostic not robust | Stronger initial gate helped some cases, not consistently |
| v4 Q-former no gate | `epoch=011` selected renders still collided on all five routes: scores `60, 60, 60, 42, 42` | No gate + trainable adaptors did not fix motion failures |
| Delta feature v1 | Trained through `epoch=013`; selected renders not clearly better; still produced problematic stopped-vehicle commentary on `4683` | Useful ablation, not a clear fix |
| Delta feature v2 | Completed `epoch=013`; no gate, spatial pooling, motion prompt/descriptions, aux motion loss, dynamic weighting. Final selected renders: route `3936` scored `100`, but `11755`, `4183`, `4468`, and `4683` still scored `60` with vehicle collisions | Useful intermediate run, but trained before strict interaction-commentary cleanup; one selected-route success is not robust |
| Q-former v5 | Completed `epoch=013`; 32 queries over `[past, current-past, abs(current-past)]`, same supervision changes as delta v2. Final selected renders still collided on all five routes; route `4183` scored `36` with two vehicle collisions | Explicit deltas helped make the architecture more logical, but did not fix failures before strict cleanup |
| Delta feature v3 | Running as `simlingo_delta_v3_ckpt`; same delta v2 architecture, now with strict interaction-yield commentary cleanup and strict interaction sample weighting | Best current delta run |
| Q-former v6 | Running as `simlingo_qformer_v6_ckpt`; same Q-former v5 architecture, now with strict interaction-yield commentary cleanup and strict interaction sample weighting | Best current Q-former run |

Selected-render results at final v2/v5 checkpoints:

| Route | Scenario | Delta v2 `epoch=013` | Q-former v5 `epoch=013` |
| --- | --- | --- | --- |
| `11755` | `EnterActorFlow_1` | score `60`, `1` vehicle collision | score `60`, `1` vehicle collision |
| `3936` | `SignalizedJunctionLeftTurn_1` | score `100`, no collision | score `60`, `1` vehicle collision |
| `4183` | `SignalizedJunctionLeftTurn_1` | score `60`, `1` vehicle collision | score `36`, `2` vehicle collisions |
| `4468` | `SignalizedJunctionLeftTurn_1` | score `60`, `1` vehicle collision | score `60`, `1` vehicle collision |
| `4683` | `SignalizedJunctionLeftTurn_1` | score `60`, `1` vehicle collision | score `60`, `1` vehicle collision |

Interpretation:

- Delta v2 is more promising than Q-former v5 on the selected routes, but only route `3936` clearly improved.
- Q-former v5 did not fix the selected dynamic-interaction failures; `4183` was worse with two collisions.
- Saved commentary still shows stale green-light/junction reasoning. Example pattern: "Accelerate to drive through the junction because the other vehicles are stopped ... and the junction is clear" while the same answer also mentions a moving actor.
- The old OOD phrase "ignore instruction as it leads to a crash" did not appear in these final selected renders.
- These results support using v2/v5 as pre-strict-cleanup evidence and waiting for v3/v6 before making further architectural changes.

## 12. Diagnostic Tests And Results

These are controlled renders/audits, not full benchmark results.

| Question | Result | Interpretation |
| --- | --- | --- |
| Does temporal signal exist? | At route `4683`, frame `185`: Q-former v3 real-vs-repeat L2 `2.00`, cosine `0.986`; Delta v1 L2 `19.81`, cosine `-0.283` | Input changes exist; delta tokens are much more motion-sensitive than Q-former v3 tokens |
| Does temporal history change rollout? | v3 route `4683`: real score `42`, repeat-current `36`, zero `60` | Temporal tokens affect rollout, but real history was not reliably better |
| Is generated commentary conditioning the bottleneck? | No-CoT action on selected v3 routes did not improve failures; route `4683` got worse | Keep normal commentary-conditioned driving; no-CoT is diagnostic only |
| Can prompting alone fix it? | Strong motion prompt on v4 did not improve selected renders and produced OOD text such as "ignore instruction as it leads to a crash" | Prompt-only changes are risky unless trained with matching wording |
| Are data/supervision sane? | Temporal dataloader audit: `40` driving + `40` dreamer samples, `0` failures. Supervision audit: `300` samples, moving actors in `254/300`, same-lane moving actors in `155/300`, same-lane stopped actors in `49/300`, `18/300` heuristic flags | Data pipeline is not the main problem; labels have some ambiguity but no broad corruption |
| Does strict interaction cleanup select the right cases? | `interaction_supervision_audit_strict_v4`: `96` samples, all with `actor_motion_labels[3]=1`; all final commentaries contain yield; no final stale `accelerate`, `drive through`, `junction is clear`, `other vehicles are stopped`, or `moving away` phrases; `13` conflicting claims removed | Strict cleanup now targets the observed failure mode much better |

Delta diagnostic heatmaps:

- `weighted_signed_delta_norm`: decay-weighted signed motion trail.
- `weighted_abs_delta_norm`: decay-weighted absolute motion magnitude.
- `latest_delta_norm`: current-vs-closest-past feature-delta magnitude.
- `projected_motion_norm`: learned motion features after `delta_projection + motion_norm`, before spatial pooling.

Main diagnosis:

- Q-former v3 compressed real history close to repeated-current history.
- Delta v1 produced motion-sensitive temporal tokens, but the LLM/action heads did not reliably use them.
- The remaining bottleneck is likely temporal injection/use by the LLM/action heads, plus weak or indirect loss pressure for motion-critical yielding.

## 13. Next Actions

Current priority: keep code stable while v3/v6 train and write thesis sections
that do not depend on final results. Final v2/v5 selected renders have now been
checked and should be treated as pre-strict-cleanup evidence.

1. Let `temporal_delta_feature_v3` and `temporal_qformer_v6` train with strict interaction supervision.
2. When v3/v6 reach a useful checkpoint, render the same five selected failure routes before deciding on full evaluation.
3. If v3/v6 still fail, the next intervention should be loss/sampling or stronger interaction supervision, not simply more Q-former queries.
4. Do not remove current-frame tokens by default; use token dropout or ablations only if there is time.

Check jobs and GPU hours:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```
