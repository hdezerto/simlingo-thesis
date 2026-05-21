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
thesis/slurm/train_temporal_delta_feature_v3.slurm   # completed result version
thesis/slurm/train_temporal_qformer_v6.slurm         # completed result version
thesis/slurm/train_temporal_delta_feature_v4.slurm   # next run: scene-fact cleanup + fresh LLM LoRA
thesis/slurm/train_temporal_qformer_v7.slurm         # next run: scene-fact cleanup + fresh LLM LoRA
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

| Method | Core idea | Current version |
| --- | --- | --- |
| Q-former | Learned query tokens attend over temporal visual memory | v6: 32 queries over `[past, current-past, abs(current-past)]`, 2 layers, 8 heads, `dropout=0.1`, no gate |
| Delta feature | Encode current-vs-past feature deltas into compact motion tokens | v3: signed + absolute deltas, `delta_decay=0.9`, spatial pooling to 64 tokens, no gate |

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

The loss is masked BCE-with-logits, weighted by `aux_loss_weight=0.5`. It does not replace the main language/route/waypoint losses. `dynamic_sample_weight=3.0` is separate and upweights main losses when `actor_motion_labels[3]=1`.

### Supervision Versions

Current results use the V3/V6 supervision:

- Labels are computed from current boxes and future expert waypoints.
- Stale junction text is removed only when preliminary labels show a moving actor near the ego path.
- `actor_motion_labels[3]` can be suppressed by cleaned commentary that looks like red-light stopping, stop-sign stopping, construction, front-car following, or explicit go/accelerate.
- This explains remaining rendered failures: stale `black car` / `stay behind`, green-light acceleration, and `vehicles are stopped at the junction` text can still leak through.

Current working-tree refactor for the next run:

- Builds one scene-fact object from boxes, current measurements, and future expert waypoints.
- Computes labels from scene facts only; cleaned commentary no longer decides whether label `3` is true.
- Removes stale `junction is clear` / `vehicles are stopped` claims, false traffic-light color claims, and green-light go/speed-up instructions contradicted by moving-conflict facts.
- In strict dynamic-yield samples, removes following/color-distance distractions such as `stay behind` and `black car` even when a real lead vehicle exists, so the language target focuses on the interaction-yield reason.
- Sets `actor_motion_labels[3]=1` for dynamic-yield cases: a moving side/cross actor, or a moving actor near the curved future turn path, is close to the ego future path; the expert waypoints slow or stop; and no verified red/yellow light, stop sign, construction, or pure lead-vehicle-following reason explains the slowdown.

Current-working-tree commentary changes:

| Case | Example sentence | When changed | Result |
| --- | --- | --- | --- |
| Stale stopped-junction claim | `vehicles are stopped at the junction` | Moving actor/path or junction conflict exists | Sentence removed |
| Stale clear-junction claim | `junction is clear` | Moving actor/path or junction conflict exists | Sentence removed |
| Stale moving-away claim | `vehicle in the junction is moving away` | Moving actor/path or junction conflict exists | Sentence removed |
| Green-light go/speed-up instruction | `Accelerate because the traffic light is green`, `Speed up because the traffic light is green` | Moving conflict or expert-yield context exists | Sentence removed |
| False red-light claim | `Remain stopped due to the red traffic light` | Affecting light is verified green | Sentence removed |
| False green-light claim | `traffic light is green` | Red/yellow light or light hazard is verified | Sentence removed |
| Following/color-distance distraction | `stay behind`, `remain behind`, `black car` | Strict dynamic-yield sample, or no verified lead vehicle | Sentence removed |
| Valid rule stop | `Remain stopped due to the red traffic light` | Red/yellow light is verified | Sentence kept; label `3=0` |
| Valid lead following | `stay behind the vehicle in front` | Lead vehicle is verified and no strict dynamic-yield label is set | Sentence kept; label `3=0` |

If all sentences are removed, the cleaned commentary becomes `Follow the route.`.
Then one motion sentence can be appended: strict dynamic-yield text, moving
left/right actor text, moving actor near-path text, or stopped actor near-path
text.

Examples:

- `Turn left. Accelerate because the traffic light is green and the other vehicles are stopped at the junction and the junction is clear.` -> `Turn left. A moving actor is close to the ego path, so the ego should yield until the path is clear.`
- `Follow the route. Decelerate to stay behind the black car that is to the front left in 29.2 meters.` -> `Follow the route. A moving actor is close to the ego path, so the ego should yield until the path is clear.` for strict dynamic-yield samples.
- `Turn left. Remain stopped due to the red traffic light in 1.8 meters.` -> `Turn left.` when the affecting light is verified green.

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
| Does scene-fact interaction cleanup select useful cases? | `interaction_supervision_audit_scene_facts_v2`: `150` yield samples, all with interaction text; `92` conflicting claims removed; final targets contain no tracked stale `junction is clear`, `vehicles are stopped`, `black car`, `stay behind`, green-light, or accelerate/speed-up phrases | Current supervision is clean enough to launch the next training versions |

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
| Delta feature v4 planned | Same architecture/training setup as Delta v3, but with scene-fact supervision cleanup v2 and fresh LLM LoRA loaded from the SimLingo checkpoint | Tests whether removing stale language priors fixes delta failures |
| Q-former v7 planned | Same architecture/training setup as Q-former v6, but with scene-fact supervision cleanup v2 and fresh LLM LoRA loaded from the SimLingo checkpoint | Most promising next run because Q-former v6 already fixed some selected failures |

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

Current priority: launch the supervision-focused next runs, while preserving Q-former v6 `epoch=013` as the strongest completed result so far.

1. Train Delta feature v4 and Q-former v7 if GPU resources allow.
2. Both load the SimLingo `epoch=013` checkpoint, freeze the vision backbone and waypoint input encoder, reset LLM LoRA from scratch, and keep the v3/v6 temporal architectures unchanged.
3. The only intended changes relative to v3/v6 are scene-fact supervision cleanup v2 and fresh LLM LoRA.
4. Render selected routes at `epoch=009` and the final checkpoint, then compare against Delta v3 and Q-former v6.
5. If v4/v7 still produce stale commentary, the next hypothesis is that stale behavior lives outside LoRA or that front-camera-only observations are insufficient for some actor-flow cases.

Check jobs and GPU hours:

```bash
squeue -u "${USER}"
projinfo -m "${SLURM_ACCOUNT}"
```




To do in the future:
- Ablation for the temporal module
- Try multi-view