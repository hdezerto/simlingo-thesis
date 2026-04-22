# SimLingo Temporal Training on Berzelius

Compact runbook for reproducing the current temporal Q-former training setup.

## Environment

For long-running interactive work on this branch, use one tmux session and reuse it:

```bash
tmux new -s temporal
```

If it already exists:

```bash
tmux attach -t temporal
```

Inside that tmux session, load the environment:

```bash
export USERNAME="${USERNAME:-${USER}}"
cd /proj/berzelius-2023-154/users/${USERNAME}/simlingo-thesis

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh
```

Unless noted otherwise, the remaining commands assume you are still in this `temporal` tmux session with that environment loaded.

Key variables from `thesis/env.sh`:
- `BASE_DIR=/proj/${PROJECT_ID}/users/${USERNAME}`
- `REPO_DIR=${BASE_DIR}/simlingo-thesis`
- `EVAL_OUT_ROOT=${BASE_DIR}/eval_results/Bench2Drive`
- `EVAL_RUN_NAME=simlingo`
- `BENCH2DRIVE_ROOT=${EVAL_OUT_ROOT}/${EVAL_RUN_NAME}/bench2drive`
- `LOG_ROOT=${BASE_DIR}/logs`
- `MODEL_CKPT=${BASE_DIR}/checkpoints/simlingo_pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- `SLURM_ACCOUNT=berzelius-2025-435`

Quick checks:

```bash
echo "$REPO_DIR"
echo "$MODEL_CKPT"
echo "$SLURM_ACCOUNT"
test -f "$MODEL_CKPT" && echo "checkpoint ok"
```

## Dataset Download

Download the released dataset from Hugging Face with Git LFS:

```bash
cd "${BASE_DIR}"
mkdir -p database
cd database

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/RenzKa/simlingo simlingo_hf
cd simlingo_hf
git lfs pull
```

## Extract Dataset And Buckets

```bash
cd "${BASE_DIR}/database"
mkdir -p simlingo
mkdir -p bucketsv2_simlingo
cp simlingo_hf/buckets_paths.pkl bucketsv2_simlingo/

cd simlingo_hf
for f in *.tar.gz; do
  echo "Extracting $f"
  tar -xzf "$f" -C "${BASE_DIR}/database/simlingo"
done
```

Expected outputs:
- `${BASE_DIR}/database/simlingo/data/simlingo`
- `${BASE_DIR}/database/bucketsv2_simlingo/buckets_paths.pkl`

## Create Repo Symlinks

The training scripts expect repo-local `database/` symlinks:

```bash
cd "${REPO_DIR}"
mkdir -p database
ln -sfn "${BASE_DIR}/database/simlingo" database/simlingo
ln -sfn "${BASE_DIR}/database/bucketsv2_simlingo" database/bucketsv2_simlingo
```

Verify:

```bash
readlink -f database/simlingo
readlink -f database/bucketsv2_simlingo
test -d database/simlingo/data/simlingo && echo "dataset ok"
test -f database/bucketsv2_simlingo/buckets_paths.pkl && echo "buckets ok"
```

## FlashAttention2

The current temporal training runs use FlashAttention2.

```bash
export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=8
/home/x_hugaf/.conda/envs/simlingo/bin/python -m pip install flash-attn --no-build-isolation
```

## Relevant SLURM Scripts

- `thesis/slurm/train_temporal.slurm`
  - full 8-GPU temporal training
  

## Training

Create logs once:

```bash
mkdir -p "${LOG_ROOT}/training"
```

Training is split between experiment configs and SLURM launchers:
- experiment configs live in `simlingo_training/config/experiment/`
- the current training launcher lives in `thesis/slurm/train_temporal.slurm`

What to edit where:
- edit experiment configs for model, temporal, dataset, optimizer, and other reproducible training settings
- edit `train_temporal.slurm` for cluster resources, walltime, job name, mail settings, and small run-specific Hydra overrides

For the temporal setup in this branch:
- the current launch config is `simlingo_training/config/experiment/temporal_qformer_v2.yaml`
- the training launcher is `thesis/slurm/train_temporal.slurm`
- it currently requests `8` GPUs, `64` CPU cores, `900G` RAM, and `3` days walltime

Submit from the repo root:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_temporal.slurm
```

Monitor:

```bash
squeue -u "$USERNAME"
tail -f "${LOG_ROOT}/training/<jobid>.out"
tail -f "${LOG_ROOT}/training/<jobid>.err"
```

## Bench2Drive Evaluation

The thesis workflow is:
1. run the model online on all Bench2Drive routes
2. merge the per-route JSONs
3. run the scenario-failure analysis
4. optionally render selected cases

Do not run merge/analyze before the online evaluation has finished.

The launcher is:
- `start_eval_simlingo.py`

It writes results under:
- `${EVAL_OUT_ROOT}/${EVAL_RUN_NAME}/bench2drive/<seed>/...`

Important:
- if you keep the default `EVAL_RUN_NAME=simlingo`, the temporal evaluation will share the same result tree as the baseline
- that is unsafe because `start_eval_simlingo.py` skips routes that already have completed result files
- use a distinct run name for every temporal evaluation run

Inside the existing `temporal` tmux session:

```bash
export MODEL_CKPT="${REPO_DIR}/outputs/2026_04_19_09_03_16_temporal_qformer_v1_8gpu/checkpoints/epoch=000.ckpt"
export EVAL_RUN_NAME="simlingo_temporal_v1_8gpu_epoch000"
source thesis/env.sh
echo "8" > max_num_jobs.txt
python start_eval_simlingo.py
```

This gives:
- `${BENCH2DRIVE_ROOT}=${BASE_DIR}/eval_results/Bench2Drive/simlingo_temporal_v1_8gpu_epoch000/bench2drive`

Notes:
- using the DeepSpeed checkpoint directory is fine here; the temporal agent can load it directly
- for many eval jobs, the converted `pytorch_model.bin` may start slightly faster if you prefer to point `MODEL_CKPT` there instead
- reuse the same `temporal` tmux session for this branch rather than creating extra sessions
- `thesis/slurm/eval_simlingo.slurm` still works, but the tmux workflow is the baseline-style orchestrator path and is easier to monitor

## Merge And Analyze Results

After all Bench2Drive jobs are finished:

```bash
export EVAL_RUN_NAME="simlingo_temporal_v1_8gpu_epoch000"
source thesis/env.sh

python thesis/analysis/merge_bench2drive_results.py -b "${BENCH2DRIVE_ROOT}"

python thesis/analysis/analyze_scenario_failures.py \
  -b "${BENCH2DRIVE_ROOT}" \
  -o thesis/results/scenario_failure_report_temporal_v1_8gpu_epoch000.txt
```

Notes:
- `merge_bench2drive_results.py` reads `${BENCH2DRIVE_ROOT}/1/res`, `/2/res`, and `/3/res`
- `analyze_scenario_failures.py` reads the same merged run tree
- they expect the full benchmark files `000_res.json` ... `219_res.json`, not manifest render outputs such as `rendering_route2201_seed1_res.json`
- if you omit `-o`, the analysis report will overwrite `thesis/results/scenario_failure_report.txt`

## Checkpoints

DeepSpeed checkpoints are saved as directories, for example:
- `outputs/.../checkpoints/epoch=000.ckpt`
- `outputs/.../checkpoints/last.ckpt`

These are training-resume checkpoints, not plain `torch.load(...)` model files.

## Convert A Checkpoint For Inference

Training checkpoints are saved in DeepSpeed format as directories such as `epoch=000.ckpt/`.

If you want a plain single-file PyTorch checkpoint for inference, convert it with `zero_to_fp32.py`:

```bash
cd "${REPO_DIR}"
mkdir -p /tmp/${USER}_triton
PYTHONPATH="${REPO_DIR}" \
TRITON_CACHE_DIR="/tmp/${USER}_triton" \
/home/x_hugaf/.conda/envs/simlingo/bin/python \
outputs/2026_04_19_09_03_16_temporal_qformer_v1_8gpu/checkpoints/epoch=000.ckpt/zero_to_fp32.py \
outputs/2026_04_19_09_03_16_temporal_qformer_v1_8gpu/checkpoints/epoch=000.ckpt \
outputs/2026_04_19_09_03_16_temporal_qformer_v1_8gpu/checkpoints/epoch=000_fp32
```

Result:
- `outputs/2026_04_19_09_03_16_temporal_qformer_v1_8gpu/checkpoints/epoch=000_fp32/pytorch_model.bin`

Notes:
- This creates a normal fp32 state dict file.
- Rendering can load either the DeepSpeed checkpoint directory or the converted `pytorch_model.bin`.
- The conversion is optional.
- For many render jobs, the converted `pytorch_model.bin` is recommended because startup is faster and each job avoids reconstructing the fp32 state dict from the DeepSpeed shards.



## Render Failure Cases With The Temporal Model

Rendering uses the online CARLA agent:
- `team_code/agent_simlingo.py`

It can load either:
- a DeepSpeed checkpoint directory such as `outputs/.../checkpoints/epoch=000.ckpt`
- or a converted plain checkpoint such as `outputs/.../checkpoints/epoch=000_fp32/pytorch_model.bin`

Use the DeepSpeed checkpoint directly for convenience, or the converted `.bin` for faster startup across many render jobs.

Edit the manifest, for example:
- `thesis/rendering/manifests/render_manifest_temporal_v1_8gpu_epoch004_selected.json`

Set:
- `agent`
  - a distinct output name such as `simlingo_temporal_v1_8gpu_epoch004_render`
- `agent_file`
  - `team_code/agent_simlingo.py`
- `checkpoint`
  - the temporal checkpoint you want to render
- `out_root`
  - `${BASE_DIR}/eval_results/Bench2Drive`

Dry run:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_temporal_v1_8gpu_epoch004_selected.json --dry-run
```

Submit:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_temporal_v1_8gpu_epoch004_selected.json
```

Monitor:

```bash
squeue -u "$USERNAME"
```

These jobs run the temporal model online for the selected route/seed cases and write frames, metadata, and optional MP4s under the configured output root.


## Cleanup

After extraction and verification, the original Hugging Face clone can be removed:

```bash
rm -rf "${BASE_DIR}/database/simlingo_hf"
```



-----------------------------------------------

NOTES:

Temporal v1 experiment:

- config:
  - `simlingo_training/config/experiment/temporal_qformer_v1.yaml`
- launcher:
  - `thesis/slurm/train_temporal.slurm`
- settings:
  - `hist_len=3`
  - `history_stride=1`
  - `temporal_model.enabled=true`
  - `temporal_model.num_queries=8`
  - `temporal_model.num_layers=2`
  - `temporal_model.gate_init=-2.0`
  - `data_module.batch_size=6`
  - `max_epochs=5`
  - `val_every_n_epochs=1`
- frozen:
  - vision encoder
  - language model
  - adaptors
  - waypoint encoder
- trainable:
  - temporal Q-former

Recommended next experiment:

- `hist_len=5`
- `history_stride=1`
- `freeze_language_model=false`
- keep `language_model.lora=True`
- `data_module.batch_size=12`
- `temporal_model.num_queries=16`
- `temporal_model.gate_init=-2.0`
- `max_epochs=14`
- `val_every_n_epochs=2`

- Reason:
  - `hist_len=3` is likely too short to expose clear motion cues in junction failures.
  - allowing the LoRA-adapted language model to train should help the LLM learn how to use the new temporal tokens.
  - `batch_size=12`, `14` epochs, and validation every `2` epochs bring the run closer to the SimLingo paper recipe while still testing the temporal extension.
  - `num_queries=16` is more aligned with the ORION-inspired query-based temporal design than `12`, while still remaining lightweight compared with larger Q-Former-style bottlenecks.
  - keep `gate_init=-2.0` conservative so the new temporal branch starts with low influence instead of injecting too much noise into the pretrained stack early in training.
  - `14` epochs gives the temporal branch substantially more time to become useful than the current short run.

- Suggested follow-up after that:
  - if this still does not help, try a longer temporal span with non-contiguous history sampling instead of only adjacent frames.
  - the dataset stores one frame every `0.25 s` (`20 FPS` CARLA, saved every `5` sim steps), so a good first non-contiguous setting is `history_stride=2` over saved frames, i.e. `0.5 s` between sampled history frames.
  - training now supports `history_stride` directly in the experiment config.
  - inference now mirrors the training spacing automatically: it samples past frames every `history_stride * 5` simulator steps by default.
  - only override this manually if needed via `TEMPORAL_INFERENCE_STRIDE=<sim_steps>`.

Potential thesis ablation:

- quick development ablation:
  - keep the same temporal setup but compare `freeze_language_model=true` vs `false`
  - purpose:
    - test whether adapting the LoRA language model is necessary for the LLM to learn how to use the temporal tokens

- parameter-matched no-temporal control:
  - keep the same temporal Q-former architecture and training setup
  - but replace the past-frame inputs with copies of the current frame
  - purpose:
    - test whether any gain comes from actual temporal information rather than just extra parameters or extra token capacity


----------------------------------
