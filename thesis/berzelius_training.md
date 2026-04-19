# SimLingo Temporal Training on Berzelius

Compact runbook for reproducing the current temporal Q-former training setup.

## Environment

```bash
export USERNAME="${USERNAME:-${USER}}"
cd /proj/berzelius-2023-154/users/${USERNAME}/simlingo-thesis

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh
```

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

- `thesis/slurm/train_smoke.slurm`
  - baseline 1-GPU smoke test
- `thesis/slurm/train_temporal_smoke.slurm`
  - temporal path smoke test
- `thesis/slurm/train_temporal_pilot.slurm`
  - short 1-GPU temporal pilot
- `thesis/slurm/train_temporal_multigpu_smoke.slurm`
  - short temporal distributed dry run
- `thesis/slurm/train_temporal_final.slurm`
  - full 8-GPU temporal training

## Current Temporal Experiment

Config:
- `simlingo_training/config/experiment/temporal_qformer_seed1.yaml`

Current settings:
- `hist_len=3`
- `temporal_model.enabled=true`
- `num_queries=8`
- `num_layers=2`
- `batch_size=6`
- `max_epochs=5`
- `val_every_n_epochs=1`

Frozen:
- vision encoder
- language model
- adaptors
- waypoint encoder

Trainable:
- temporal Q-former only

## Submit Training Jobs

Create logs once:

```bash
mkdir -p "${LOG_ROOT}/training"
```

Submit from the repo root:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/<script>.slurm
```

Monitor:

```bash
squeue -u "$USERNAME"
tail -f "${LOG_ROOT}/training/<jobid>.out"
tail -f "${LOG_ROOT}/training/<jobid>.err"
```

## Final Temporal Training

Launcher:
- `thesis/slurm/train_temporal_final.slurm`

Current final setup:
- 8x A100 80GB
- DeepSpeed stage 2
- offline W&B
- FlashAttention2-enabled env
- `data_module.num_workers=8`

Launch:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_temporal_final.slurm
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

Recommended temporal setup:

```bash
cd "${REPO_DIR}"
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo

export MODEL_CKPT="${REPO_DIR}/outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000.ckpt"
export EVAL_RUN_NAME="simlingo_temporal_epoch000"
source thesis/env.sh
```

This gives:
- `${BENCH2DRIVE_ROOT}=${BASE_DIR}/eval_results/Bench2Drive/simlingo_temporal_epoch000/bench2drive`

Run the orchestrator like the baseline, inside tmux:

```bash
cd "${REPO_DIR}"
tmux new -s temporal_eval
```

Inside tmux:

```bash
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
export MODEL_CKPT="${REPO_DIR}/outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000.ckpt"
export EVAL_RUN_NAME="simlingo_temporal_epoch000"
source thesis/env.sh
echo "8" > max_num_jobs.txt
python start_eval_simlingo.py
```

Notes:
- using the DeepSpeed checkpoint directory is fine here; the temporal agent can load it directly
- for many eval jobs, the converted `pytorch_model.bin` may start slightly faster if you prefer to point `MODEL_CKPT` there instead
- `thesis/slurm/eval_simlingo.slurm` still works, but the tmux workflow is the baseline-style orchestrator path and is easier to monitor

## Merge And Analyze Results

After all Bench2Drive jobs are finished, merge first:

```bash
cd "${REPO_DIR}"
export EVAL_RUN_NAME="simlingo_temporal_epoch000"
source thesis/env.sh
python thesis/analysis/merge_bench2drive_results.py -b "${BENCH2DRIVE_ROOT}"
```

Then run the failure analysis:

```bash
cd "${REPO_DIR}"
export EVAL_RUN_NAME="simlingo_temporal_epoch000"
source thesis/env.sh
python thesis/analysis/analyze_scenario_failures.py \
  -b "${BENCH2DRIVE_ROOT}" \
  -o thesis/results/scenario_failure_report_temporal_epoch000.txt
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

Example: convert epoch 0 of the current temporal run to a plain weights file:

```bash
cd "${REPO_DIR}"
mkdir -p /tmp/${USER}_triton
PYTHONPATH="${REPO_DIR}" \
TRITON_CACHE_DIR="/tmp/${USER}_triton" \
/home/x_hugaf/.conda/envs/simlingo/bin/python \
outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000.ckpt/zero_to_fp32.py \
outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000.ckpt \
outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000_fp32
```

Result:
- `outputs/2026_04_19_09_03_16_temporal_qformer_final_8gpu/checkpoints/epoch=000_fp32/pytorch_model.bin`

Notes:
- Rendering can now load either the DeepSpeed checkpoint directory or the converted `pytorch_model.bin`.
- For many render jobs, the converted `pytorch_model.bin` is recommended because startup is faster and each job avoids reconstructing the fp32 state dict from the DeepSpeed shards.

## Render Failure Cases With The Temporal Model

The thesis rendering pipeline uses the online CARLA agent:
- `team_code/agent_simlingo.py`

That agent now supports the temporal module:
- it reads `hist_len` and temporal settings from the run config
- it keeps a temporal frame buffer
- it can load either a DeepSpeed checkpoint directory or a converted `pytorch_model.bin`

Recommended checkpoint choices for the render manifest:
- direct DeepSpeed checkpoint:
  - `outputs/.../checkpoints/epoch=000.ckpt`
- faster startup for many jobs:
  - `outputs/.../checkpoints/epoch=000_fp32/pytorch_model.bin`

Edit the manifest, for example:
- `thesis/rendering/manifests/render_manifest_2.json`

Set:
- `agent` to a distinct run name such as `simlingo_temporal_epoch000_render`
- `agent_file` to `team_code/agent_simlingo.py`
- `checkpoint` to the temporal checkpoint you want to render
- `out_root` to `${BASE_DIR}/eval_results/Bench2Drive`

Dry run:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_2.json --dry-run
```

Submit:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_2.json
```

Monitor:

```bash
squeue -u "$USERNAME"
```

The render jobs will:
- run the temporal model online for the selected route/seed cases
- save frames and metadata under the configured output root
- stitch MP4s when `debug_viz=true`

`simlingo_training/eval.py` is optional and is not part of the thesis render/failure-analysis workflow unless you want offline dataset inference.

## Cleanup

After extraction and verification, the original Hugging Face clone can be removed:

```bash
rm -rf "${BASE_DIR}/database/simlingo_hf"
```
