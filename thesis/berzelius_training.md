# SimLingo Training on Berzelius

Concise runbook for baseline training setup before implementing the temporal method.

## 0. Load Environment

```bash
export USERNAME="${USERNAME:-${USER}}"
cd /proj/berzelius-2023-154/users/${USERNAME}/simlingo-thesis

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh
```

Quick checks:

```bash
echo "$REPO_DIR"
echo "$MODEL_CKPT"
echo "$SLURM_ACCOUNT"
echo "$SLURM_PARTITION"
test -f "$MODEL_CKPT" && echo "checkpoint ok"
```

## 1. Download Dataset

Store the dataset outside the repo:

```bash
cd "${BASE_DIR}"
mkdir -p database
cd database

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/RenzKa/simlingo simlingo_hf
cd simlingo_hf
git lfs pull
```

This downloads:
- the compressed dataset tarballs
- `buckets_paths.pkl`

## 2. Extract Dataset

Create target folders:

```bash
cd "${BASE_DIR}/database"
mkdir -p simlingo
mkdir -p bucketsv2_simlingo
cp simlingo_hf/buckets_paths.pkl bucketsv2_simlingo/
```

Extract all tarballs into one dataset root:

```bash
cd "${BASE_DIR}/database/simlingo_hf"
for f in *.tar.gz; do
  echo "Extracting $f"
  tar -xzf "$f" -C "${BASE_DIR}/database/simlingo"
done
```

Expected extracted top-level folders:
- `data/`
- `commentary/`
- `dreamer/`
- `drivelm/`

Quick checks:

```bash
test -d "${BASE_DIR}/database/simlingo/data/simlingo" && echo "dataset ok"
test -f "${BASE_DIR}/database/bucketsv2_simlingo/buckets_paths.pkl" && echo "bucket file ok"
```

## 3. Expose Dataset Paths Inside the Repo

The training configs expect:
- `database/simlingo`
- `database/bucketsv2_simlingo`

Create repo-local symlinks once:

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
find -L database/simlingo/data/simlingo -maxdepth 3 -type d | sed -n '1,20p'
ls -lh database/bucketsv2_simlingo/buckets_paths.pkl
```

## 4. Training Uses SLURM

Do not run `python simlingo_training/train.py ...` directly from a login shell.

Use the smoke-test SLURM launcher:
- `thesis/slurm/train_smoke.slurm`

Create log folder once:

```bash
mkdir -p "${LOG_ROOT}/training"
```

Submit the smoke test:

```bash
cd "${REPO_DIR}"
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_smoke.slurm
```

Monitor:

```bash
squeue -u "$USERNAME"
tail -f "${LOG_ROOT}/training/<jobid>.out"
tail -f "${LOG_ROOT}/training/<jobid>.err"
```

The smoke-test job:
- requests 1 GPU
- uses `experiment=debug`
- keeps WandB offline
- uses batch size 1 and `num_workers=0`
- uses `overfit=1` so the epoch finishes quickly on a single repeated batch
- runs 1 epoch
- validates at epoch 1 so Lightning writes a checkpoint
- loads the released baseline checkpoint through `MODEL_CKPT`
- sends email on `END` and `FAIL`
- keeps `CUDA_HOME` and `PATH`
- makes batch jobs robust to unset shell vars:
  - `CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"`
  - `PYTHONPATH="${PYTHONPATH:-}:..."`
- explicitly unsets `LD_LIBRARY_PATH` before launching Python so PyTorch does not pick the system cuDNN from `/software/sse/manual/CUDA/.../lib64`

Resume smoke test:

```bash
cd "${REPO_DIR}"
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_resume_smoke.slurm
```

This resumes from:
- `outputs/2026-04-18/14-26-58/checkpoints/last.ckpt`

and verifies that Lightning/DeepSpeed can restore trainer state from a saved checkpoint.

Bucketed multi-GPU dry run:

```bash
cd "${REPO_DIR}"
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_bucket_multigpu_smoke.slurm
```

This uses:
- `experiment=simlingo_seed1`
- `gpus=2`
- `overfit=1`
- `max_epochs=1`

and is meant only to verify that the bucketed datamodule and multi-GPU training work correctly.

## 5. Success Criteria

The smoke test is successful if:
- dataset paths resolve
- `buckets_paths.pkl` loads
- InternVL loads
- the released checkpoint loads
- at least one training batch runs
- a loss value is produced
- the run directory and `checkpoints/` directory are created
- at least one checkpoint file is written

Current status:
- `train_smoke.slurm` is working
- the successful checkpoint smoke test wrote:
  - `outputs/2026-04-18/14-26-58/checkpoints/epoch=000.ckpt`
  - `outputs/2026-04-18/14-26-58/checkpoints/last.ckpt`
- `train_resume_smoke.slurm` successfully restored from `last.ckpt` and resumed at `Epoch 1`
- the resume smoke test then failed on the LR scheduler with:
  - `ValueError: Tried to step 2 times. The specified number of total steps is 1`
- interpretation:
  - checkpoint restore works
  - the tiny `overfit=1` resume setup is not a clean full resume-validation test for scheduler state
- `train_bucket_multigpu_smoke.slurm` completed successfully
- the successful bucketed 2-GPU dry run wrote:
  - `outputs/2026_04_18_22_54_15_bucket_multigpu_smoke/checkpoints/epoch=000.ckpt`
  - `outputs/2026_04_18_22_54_15_bucket_multigpu_smoke/checkpoints/last.ckpt`
- this confirms:
  - the bucketed datamodule works
  - 2-GPU distributed training works
  - checkpoint saving also works in the bucketed multi-GPU setup
- there is no remaining baseline infrastructure blocker before implementing the temporal module

## 6. Cleanup Later

After the smoke test succeeds, remove the original compressed clone to free space:

```bash
rm -rf "${BASE_DIR}/database/simlingo_hf"
```

Do not remove it before the smoke test passes.
