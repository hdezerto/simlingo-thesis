# SimLingo Training on Berzelius

Concise runbook for the current working training setup.

## 1. Load Environment

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
test -f "$MODEL_CKPT" && echo "checkpoint ok"
```

## 2. Download And Extract Dataset

```bash
cd "${BASE_DIR}"
mkdir -p database
cd database

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/RenzKa/simlingo simlingo_hf
cd simlingo_hf
git lfs pull
```

```bash
cd "${BASE_DIR}/database"
mkdir -p simlingo
mkdir -p bucketsv2_simlingo
cp simlingo_hf/buckets_paths.pkl bucketsv2_simlingo/

cd "${BASE_DIR}/database/simlingo_hf"
for f in *.tar.gz; do
  echo "Extracting $f"
  tar -xzf "$f" -C "${BASE_DIR}/database/simlingo"
done
```

Expected:
- `${BASE_DIR}/database/simlingo/data/simlingo`
- `${BASE_DIR}/database/bucketsv2_simlingo/buckets_paths.pkl`

## 3. Create Repo Symlinks

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
```

## 4. Submit Jobs Through SLURM

Create the log directory once:

```bash
mkdir -p "${LOG_ROOT}/training"
```

All working SLURM scripts:
- unset `LD_LIBRARY_PATH` before Python
- use the released baseline checkpoint from `MODEL_CKPT`
- send email on `END,FAIL`

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

## 5. Useful Scripts

Baseline checkpoint smoke test:
- `thesis/slurm/train_smoke.slurm`
- purpose: verify 1-GPU training, validation, and checkpoint writing

Temporal smoke test:
- `thesis/slurm/train_temporal_smoke.slurm`
- purpose: verify `hist_len=3` and the temporal module path end to end

Temporal 1-GPU pilot:
- `thesis/slurm/train_temporal_pilot.slurm`
- uses `experiment=temporal_qformer_seed1`
- purpose: short bucketed temporal training run with the temporal module as the only new trainable component

Optional baseline bucketed multi-GPU dry run:
- `thesis/slurm/train_bucket_multigpu_smoke.slurm`
- purpose: verify bucketed datamodule plus distributed training

Temporal 2-GPU dry run:
- `thesis/slurm/train_temporal_multigpu_smoke.slurm`
- purpose: verify temporal training also works in distributed mode before a final long run

Final temporal training:
- `thesis/slurm/train_temporal_final.slurm`
- purpose: full 8-GPU temporal-only training run using the full dataset and bucketed recipe

## 6. Current Temporal Experiment

Experiment config:
- `simlingo_training/config/experiment/temporal_qformer_seed1.yaml`

Current settings:
- `hist_len=3`
- `model.temporal_model.enabled=true`
- vision encoder frozen
- language model frozen
- adaptor frozen
- waypoint encoder frozen

This means the first temporal experiment trains only the temporal Q-former.

## 7. About The Temporal Pilot

`train_temporal_pilot.slurm` currently overrides:

```bash
data_module.base_dataset.max_routes=20
data_module.base_dataset.max_samples=50
max_epochs=1
gpus=1
data_module.batch_size=1
data_module.num_workers=0
```

Meaning:
- `max_routes=20`: only scan 20 routes per dataset build
- `max_samples=50`: cap each bucket dataset at 50 samples after bucket filtering

Important:
- the pilot still uses the bucketed code path
- but `max_samples=50` is applied per bucket dataset, so many rare buckets may end up empty
- good for a fast pilot, not representative of the full bucket distribution

For a more realistic temporal pilot later:
- keep `max_routes=20`
- remove `max_samples`

## 8. Current Status

Validated:
- baseline 1-GPU smoke test works and writes checkpoints
- bucketed 2-GPU baseline dry run works
- `hist_len=3` fallback smoke test works
- temporal smoke test works
- temporal 1-GPU pilot works

Still recommended before a final long temporal run:
- run `train_temporal_multigpu_smoke.slurm`

After that, the final launcher is:

```bash
cd "${REPO_DIR}"
source thesis/env.sh
sbatch --account="${SLURM_ACCOUNT}" thesis/slurm/train_temporal_final.slurm
```

## 9. Cleanup

After setup is stable, you can remove the original compressed clone:

```bash
rm -rf "${BASE_DIR}/database/simlingo_hf"
```

This should not affect the current training setup, which uses:
- `${BASE_DIR}/database/simlingo`
- `${BASE_DIR}/database/bucketsv2_simlingo`
