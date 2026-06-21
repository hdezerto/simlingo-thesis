# SimLingo Baseline on Berzelius

This guide is linked from the repository `README.md` and gives the
Berzelius-specific workflow for reproducing the corrected SimLingo baseline with
CARLA 0.9.15 and Bench2Drive. CARLA binaries, pretrained weights, raw logs, and
raw evaluation outputs are expected outside the repository.

## Index

- [0. Set Variables (recommended)](#0-set-variables-recommended)
- [1. Expected Layout](#1-expected-layout)
- [2. Install CARLA 0.9.15](#2-install-carla-0915)
    - [2.1 Download and extract](#21-download-and-extract)
    - [2.2 Install Additional Maps (required for Bench2Drive)](#22-install-additional-maps-required-for-bench2drive)
- [3. Code and Python Environment](#3-code-and-python-environment)
- [4. Pretrained Weights](#4-pretrained-weights)
- [5. Environment Variables](#5-environment-variables)
- [6. Smoke Test CARLA](#6-smoke-test-carla)
- [7. Run Corrected Baseline Evaluation](#7-run-corrected-baseline-evaluation)
- [8. Monitor, Resume, and Stop](#8-monitor-resume-and-stop)
- [9. Results and Analysis](#9-results-and-analysis)
- [10. Render Selected Baseline Videos](#10-render-selected-baseline-videos)
- [Troubleshooting](#troubleshooting)

## 0. Set Variables

Run once per shell before following commands in this README:

```bash
export USERNAME="${USERNAME:-${USER:-<your-liu-id>}}"
cd /proj/berzelius-2023-154/users/${USERNAME}/simlingo-thesis
source thesis/env.sh
```

Note:
- Thesis-specific additions are grouped under `thesis/`.
- `thesis/env.sh` configures the repository runtime used by the Berzelius workflow.

Quick preflight checks:

```bash
echo "$PROJECT_ID"
echo "$USERNAME"
echo "$BASE_DIR"
echo "$CARLA_ROOT"
echo "$MODEL_CKPT"
echo "$SLURM_ACCOUNT"
echo "$SLURM_PARTITION"
test -f "$MODEL_CKPT" && echo "checkpoint ok"
test -x "$CARLA_ROOT/CarlaUE4.sh" && echo "carla ok"
```

## 1. Expected Layout

```text
${BASE_DIR}/
├── simlingo-thesis/
├── carla/
│   └── CARLA_${CARLA_VERSION}/
├── checkpoints/
│   └── simlingo_pretrained/
├── eval_results/
│   └── Bench2Drive/
└── logs/
    ├── carla_test/
    └── baseline/
```

Create base folders if needed:

```bash
mkdir -p ${BASE_DIR}/{carla,checkpoints,logs,eval_results}
```

## 2. Install CARLA 0.9.15

### 2.1 Download and extract

```bash
cd ${BASE_DIR}/carla
wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz
tar -xzf CARLA_${CARLA_VERSION}.tar.gz
```

Note:
- This archive extracts files directly (for example `CarlaUE4/`, `CHANGELOG`) and not always into a wrapper directory.

If needed, normalize to the expected path:

```bash
mkdir -p "${CARLA_ROOT}"
shopt -s dotglob extglob
mv -- ${BASE_DIR}/carla/!(CARLA_${CARLA_VERSION}|CARLA_${CARLA_VERSION}.tar.gz) \
    "${CARLA_ROOT}/"
```

Then remove archive:

```bash
rm -f ${BASE_DIR}/carla/CARLA_${CARLA_VERSION}.tar.gz
```

Verify launcher exists:

```bash
ls ${CARLA_ROOT}/CarlaUE4.sh
```

### 2.2 Install Additional Maps (required for Bench2Drive)

If missing, routes fail with errors like `Map 'Town12' not found` or `Map 'Town13' not found`.

```bash
cd ${CARLA_ROOT}/Import
wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_${CARLA_VERSION}.tar.gz
cd ${CARLA_ROOT}
bash ImportAssets.sh
```

Verify maps were imported:

```bash
find ${CARLA_ROOT}/CarlaUE4/Content/Carla/Maps \
  -type f \( -iname "Town12*" -o -iname "Town13*" \) | head
```

Optional cleanup after successful import:

```bash
rm -f ${CARLA_ROOT}/Import/AdditionalMaps_${CARLA_VERSION}.tar.gz
```

## 3. Code and Python Environment

All commands below run on a Berzelius login node.

```bash
ssh "${USERNAME}"@berzelius.nsc.liu.se
cd ${BASE_DIR}
git clone https://github.com/hdezerto/simlingo-thesis.git
cd simlingo-thesis
git checkout temporal-module
```

Create and activate environment (first time):

```bash
module load Miniforge3/24.7.1-2-hpc1-bdist
conda env create -f environment.yaml
conda activate simlingo
pip install torch==2.2.0
pip install flash-attn==2.7.0.post2 || true
```

If `simlingo` already exists and imports work, you can reuse it without recreating.

## 4. Pretrained Weights

```bash
cd ${CHECKPOINT_ROOT}
git lfs install
git clone https://huggingface.co/RenzKa/simlingo simlingo_pretrained
```

Verify checkpoint file:

```bash
ls -lh ${MODEL_CKPT}
```

## 5. Environment Variables

From `simlingo-thesis/`:

```bash
source thesis/env.sh
```

Quick checks:

```bash
echo "$PROJECT_ROOT"
echo "$CARLA_ROOT"
echo "$CHECKPOINT_DIR"
```

## 6. Smoke Test CARLA

Create log folders once:

```bash
mkdir -p ${BASE_DIR}/logs/carla_test
mkdir -p ${BASE_DIR}/logs/baseline
```

Submit smoke test:

```bash
cd ${BASE_DIR}/simlingo-thesis
sbatch thesis/slurm/carla_test.slurm
squeue -u "$USERNAME"
```

Important:
- This smoke job only launches CARLA in foreground and does not auto-exit.
- After confirming startup in logs, cancel manually.

Monitor logs:

```bash
ls -ltr ${BASE_DIR}/logs/carla_test | tail
tail -f ${BASE_DIR}/logs/carla_test/<JOBID>.out
tail -f ${BASE_DIR}/logs/carla_test/<JOBID>.err
```

Cancel when confirmed:

```bash
scancel <JOBID>
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

`CANCELLED` is expected for this manual smoke-test stop.

## 7. Run Corrected Baseline Evaluation

Run the full corrected baseline evaluation from an interactive `tmux` session.
The launcher sets the thesis baseline name, checkpoint, output root, and maximum
number of concurrent route jobs.

```bash
cd ${BASE_DIR}/simlingo-thesis
tmux new -As baseline
```

Inside tmux:

```bash
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh

MAX_EVAL_JOBS=8 bash thesis/scripts/launch_baseline_current_full_eval.sh
```

The launcher evaluates the released SimLingo `epoch=013` checkpoint with the
current thesis evaluation code and writes results to:

```bash
${BASE_DIR}/eval_results/Bench2Drive/simlingo_baseline_evalfix_full/bench2drive
```

Notes:

- The full benchmark is `220` routes over `3` seeds, i.e. `660` route-seed
  attempts.
- `start_eval_simlingo.py` skips completed result files, so the same launcher can
  resume a partial run after active route jobs have finished.
- Output such as repeated queue-length numbers is normal manager output, not an
  immediate failure.

Detach tmux with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t baseline
```

## 8. Monitor, Resume, and Stop

Monitor active jobs:

```bash
squeue -u "$USERNAME"
```

If the manager stops but route jobs have finished or failed, rerun the same
launcher from tmux to resume missing attempts:

```bash
MAX_EVAL_JOBS=8 bash thesis/scripts/launch_baseline_current_full_eval.sh
```

To stop a running evaluation intentionally:

1. Stop the tmux manager with `Ctrl+C`.
2. Cancel remaining route jobs:

```bash
scancel -u "$USERNAME"
```

## 9. Results and Analysis

After the route queue is empty, merge the benchmark outputs:

```bash
export EVAL_RUN_NAME="simlingo_baseline_evalfix_full"
export EVAL_AGENT_NAME="${EVAL_RUN_NAME}"
source thesis/env.sh

python thesis/analysis/merge_bench2drive_results.py \
  -b "${BENCH2DRIVE_ROOT}" \
  > thesis/results/metrics_baseline_evalfix_full.txt

cat thesis/results/metrics_baseline_evalfix_full.txt
```

The committed corrected-baseline summary is:

```text
Driving Score: 86.61 +/- 1.02
Success Rate:  68.18% +/- 0.98%
```

To inspect one route result, use the run index inside the corresponding seed
folder, for example:

```bash
cat "${BENCH2DRIVE_ROOT}/1/res/000_res.json"
```

Scenario comparisons against temporal variants are documented in
`thesis/berzelius_training.md` and use `thesis/analysis/compare_bench2drive_scenarios.py`.

## 10. Render Selected Baseline Videos

Selected baseline videos use the same rendering pipeline as the temporal
variants. To render only the corrected-baseline diagnostic and result-selected
cases:

```bash
cd ${BASE_DIR}/simlingo-thesis
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh

bash thesis/scripts/launch_selected_renders.sh --dry-run baseline
bash thesis/scripts/launch_selected_renders.sh baseline
```

The baseline render manifest is:

```bash
thesis/rendering/manifests/render_manifest_baseline_evalfix_diagnostic_and_result_selected.json
```

Render outputs are written under:

```bash
${BASE_DIR}/eval_results/Bench2Drive/renders/<agent_name>/bench2drive/<seed>/
```

The compressed supplementary videos committed to the repository are served by the
GitHub Pages video player at `thesis/videos/`.

---

## Troubleshooting

**Missing maps:** errors such as `Map 'Town12' not found` or `Map 'Town13' not
found` usually mean the CARLA additional maps were not imported.

**Missing checkpoint:** rerun the pretrained-weight setup and check:

```bash
ls -lh "${MODEL_CKPT}"
```

**Immediate route failures:** inspect the route error logs under
`${BENCH2DRIVE_ROOT}/<seed>/err/`.

**Starting over:** deleting evaluation outputs is destructive. Only remove a run
directory when you intentionally want to rerun it from scratch.
