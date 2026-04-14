# SimLingo Baseline on Berzelius

## Thesis Information

| Field | Value |
| --- | --- |
| Thesis title | [pending update] |
| Author | Hugo Afonso Dezerto |
| Degree programme | Master's Programme in Systems, Control and Robotics |
| Degree level | MSc Thesis |
| Institution | KTH Royal Institute of Technology |
| Department | EECS |
| KTH Supervisor | Truls Nyberg |
| External Supervisors (TRATON) | Truls Nyberg (truls.nyberg@scania.com); Carol Yi Yang (carol-yi.yang@scania.com); Jesper Eriksson (jesper.x.eriksson@scania.com) |
| Examiner | Olov Andersson (EECS) |
| Keywords | [update] |
| Contact | hugoad@kth.se |
| Academic year / term | 2026 |
| Last updated | 2026-03-22 |

---

This guide reproduces the SimLingo baseline on Berzelius with CARLA 0.9.15 and Bench2Drive.

Context:
- The original upstream project documentation remains at repository root in `README_UPSTREAM.md`.
- This guide documents the Berzelius-specific baseline branch layout and workflow.

Important scope:
- The repository contains code only.
- CARLA binaries, pretrained weights, logs, and results live outside the repo.

## Index

- [Thesis Information](#thesis-information)
- [0. Set Variables (recommended)](#0-set-variables-recommended)
- [1. Expected Layout](#1-expected-layout)
- [2. Install CARLA 0.9.15](#2-install-carla-0915)
    - [2.1 Download and extract](#21-download-and-extract)
    - [2.2 Install Additional Maps (required for Bench2Drive)](#22-install-additional-maps-required-for-bench2drive)
- [3. Code and Python Environment](#3-code-and-python-environment)
- [4. Pretrained Weights](#4-pretrained-weights)
- [5. Environment Variables](#5-environment-variables)
- [6. Smoke Test CARLA](#6-smoke-test-carla)
- [7. Run Baseline Orchestrator](#7-run-baseline-orchestrator)
- [8. Monitor and Stop](#8-monitor-and-stop)
- [9. Results and Analysis](#9-results-and-analysis)
- [10. Render Multiple Videos (Manifest-based)](#10-render-multiple-videos-manifest-based)
    - [10.1 Edit the case list](#101-edit-the-case-list)
    - [10.2 Prepare environment (login node)](#102-prepare-environment-login-node)
    - [10.3 (Optional) Dry-run job generation](#103-optional-dry-run-job-generation)
    - [10.4 Submit all render jobs](#104-submit-all-render-jobs)
    - [10.5 Monitor](#105-monitor)
- [Troubleshooting / Notes](#troubleshooting--notes)

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
git checkout baseline
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

## 7. Run Baseline Orchestrator

Do not run baseline with direct `sbatch`. Use the manager script:

```bash
cd ${BASE_DIR}/simlingo-thesis
tmux new -s baseline
```

If session already exists:

```bash
tmux attach -t baseline
```

Inside tmux:

```bash
# 1. Load Environment
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh

# 2. Set Parallelism (Limit to 8 concurrent jobs = 1 full node)
echo "8" > max_num_jobs.txt

# 3. Start the Orchestrator
python start_eval_simlingo.py
```

Notes:
- The run is approximately 660 jobs (220 routes x 3 seeds).
- Output like `660` printed repeatedly is normal debug output from queue length, not immediate failure.

Detach tmux and leave running:
- Press `Ctrl+B`, then `D`.

Reattach later:

```bash
tmux attach -t baseline
```

## 8. Monitor and Stop

Monitor cluster jobs:

```bash
squeue -u "$USERNAME"
```

When you want to stop all running baseline jobs:

1. In tmux, stop manager with `Ctrl+C`.
2. Cancel jobs:

```bash
scancel -u "$USERNAME"
```

## 9. Results and Analysis

After queue is empty and all jobs are done:

First, merge and summarize the results for each seed:
```bash
python thesis/analysis/merge_bench2drive_results.py -b ${BENCH2DRIVE_ROOT}
```

Then, run the scenario failure analysis to generate tables of systematic and hardest scenario failures:
```bash
python thesis/analysis/analyze_scenario_failures.py -b ${BENCH2DRIVE_ROOT}
```

By default, the generated report is written to:
```bash
thesis/reports/scenario_failure_report.txt
```

A sample report artifact is kept in:
```bash
thesis/reports/scenario_failure_report_example.txt
```

To inspect a specific benchmark run result, change the run index (e.g., `000`) as needed:
```bash
cat ${BENCH2DRIVE_ROOT}/1/res/000_res.json
```

**Expected Performance:**

* **Driving Score (DS):** ≈ 85 ± 1

## 10. Render Multiple Videos (Manifest-based)

To render a list of selected route/seed cases, use the manifest-driven pipeline.

### 10.1 Edit the case list

Start from the template manifest:
```bash
thesis/rendering/manifests/render_manifest_template.json
```

You can also inspect the committed example manifests:

- `thesis/rendering/manifests/render_manifest_1.json`
- `thesis/rendering/manifests/render_manifest_2.json`

Add all relevant route/seed pairs under `cases`.

Case fields:
- `route_id` (required): route number as string or integer.
- `seed` (required): traffic-manager seed (integer).

Render/debug controls (in `defaults`, or per case):
- `debug_viz` (bool):
    - `true` => generate debug-overlay frames and stitch MP4.
    - `false` => metrics-only run (no frame generation, no MP4 stitching).
- `debug_stride` (int, >=1): save one debug frame every N simulator steps.
- `debug_save_language` (bool): include prompt/answer text panel in debug overlays.

Optional notification fields (in `defaults`, or per case):
- `mail_user`: email address for SLURM notifications.
- `mail_type`: SLURM mail types (default in script generator: `END,FAIL`).

Example:
```json
{
    "route_id": "2201",
    "seed": 1
}
```

Example with notifications in `defaults`:
```json
"mail_user": "your.name@liu.se",
"mail_type": "END,FAIL"
```

Example with debug rendering enabled:
```json
"debug_viz": true,
"debug_stride": 5,
"debug_save_language": true
```

### 10.2 Prepare environment (login node)

```bash
cd ${BASE_DIR}/simlingo-thesis
module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh
```

### 10.3 (Optional) Dry-run job generation

```bash
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_template.json --dry-run
```

### 10.4 Submit all render jobs

```bash
python thesis/rendering/submit_render_jobs.py --manifest thesis/rendering/manifests/render_manifest_template.json
```

### Current execution behavior

- The render pipeline is frame-based (no CARLA replay):
    1) run evaluator for the route subset,
    2) save multiview frame folders (`rgb_front/`, `rgb_left/`, `rgb_right/`, `rgb_rear/`) + `meta/`,
    3) stitch MP4 from those frames.
- If `debug_viz=false`: the job still runs evaluation, but skips frame generation and MP4 stitching (metrics-only mode).
- If `debug_viz=true`: the job generates debug-overlay frames and stitches MP4.
- Stitching FPS is auto-inferred from frame index spacing in `thesis/rendering/generate_video_from_frames.py` (with `--sim-fps` from manifest `defaults.fps`).


### 10.5 Monitor

```bash
squeue -u "$USERNAME"
```

Each case writes logs to:
```bash
${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive/<SEED>/out/render_route<ROUTE_ID>_seed<SEED>.out
${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive/<SEED>/err/render_route<ROUTE_ID>_seed<SEED>.err
```

### Output layout

For each seed, outputs are written under:
```bash
${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive/<SEED>/
```

Main artifacts:
```bash
/run/render_route<ROUTE_ID>_seed<SEED>.slurm
/res/rendering_route<ROUTE_ID>_seed<SEED>_res.json
/recordings/RouteScenario_<ROUTE_ID>_seed<SEED>_rep0.mp4   # only when debug_viz=true
/frames/route<ROUTE_ID>_seed<SEED>/rgb_front/*.jpg          # only when debug_viz=true
/frames/route<ROUTE_ID>_seed<SEED>/rgb_left/*.jpg           # only when debug_viz=true
/frames/route<ROUTE_ID>_seed<SEED>/rgb_right/*.jpg          # only when debug_viz=true
/frames/route<ROUTE_ID>_seed<SEED>/rgb_rear/*.jpg           # only when debug_viz=true
/frames/route<ROUTE_ID>_seed<SEED>/meta/*.json              # only when debug_viz=true
/out/render_route<ROUTE_ID>_seed<SEED>.out
/err/render_route<ROUTE_ID>_seed<SEED>.err
```

Naming note:
- Benchmark aggregate runs keep index-based files such as `<RUN_INDEX_PAD3>_res.json` (e.g., `000_res.json`).
- Manifest render jobs use `rendering_route<ROUTE_ID>_seed<SEED>_res.json` to avoid confusion between benchmark run index and actual route id.

Inspect a manifest render result example:
```bash
cat ${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive/1/res/rendering_route2201_seed1_res.json
```

---

## Troubleshooting / Notes

* **Immediate Job Crashes:** If jobs fail instantly, check the error logs for a specific route:
```bash
cat ${BASE_DIR}/eval_results/Bench2Drive/simlingo/bench2drive/1/err/render_route2201_seed1.err
```

* **Empty folders under `viz/` with run numbers (`000`, `001`, ...):**
    These are placeholder folders from the evaluator-style save path convention (run index based). For manifest rendering, actual artifacts are written into route-named paths (for example, `viz/2201/RouteScenario_.../debug_viz/...`) and frame/video outputs are under `frames/` and `recordings/`. Empty numeric `viz` folders are safe to ignore or delete.

* **Delete All Results (if needed):** Destructive and irreversible. Only run this if you explicitly want to remove all evaluation outputs and start fresh:
```bash
rm -rf ${BASE_DIR}/eval_results/Bench2Drive
```
