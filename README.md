# SimLingo Thesis Baseline Branch

This branch packages a Berzelius-focused reproduction of the SimLingo baseline and keeps the original upstream project README in `README_UPSTREAM.md`.

Use this repository in two layers:

- Upstream/original project code:
  `Bench2Drive/`, `leaderboard/`, `leaderboard_autopilot/`, `scenario_runner/`, `scenario_runner_autopilot/`, `simlingo_base_training/`, `simlingo_training/`, `team_code/`, `dataset_generation/`, `tools/`
- Thesis/Berzelius additions:
  everything under `thesis/`, plus the small root-level adaptations in `setup_carla.sh` and `start_eval_simlingo.py`

Start here if you want to reproduce the baseline on Berzelius:

- Detailed baseline guide: `thesis/docs/berzelius_baseline.md`
- Shared environment setup: `thesis/env.sh`
- Evaluation orchestrator: `start_eval_simlingo.py`

Thesis folder layout:

- `thesis/docs/`: Berzelius baseline documentation
- `thesis/analysis/`: result aggregation and failure-analysis utilities
- `thesis/rendering/`: qualitative rendering pipeline and video stitching tools
- `thesis/rendering/manifests/`: manifest files for multi-case rendering runs
- `thesis/reports/`: generated reports and kept sample report artifacts
- `thesis/slurm/`: Berzelius-specific SLURM entry scripts

Important scope:

- The repository contains code and example config artifacts only.
- CARLA binaries, checkpoints, logs, and evaluation outputs are expected outside the repo.
- The upstream/original documentation is preserved in `README_UPSTREAM.md`.
