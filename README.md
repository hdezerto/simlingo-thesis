# SimLingo Thesis Repository

This repository is my master-thesis fork of SimLingo. The original upstream project README is preserved in `README_UPSTREAM.md`.

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

## Branch Strategy

This repository is organized around a stable baseline branch and separate method branches built on top of it.

Planned branch roles:

- `baseline`: Berzelius-focused reproduction of the SimLingo baseline
- method branches: thesis extensions built on top of the baseline, for example a temporal module or feature-delta variant

Use this repository in two layers:

- Upstream/original project code:
  `Bench2Drive/`, `leaderboard/`, `leaderboard_autopilot/`, `scenario_runner/`, `scenario_runner_autopilot/`, `simlingo_base_training/`, `simlingo_training/`, `team_code/`, `dataset_generation/`, `tools/`
- Thesis/Berzelius additions:
  everything under `thesis/`, plus the small root-level adaptations in `setup_carla.sh` and `start_eval_simlingo.py`

Start here if you want to reproduce the baseline on Berzelius:

- Detailed baseline guide: `thesis/berzelius_baseline.md`
- Training setup guide: `thesis/berzelius_training.md`
- Shared environment setup: `thesis/env.sh`
- Evaluation orchestrator: `start_eval_simlingo.py`

Thesis folder layout:

- `thesis/berzelius_baseline.md`: Berzelius baseline documentation
- `thesis/berzelius_training.md`: Berzelius training setup and smoke-test runbook
- `thesis/analysis/`: result aggregation and failure-analysis utilities
- `thesis/rendering/`: qualitative rendering pipeline and video stitching tools
- `thesis/rendering/manifests/`: manifest files for multi-case rendering runs
- `thesis/results/`: generated reports and committed baseline result artifacts
- `thesis/slurm/`: Berzelius-specific SLURM entry scripts

Important scope:

- The repository contains code and example config artifacts only.
- CARLA binaries, checkpoints, logs, and evaluation outputs are expected outside the repo.
- The upstream/original documentation is preserved in `README_UPSTREAM.md`.
