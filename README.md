# SimLingo Thesis Repository

This repository is the master's thesis fork of SimLingo used for the temporal
context experiments. The original upstream project README is preserved in
`README_UPSTREAM.md`.

## Thesis Information

| Field | Value |
| --- | --- |
| Thesis title | Temporal Context for Closed-Loop Vision-Language-Action Driving: A Case Study on SimLingo in Bench2Drive |
| Author | Hugo Afonso Dezerto |
| Degree programme | Master's Programme in Systems, Control and Robotics |
| Degree level | MSc Thesis |
| Institution | KTH Royal Institute of Technology |
| Department | EECS |
| Supervisors | Truls Nyberg; Yi Yang; Jesper Ericsson |
| Host company / cooperation partner | TRATON / Scania |
| Examiner | Olov Andersson (EECS) |
| Keywords | autonomous driving; vision-language-action models; temporal context; closed-loop evaluation; Bench2Drive |
| Contact | hugoad@kth.se |
| Academic year / term | 2026 |

## Branch Strategy

The default thesis branch contains both the corrected SimLingo baseline
reproduction and the temporal variants used in the report. Older branches are
kept only as provenance; the thesis results should be reproduced from the
default branch.

## Recommended Use

- Start here for the repository overview and scope.
- Reproduce the corrected SimLingo baseline with `thesis/berzelius_baseline.md`.
- Reproduce temporal training, evaluation, rendering, and analysis with
  `thesis/berzelius_training.md`.
- Inspect supplementary videos at
  https://hdezerto.github.io/simlingo-thesis/thesis/videos/.
- Use `thesis/env.sh` for shared Berzelius paths and `thesis/results/` for
  committed metric summaries, audit reports, and scenario comparisons.
- Refer to `README_UPSTREAM.md` for the original SimLingo documentation.

## Repository Layout

Upstream/original project code:
`Bench2Drive/`, `leaderboard/`, `leaderboard_autopilot/`, `scenario_runner/`,
`scenario_runner_autopilot/`, `simlingo_base_training/`, `simlingo_training/`,
`team_code/`, `dataset_generation/`, and `tools/`.

Thesis-specific additions:
`thesis/`, plus the small root-level adaptations in `setup_carla.sh` and
`start_eval_simlingo.py`.

Thesis folder layout:

- `thesis/berzelius_baseline.md`: corrected baseline evaluation guide
- `thesis/berzelius_training.md`: temporal training and evaluation runbook
- `thesis/analysis/`: result aggregation and failure-analysis utilities
- `thesis/rendering/`: qualitative rendering pipeline and video stitching tools
- `thesis/rendering/manifests/`: manifest files for multi-case rendering runs
- `thesis/results/`: committed metric summaries, scenario comparisons, and audit reports
- `thesis/slurm/`: Berzelius-specific SLURM entry scripts
- `thesis/videos/`: supplementary videos and GitHub Pages player

Important scope:

- The repository contains code, configuration files, committed result summaries,
  audit reports, and supplementary videos.
- CARLA binaries, checkpoints, raw logs, raw evaluation outputs, and extracted
  datasets are expected outside the repo.
- The upstream/original documentation is preserved in `README_UPSTREAM.md`.
