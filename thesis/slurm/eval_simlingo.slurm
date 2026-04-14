#!/bin/bash
#SBATCH -p berzelius
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=simlingo_baseline
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/baseline/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/baseline/%j.err

# Email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hugoad@kth.se

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source thesis/env.sh

python start_eval_simlingo.py
