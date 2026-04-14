#!/bin/bash
#SBATCH -p berzelius
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=carla_test
#SBATCH --output=/proj/berzelius-2023-154/users/%u/logs/carla_test/%j.out
#SBATCH --error=/proj/berzelius-2023-154/users/%u/logs/carla_test/%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
conda activate simlingo
source env.sh

cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen -nosound

