#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-4:00
#SBATCH -p sched_mit_hill
#SBATCH -e error_%j.out
#SBATCH -o output_%j.out
#SBATCH --mem-per-cpu=3GB

module load anaconda3/2021.11
source activate teamsenv
python -u algorithms_lorenz96.py > progress_"$SLURM_JOB_ID".out
