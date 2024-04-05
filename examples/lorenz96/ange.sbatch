#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-6:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e error_%A_%a.out
#SBATCH -o output_%A_%a.out
#SBATCH --mem-per-cpu=3GB

module load anaconda3/2022.05-x86_64
source activate teamsenv
python -u ange_lorenz96.py single $SLURM_ARRAY_TASK_ID > progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"
