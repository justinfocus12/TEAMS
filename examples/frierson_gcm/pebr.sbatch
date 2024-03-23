#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p sched_mit_hill
#SBATCH -e error_%A_%a.out
#SBATCH -o output_%A_%a.out
#SBATCH --mem-per-cpu=2GB

module load anaconda3/2021.11
source activate teamsenv
python -u pebr_frierson.py single $SLURM_ARRAY_TASK_ID > progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out

