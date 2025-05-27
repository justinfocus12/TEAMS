#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-2:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /orcd/archive/pog/001/ju26596/TEAMS/logs/pebr_%A_%a_error.out
#SBATCH -o /dev/null
#SBATCH --mem-per-cpu=8GB

module load deprecated-modules
module load anaconda3
source activate teamsenv
python -u pebr_frierson.py single $SLURM_ARRAY_TASK_ID > /orcd/archive/pog/001/ju26596/TEAMS/logs/pebr_progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out

