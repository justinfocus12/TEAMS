#!/bin/bash

#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH -p sched_mit_hill
#SBATCH -e /orcd/archive/pog/001/ju26596/TEAMS/logs/teams_%A_%a_error.out
#SBATCH -o /dev/null
#SBATCH --mem-per-cpu=2GB

module load deprecated-modules
module load anaconda3
source activate teamsenv
python -u teams_frierson.py single $SLURM_ARRAY_TASK_ID > /orcd/archive/pog/001/ju26596/TEAMS/logs/teams_run_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_progress.out

