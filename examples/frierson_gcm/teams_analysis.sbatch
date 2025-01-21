#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-3:00
#SBATCH --exclude=node2422
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /orcd/archive/pog/001/ju26596/TEAMS/logs/teams_%A_%a_error.out
#SBATCH -o /dev/null
#SBATCH --mem-per-cpu=8GB

module load anaconda3
source activate teamsenv
python -u teams_frierson.py multiseed $SLURM_ARRAY_TASK_ID > /orcd/archive/pog/001/ju26596/TEAMS/logs/teams_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_progress.out

