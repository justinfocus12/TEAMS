#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-2:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/lorenz96_teams_%A_%a_error.out
#SBATCH -o /dev/null 
#SBATCH --mem-per-cpu=32GB

module load anaconda3
source activate teamsenv
python -u teams_lorenz96_coldstart.py single $SLURM_ARRAY_TASK_ID > /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/lorenz96_teams_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_output.out
