#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-2:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/error_%A_%a.out
#SBATCH -o /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/output_%A_%a.out
#SBATCH --mem-per-cpu=2GB

module load anaconda3/2022.05-x86_64
source activate teamsenv
python -u teams_lorenz96_coldstart.py multiseed $SLURM_ARRAY_TASK_ID > /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out
