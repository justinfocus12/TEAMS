#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-4:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/dns_%A_%a_error.out
#SBATCH -o /dev/null
#SBATCH --mem-per-cpu=10GB

module load anaconda3
source activate teamsenv
python -u dns_frierson.py single $SLURM_ARRAY_TASK_ID > /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out
