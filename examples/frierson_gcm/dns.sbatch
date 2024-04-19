#!/bin/bash

#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-8:00
#SBATCH -p sched_mit_hill
#SBATCH -e /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/dns_error_%A_%a.out
#SBATCH -o /dev/null
#SBATCH --mem-per-cpu=10GB

module load anaconda3/2021.11
source activate teamsenv
python -u dns_frierson.py single $SLURM_ARRAY_TASK_ID > /net/bstor002.ib/pog/001/ju26596/TEAMS/logs/progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out
