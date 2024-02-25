#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-6:00
#SBATCH -p sched_mit_hill
#SBATCH -e error_%A_%a.out
#SBATCH -o output_%A_%a.out
#SBATCH --mem-per-cpu=3GB

module load anaconda3/2021.11
source activate teamsenv
python algorithms_lorenz96.py $SLURM_ARRAY_TASK_ID 
