#!/bin/bash

#SBATCH --array=40
#SBATCH --time=04:00:00
#SBATCH --partition=upex
##SBATCH --reservation=upex_004456
#SBATCH --export=ALL
#SBATCH -J streak_finder
#SBATCH -o .%A_%a.out
#SBATCH -e .%A_%a.out

# Change the runs to process using the --array option on line 3

# Load modules and environment
source /etc/profile.d/modules.sh
module load exfel exfel-python

python ../streak_finder.py ${SLURM_ARRAY_TASK_ID} --ADU_per_photon 8 --percentile_threshold 99.5 --min_pix 15



