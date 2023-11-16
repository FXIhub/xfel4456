#!/bin/bash

#SBATCH --array=40
#SBATCH --time=04:00:00
#SBATCH --partition=upex
##SBATCH --reservation=upex_004456
#SBATCH --export=ALL
#SBATCH -J powder
#SBATCH -o .%A_%a.out
#SBATCH -e .%A_%a.out

# Change the runs to process using the --array option on line 3

# Load modules and environment
source /etc/profile.d/modules.sh
module load exfel exfel-python

python ../powder.py ${SLURM_ARRAY_TASK_ID} 


