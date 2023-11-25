#!/bin/bash

#SBATCH --array=
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J powder
#SBATCH -o .pow-%.4a-%j.out
#SBATCH -e .pow-%.4a-%j.out

##SBATCH --partition=upex

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004456

# Load modules and environment
source /etc/profile.d/modules.sh
module load exfel exfel-python

python ../powder.py ${SLURM_ARRAY_TASK_ID} 
