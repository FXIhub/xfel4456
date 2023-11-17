#!/bin/bash

#SBATCH --array=
#SBATCH --time=04:00:00
#SBATCH --partition=upex
#SBATCH --export=ALL
#SBATCH -J streak_finder
#SBATCH -o .streak-%4a-%j.out
#SBATCH -e .streak-%4a-%j.out

####SBATCH --partition=upex-beamtime
####SBATCH --reservation=upex_004456

# Load modules and environment
source /etc/profile.d/modules.sh
module load exfel exfel-python

python ../streak_finder.py ${SLURM_ARRAY_TASK_ID}
python ../add_pulsedata.py ${SLURM_ARRAY_TASK_ID}

