#!/bin/bash

#SBATCH --array=
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J pos_xy
#SBATCH -o .pos_xy-%4a-%j.out
#SBATCH -e .pos_xy-%4a-%j.out

##SBATCH --partition=upex

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004456

# Load modules and environment
source /etc/profile.d/modules.sh
module load exfel exfel-python

# add sample x and y position to events file
python ../add_slowdata.py ${SLURM_ARRAY_TASK_ID} 3 /CONTROL/SPB_IRU_SACT/MOTOR/CHAN0/actualPosition/value sample_pos_mm_x -f
python ../add_slowdata.py ${SLURM_ARRAY_TASK_ID} 3 /CONTROL/SPB_IRU_SACT/MOTOR/CHAN1/actualPosition/value sample_pos_mm_y -f
