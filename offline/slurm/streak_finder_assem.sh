#!/bin/bash

#SBATCH --array=39
#SBATCH --time=4:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004456
#SBATCH --export=ALL
#SBATCH -J streak_finder_assemble
#SBATCH --output=slog/%j.out
#SBATCH --error=slog/%j.err


PREFIX=/gpfs/exfel/exp/SPB/202302/p004456/usr/Shared/butolama/xfel4456/tests

source /etc/profile.d/modules.sh
module load exfel exfel-python

THRESHOLD=10
MIN_PIX=5
MAX_PIX=1000
MIN_PEAK=5
MASK_FILE="None"
REGION="ALL"

python ${PREFIX}/streak_finder_assem.py --run  ${SLURM_ARRAY_TASK_ID} --thld ${THRESHOLD} --min_pix ${MIN_PIX} --max_pix ${MAX_PIX} --min_peak ${MIN_PEAK} --mask_file ${MASK_FILE} --region ${REGION}  


