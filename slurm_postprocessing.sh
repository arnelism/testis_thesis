#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com
#SBATCH --job-name postproc
#SBATCH --output work/slurm_logs/%J_postproc_log.txt

# Some modules
module load any/python
module load openslide
source env_thesis/bin/activate

echo "Running post-processing"

python script4_post_processing.py
