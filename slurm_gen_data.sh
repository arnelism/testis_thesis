#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com
#SBATCH --job-name gen_data
#SBATCH --output work/slurm_logs/%J_gendata_log.txt

# Some modules
module load any/python
module load cuda
module load cudnn
module load openslide
source env_thesis/bin/activate

python script1_generate_training_data.py
