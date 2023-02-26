#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --mem 64000
#SBATCH --job-name train_testis_model
#SBATCH --output work/slurm_logs/%J_train_log.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com

# Some modules
module load any/python
module load cuda
module load cudnn
source env_thesis/bin/activate

python script2_train_model.py
