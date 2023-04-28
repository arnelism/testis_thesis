#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=35:59:00
#SBATCH --gres=gpu:tesla
#SBATCH --mem 64000
#SBATCH --job-name train_testis_model_light
#SBATCH --output work/slurm_logs/%J_train_log.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com

# Some modules
module load any/python
module load cuda
module load cudnn
source env_thesis/bin/activate

echo "Environment initialized. Running training script"

batch_size=32 python script2_train_model.py
