#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:tesla
#SBATCH --nodelist=Falcon[4-6]
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

echo "Environment initialized. Running training script"

python batch_size=32 script2_train_model.py
