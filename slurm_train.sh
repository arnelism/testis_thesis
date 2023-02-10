#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem 24000
#SBATCH --job-name train_testis_model
#SBATCH --output %J_train_log.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com

# Some modules
module load any/python
module load cuda
module load cudnn
source env_thesis/bin/activate

python script2_train_model.py --level=1 --overlap=10 --color_mode=grayscale  --epochs=100 --enable_wb

