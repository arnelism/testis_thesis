#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=8:00:00
#SBATCH --gres=gpu
#SBATCH --mem 64000
#SBATCH --job-name inference
#SBATCH --output work/slurm_logs/%J_inference_log.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com

# Some modules
module load any/python
module load cuda
module load cudnn
source env_thesis/bin/activate

echo "Environment initialized. Running inference script"

python script3_generate_model_outputs.py
