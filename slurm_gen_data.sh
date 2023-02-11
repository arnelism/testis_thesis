#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnelism@gmail.com

# Some modules
module load any/python
module load cuda
module load cudnn
module load openslide
source ../env_thesis/bin/activate


echo "level1, overlap 50"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=1 --tubule_threshold=50 --num_train_images=4000

echo "level1, overlap 30"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=1 --tubule_threshold=30 --num_train_images=4000

echo "level1, overlap 10"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=1 --tubule_threshold=10 --num_train_images=4000

echo "level2, overlap 50"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=2 --tubule_threshold=50 --num_train_images=4000

echo "level2, overlap 30"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=2 --tubule_threshold=30 --num_train_images=4000

echo "level2, overlap 10"
python script1_generate_training_data.py --slidefile=../dataset/19,H,16747,_,01,1,0.mrxs --annotationsfile=annotations/checkpoint.2023-01-31_2258.geojson --folder_prefix=trainonly --level=2 --tubule_threshold=10 --num_train_images=4000



#python script1_generate_training_data.py --slidefile=$1 --annotationsfile=$2 --level=$level --tubule_threshold=$overlap --color_mode=$colormode --epochs=100 --enable_wb --run_mode=single --train_size=2000
