#!/bin/bash
#SBATCH --account=cs6501_sp24
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --reservation=bi_fox_dgx
# #SBATCH -A mlia
# #SBATCH -p gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -J segmentation
#SBATCH -o segmentation-%u-%j.out
#SBATCH -e segmentation-%u-%j.err
#SBATCH --mem=64G

CONFIG_FILE=configs/config_unet_segmentation.json
source ENV/bin/activate
time python train_segmentation.py -c $CONFIG_FILE
