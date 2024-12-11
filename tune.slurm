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
#SBATCH -J tune
#SBATCH -o tune-%u-%j.out
#SBATCH -e tune-%u-%j.err
#SBATCH --mem=256G

CONFIG_FILE=configs/config_attn_unet_segmentation.json
source env.sh
time python tune_segmentation.py -c $CONFIG_FILE
