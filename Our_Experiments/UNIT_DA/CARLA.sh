#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=48:00:00
#SBATCH --mincpus=12
#SBATCH --nodelist=gnode25
module add cuda/8.0
module add cudnn/7-cuda-8.0

python2 train.py
