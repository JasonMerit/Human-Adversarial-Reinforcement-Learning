#!/bin/bash
### General options
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J Tron-DQN
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 3GB
#BSUB -W 1:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo Output.out
#BSUB -eo Output.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning

module purge
module load cuda/12.0

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl

python cuda_check.py
