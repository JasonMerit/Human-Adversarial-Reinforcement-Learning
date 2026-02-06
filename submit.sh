#!/bin/bash
### General options
#BSUB -q hpc
#BSUB -J Tron-DQN
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 1:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo Output_%J.out
#BSUB -eo Output_%J.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning

module purge
module load cuda/12.0

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl

python cuda_checker.py
