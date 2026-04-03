#!/bin/bash
### General options
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J Tron-DQN
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -M 3GB
#BSUB -W 3:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Output.out
#BSUB -eo rl_core/HPC/Error.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning

module purge
nvidia-smi
### module load cuda/12.0

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc

python -m rl_core.self_train --gamma 1.0
### python -m rl_core.HPC.cuda_check
