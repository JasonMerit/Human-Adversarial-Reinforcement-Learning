#!/bin/bash
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J BenchMirror[1-5]%5
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
# BSUB -R "rusage[mem=10GB]"
###BSUB -R "select[gpu32gb]"  # For max storage
#BSUB -M 10GB
#BSUB -W 10:00  # 10 hours wall time for 10_000_000
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Out_%I.out
#BSUB -eo rl_core/HPC/Err_%I.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc

echo "====== [$(date)] Starting $LSB_JOBID ($i) ======"
python -m rl_core.train --exp-name $LSB_JOBNAME --job-index $LSB_JOBINDEX --hpc --size 15
echo "====== [$(date)] Finished $LSB_JOBID ($i) ======"