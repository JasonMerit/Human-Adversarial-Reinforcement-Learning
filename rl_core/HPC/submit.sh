#!/bin/bash
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J KnegtMCR[1-5]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -W 8:00  # 8 hours wall time for 4_000_000 steps and size=25
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
python -m rl_core.train --exp-name $LSB_JOBNAME --job-index $LSB_JOBINDEX --hpc --knegt 
echo "====== [$(date)] Finished $LSB_JOBID ($i) ======"