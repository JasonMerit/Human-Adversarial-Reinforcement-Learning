#!/bin/bash
###BSUB -q hpc
#BSUB -q gpuv100
###BSUB -J CleanRain
###BSUB -J RainbowSimpler
#BSUB -J CleanRain[1-2]%5  # Job array with 5 tasks - remove the loop in the script if using this
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
# BSUB -R "rusage[mem=10GB]"
###BSUB -R "select[gpu32gb]"  # For max storage
#BSUB -M 10GB
#BSUB -W 1:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Out_%I.out
#BSUB -eo rl_core/HPC/Err_%I.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc

python -m rl_core.clean_rainbow.train --exp-name $LSB_JOBNAME --job-index $LSB_JOBINDEX --hpc
# for i in {1..5}
# do
#     echo "====== [$(date)] Starting $LSB_JOBID ($i) ======"
#     # python -m rl_core.train_ppo --exp-name PPO
#     python -m rl_core.rainbow.train --exp-name $LSB_JOBID --total_timesteps 10_000_000
#     # python -m rl_core.self_train_pool --exp-name Pooling --no-save-model --total_timesteps 10 --num-envs 1
#     echo ""
# done
