#!/bin/bash
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J Rainbow
###BSUB -J Rainbow[1-5]  # Job array with 5 tasks - remove the loop in the script if using this
#BSUB -n 2
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
### BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -M 32GB
#BSUB -W 5:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Output.out
#BSUB -eo rl_core/HPC/Error.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc

for i in {1..1}
do
    echo "====== [$(date)] Starting $LSB_JOBID ($i) ======"
    # python -m rl_core.train_ppo --exp-name PPO
    python -m rl_core.rainbow.train --exp-name $LSB_JOBID --training_frames 10_000_000
    # python -m rl_core.self_train_pool --exp-name Pooling --no-save-model --total_timesteps 10 --num-envs 1
    echo ""
done
