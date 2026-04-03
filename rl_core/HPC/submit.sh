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
#BSUB -W 7:00
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Output.out
#BSUB -eo rl_core/HPC/Error.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc


for i in {1..5}
do
    seed=$RANDOM
    echo "====== [$(date)] Starting run $i with seed $seed ======"
    python -m rl_core.self_train --seed $seed --exp-name BenchMark --environment TronDuo
    echo ""
done

for i in {1..5}
do
    seed=$RANDOM
    echo "====== [$(date)] Starting run $i with seed $seed ======"
    python -m rl_core.self_train --seed $seed --exp-name TwoChannel --environment DefaultTo2Channels
    echo ""
done
### python -m rl_core.self_train
### python -m rl_core.HPC.cuda_check
