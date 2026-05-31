#!/bin/bash
###BSUB -q hpc
#BSUB -q gpuv100
#BSUB -J NNSize0[1-5]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
###BSUB -W 8:00  # 8 hours wall time for 4_000_000 steps and size=25
#BSUB -W 8:00  # Sligtly bigger never tried nn sizes
#BSUB -u s216135@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo rl_core/HPC/Out_%I.out
#BSUB -eo rl_core/HPC/Err_%I.err

cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate harl_hpc

PARAMS=(
"8 16 16"
"16 16 16"
"16 16 32"
"16 32 32"
"16 32 64"
)

IDX=0
read CONV1 CONV2 HIDDEN <<< "${PARAMS[$IDX]}"
echo "Running with CONV1=$CONV1, CONV2=$CONV2, HIDDEN=$HIDDEN"

echo "====== [$(date)] Starting $LSB_JOBID ($LSB_JOBINDEX) ======"
python -m rl_core.train --exp-name $LSB_JOBNAME --job-index $LSB_JOBINDEX --hpc --conv1 $CONV1 --conv2 $CONV2 --hidden-size $HIDDEN
echo "====== [$(date)] Finished $LSB_JOBID ($LSB_JOBINDEX) ======"