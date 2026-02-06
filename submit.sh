#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J Tron-DQN
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 1:00 
### -- set the email address -- 
#BSUB -u s216135@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o0 Output_%J.out 
#BSUB -e0 Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
module purge
module load cuda/12.0            # or whatever version DTU provides [web:95]
conda activate harl

# python check_cuda.py > output.out