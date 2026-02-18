## HPC
ssh s216135@login1.gbar.dtu.dk
cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
bsub < HPC/submit.sh

## Home
conda activate harl
cd C:\Users\PC\Documents\Code\Human-Adversarial-Reinforcement-Learning
python -m environment.env

## Laptop
conda activate harl
cd C:\Users\Jason\Documents\Code\Human-Adversarial-Reinforcement-Learning

# Run environment
python -m rl_core.environment.env

# Run server - visit http://localhost:8000/docs#/ 
uvicorn server.main:app --port 8000

# Find native .dll here
cd C:\Users\Jason\.nuget\packages\microsoft.ml.onnxruntime\1.16.0\runtimes\win-x64\native

