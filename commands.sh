## HPC
ssh s216135@login1.gbar.dtu.dk
cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
bsub < HPC/submit.sh

## Home
cd C:\Users\PC\Documents\Code\Human-Adversarial-Reinforcement-Learning
harl\Scripts\activate

## Laptop
cd C:\Users\Jason\Documents\Code\Human-Adversarial-Reinforcement-Learning
harl\Scripts\activate

# Different modules
python -m rl_core.environment.env
python -m rl_core.agents.dqn
python -m server.upload_model

# Run server - visit http://localhost:8000/docs#/ 
uvicorn server.main:app --port 8000

# Find native .dll here
cd C:\Users\Jason\.nuget\packages\microsoft.ml.onnxruntime\1.16.0\runtimes\win-x64\native

# CleanRL
python rl_core/cleanrl/cleanrl/ppo_cnn.py --env-id Tron-v0
