## HPC
ssh s216135@login1.gbar.dtu.dk
cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
bsub < rl_core/HPC/submit.sh

## Home
cd C:\Users\PC\Documents\Code\Human-Adversarial-Reinforcement-Learning
harl\Scripts\activate

## Laptop
cd C:\Users\Jason\Documents\Code\Human-Adversarial-Reinforcement-Learning
.venv\Scripts\activate

# Different modules
python -m rl_core.self_train
python -m rl_core.sweep

# Run server - visit http://localhost:8000/docs#/ 
uvicorn server.main:app --port 8000

# Find native .dll here
cd C:\Users\Jason\.nuget\packages\microsoft.ml.onnxruntime\1.16.0\runtimes\win-x64\native

# CleanRL
python rl_core/cleanrl/cleanrl/ppo_cnn.py --env-id Tron-v0


python -m rl_core.self_train --no-save-model --num-envs 1
python -m rl_core.self_play runs/self_train_4
python -m rl_core.self_play_onnx runs/self_train_4
python -m rl_core.upload runs/self_train_4/adversary.pth
python -m rl_core.battle runs/self_train_6 runs/self_train_gamma_0
python -m rl_core.battle runs\BenchMark_2 runs\BenchMark_2


conda env create -f environment.yml