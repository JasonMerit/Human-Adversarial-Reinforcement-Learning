## HPC
ssh s216135@login1.gbar.dtu.dk
cd /zhome/8e/9/169771/Human-Adversarial-Reinforcement-Learning
bsub < HPC/submit.sh

## Home
conda activate harl
cd C:\Users\PC\Documents\Code\Human-Adversarial-Reinforcement-Learning
cd C:\Users\Jason\Documents\Code\Human-Adversarial-Reinforcement-Learning
python -m environment.env

# Run server
conda activate harl
cd C:\Users\Jason\Documents\Code\Human-Adversarial-Reinforcement-Learning\server
uvicorn app.main:app --port 8000