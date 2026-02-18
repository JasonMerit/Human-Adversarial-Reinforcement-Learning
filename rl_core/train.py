import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

import yaml
from rl_core import TronSingleEnv, TronDualEnv
from rl_core.environment.wrappers import (TronView, TronDualImage, 
                                    TronEgo, TronImage, 
                                    TronDualEgo)
from rl_core.agents import (DeterministicAgent, RandomAgent, SemiDeterministicAgent, 
                    HeuristicAgent, DQNAgent, DQNSoftAgent)
from rl_core.utils import StateViewer
from rl_core.agents.dqn import QNet
from rl_core.utils.buffer import ReplayBuffer
from rl_core.utils.helper import bcolors

if __name__ == "__main__":
    # ---------------------
    # Hyperparameters
    # ---------------------
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    NUM_EPISODES = 500

    # ---------------------
    # Environment
    # ---------------------
    with open("rl_core/config.yml", "r") as f:
        config = yaml.safe_load(f)
    single = config.get("single", True)
    size = tuple(config.get("grid"))

    env = TronSingleEnv(SemiDeterministicAgent(.5), size)
    env = TronEgo(TronImage(env))
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape
    print(f"Defining network with input shape: {bcolors.OKCYAN}{state_shape}{bcolors.ENDC}, and num actions: {bcolors.OKCYAN}{num_actions}{bcolors.ENDC}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START

    # ---------------------
    # Training Loop
    # ---------------------
    t0 = time()
    pbar = tqdm(range(NUM_EPISODES), desc="Training")
    for episode in pbar:
        state,_ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if random.random() < eps:
                action = np.random.randint(num_actions)
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            buffer.push(state.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done)

            state = next_state_tensor
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample_torch(BATCH_SIZE)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    q_target = r + GAMMA * q_next * (1 - d)
                loss = nn.functional.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(EPS_END, eps * EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        pbar.set_postfix({"Episode Reward": total_reward, "Epsilon": eps})


    # Save the model
    torch.save(q_net.state_dict(), "q_net.pth")
    torch.save(target_net.state_dict(), "target_net.pth")
    print("\nTraining finished and models saved.")
    print("\033[92mDONE\033[0m")
    print(f"Total training time: {time() - t0:.2f} seconds")