import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time


from environment.env import TronEnv
from environment.wrappers import TronEgo
from agents.deterministic import DeterministicAgent
from agents.dqn import QNet
from utils.buffer import ReplayBuffer

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
    TARGET_UPDATE_STEPS = 10_000
    NUM_EPISODES = 50_000
    NUM_ENVS = 32
    EPS_DECAY_STEPS = NUM_EPISODES * 10  # Trying to hit 50-80% of training time

    # ---------------------
    # Environment
    # ---------------------
    envs = [TronEgo(TronEnv(DeterministicAgent(start_left=True), 10, 10)) for _ in range(NUM_ENVS)]
    num_actions = envs[0].action_space.n
    state_shape = envs[0].observation_space.shape

    states = [None] * NUM_ENVS
    total_rewards = [0.0] * NUM_ENVS
    episode_lengths = [0] * NUM_ENVS
    total_lengths = 0
    completed_returns = 0.0
    for i, env in enumerate(envs):
        s, _ = env.reset()
        states[i] = s

    # ----------------------
    # Network
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    # ---------------------
    # Training Loop
    # ---------------------
    eps = EPS_START
    completed_episodes = 0
    global_step = 0
    last_target_update_step = 0
    t0 = time()

    while completed_episodes < NUM_EPISODES:
        # Build batch of states
        state_batch = torch.from_numpy(np.stack(states)).float().to(device)

        # Epsilon-greedy action selection
        with torch.no_grad():
            q_batch = q_net(state_batch)
            greedy_actions = q_batch.argmax(dim=1).cpu().numpy()

        rand_vals = np.random.rand(NUM_ENVS)
        random_actions = np.random.randint(num_actions, size=NUM_ENVS)
        actions = np.where(rand_vals < eps, random_actions, greedy_actions)

        # Step each environment
        for i, env in enumerate(envs):
            a = int(actions[i])
            next_state, reward, done_flag, _, _ = env.step(a)

            buffer.push(states[i], a, reward, next_state, done_flag)
            episode_lengths[i] += 1

            if done_flag:
                completed_episodes += 1
                completed_returns += total_rewards[i]
                total_lengths += episode_lengths[i]

                s, _ = env.reset()
                next_state = s
                total_rewards[i] = 0.0
                episode_lengths[i] = 0

            states[i] = next_state
            total_rewards[i] += reward

        # --------
        # Learning
        # --------
        if len(buffer) >= BATCH_SIZE:
            s, a, r, s2, d = buffer.sample(BATCH_SIZE)

            s = torch.from_numpy(s).float().to(device)
            s2 = torch.from_numpy(s2).float().to(device)
            a = torch.tensor(a, dtype=torch.long, device=device)
            r = torch.tensor(r, dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device)

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target_net(s2).max(1)[0]
                q_target = r + GAMMA * q_next * (1 - d)

            loss = nn.functional.mse_loss(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------------------------
        # Epsilon update (per step)
        # -------------------------
        global_step += NUM_ENVS
        eps = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS * (EPS_START - EPS_END))

        if global_step - last_target_update_step >= TARGET_UPDATE_STEPS:
            target_net.load_state_dict(q_net.state_dict())
            last_target_update_step = global_step
            print(f"Episode {completed_episodes} | Avg Reward: {completed_returns / completed_episodes:.2f} | Epsilon: {eps:.3f}")

            

    # Save the model
    torch.save(q_net.state_dict(), "q_net.pth")
    torch.save(target_net.state_dict(), "target_net.pth")
    print("\nTraining finished and models saved.")
    print(f"Average episode length: {total_lengths / completed_episodes:.2f}")

        

    print("\033[92mDONE\033[0m")
    print(f"Total training time: {time() - t0:.2f} seconds")