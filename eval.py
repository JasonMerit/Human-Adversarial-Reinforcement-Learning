import torch

def f(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

def eval(env, agent, episodes=10_000):
    agent.eval()
    state, _ = env.reset()
    state = f(state)
    
    total_reward = 0.0
    for _ in range(episodes):

        done = False
        while not done:
            q_values = agent.forward(state)
            action = q_values.argmax().item()

            state, reward, done, _, _ = env.step(action)
            state = f(state)

            if done:
                state, _ = env.reset()
                state = f(state)
                
                if reward > 0.0:  # Only count draws (.5) and wins (1.0)
                    total_reward += reward

    return total_reward / episodes