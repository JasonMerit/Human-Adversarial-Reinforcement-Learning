def eval(env, agent, episodes=10_000):
    agent.eval()
    state, _ = env.reset()
    
    total_reward = 0.0
    for i in range(episodes):

        done = False
        while not done:
            q_values = agent.forward(state)
            action = q_values.argmax().item()

            state, reward, done, _, _ = env.step(action)

            if done:
                state, _ = env.reset()
                
                if reward > 0.0:  # Only count draws (.5) and wins (1.0)
                    total_reward += reward

                if i % 1000 == 0:
                    print(f"Episode {i}, Average Reward: {total_reward / (i + 1):.4f}")

    return total_reward / episodes

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from environment.env import TronEnv
    from environment.wrappers import TronView, TronEgo, TronTorch
    from agents.deterministic import DeterministicAgent
    from train import QNet

    agent = QNet.load("q_net.pth")
    env = TronEnv(DeterministicAgent(), width=10, height=10)
    env = TronEgo(env)
    env = TronTorch(env)

    print(eval(env, agent))