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
    from .env import TronEnv, TronView
    from .env import utils
    env = TronEnv()

    print(utils.blue("NOT IMPLEMENTED! FUCK OFF"))
    quit()
    agent = None

    print(eval(env, agent))