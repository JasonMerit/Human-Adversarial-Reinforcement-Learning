from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo
from agents.deterministic import DeterministicAgent, Random

env = TronEnv(DeterministicAgent(start_left=True), width=10, height=10)
env = TronEgo(env)
env = TronView(env, fps=100, scale=70)
state, _ = env.reset()

agent = Random(env)
# agent = DeterministicAgent(1)

done = False
total_reward = 0.0
episodes = 1
while True:
    # TronView.view(state, scale=70)
    # action = TronView.wait_for_keypress()
    action = agent.compute_single_action(state)
    # action = 1 

    state, reward, done, _, info = env.step(action)
    if done:
        total_reward += reward
        print(f"{episodes}: Avg reward = {round(total_reward / episodes, 2)}", end='\r')
        env.reset()
        agent.reset()
        episodes += 1