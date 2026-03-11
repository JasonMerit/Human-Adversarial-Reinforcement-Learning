
from rl_core.tron_env.tron_env import TronDuoEnv
from rl_core.tron_env.tron_env.utils import StateViewer

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

if __name__ == "__main__":
    env = TronDuoEnv(25)
    agent = RandomAgent(env.action_space)
    sv = StateViewer(size=25, scale=20, fps=2, single=False)

    obs, info = env.reset()
    sv.view_dual(obs)

    while True:
        # action = agent.act(obs)
        action = sv.get_dual_action()
        obs, reward, done, _, info = env.step(action)

        if done:
            print(f"{info['result']}", reward)
            obs, info = env.reset()

        sv.view_dual(obs)