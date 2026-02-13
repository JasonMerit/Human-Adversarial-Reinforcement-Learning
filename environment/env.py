import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import gymnasium as gym

from environment.tron import Tron

class TronEnv(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=int)  # up, right, down, left
    reward_mapping = [0.0, -1.0, 1, 0.5]  # playing, lose, win, draw

    def __init__(self, opponent, width, height):
        self.tron = Tron(width, height)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(height, width), dtype=np.int64),
            gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.int64),
            gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.int64)
        ))

        self.opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.tron.tick(self.action_mapping[1], self.action_mapping[3])  # First facing
        self.opponent.reset(seed=seed)
            
        return self._get_state(), {'result': 0}
    
    def step(self, action):
        assert self.action_space.contains(action), "Jason! Invalid Action"
        opp_action = self.opponent(self._get_state())
        
        dir1 = self.action_mapping[action]
        dir2 = self.action_mapping[opp_action]
    
        result = self.tron.tick(dir1, dir2)
        done = result != 0
        state = self._get_state()
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        return self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
    
if __name__ == "__main__":
    from environment.wrappers import TronView, TronEgo
    from agents import DeterministicAgent, RandomAgent

    env = TronEnv(DeterministicAgent(), width=10, height=10)
    env = TronEgo(env)
    env = TronView(env, fps=100, scale=70)
    state, _ = env.reset()

    agent = RandomAgent()
    agent.bind_env(env)
    # agent = DeterministicAgent(1)

    done = False
    total_reward = 0.0
    episodes = 1
    while True:
        TronView.view(state, scale=70)
        action = TronView.wait_for_keypress()

        state, reward, done, _, info = env.step(action)
        if done:
            if reward > 0.9:
                total_reward += reward
            print(f"{episodes}: Avg reward = {round(total_reward / episodes, 2)}", end='\r')
            env.reset()
            agent.reset()
            episodes += 1