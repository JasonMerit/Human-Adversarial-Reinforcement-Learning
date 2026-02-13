import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import gymnasium as gym

from environment.tron import Tron

class TronDualEnv(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    reward_mapping = [0, -1, 1, 0]  # playing, lose, win, draw

    def __init__(self, opponent, width, height):
        self.tron = Tron(width, height)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4)))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=2, shape=(height, width), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), shape=(2,), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), shape=(2,), dtype=np.int8)
            )),
            gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=2, shape=(height, width), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), shape=(2,), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), shape=(2,), dtype=np.int8)
            ))
        ))

        self.opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.tron.tick(self.action_mapping[1], self.action_mapping[3])  # First facing
        self.opponent.reset(seed=seed)
            
        return self._get_state(), {'result': 0}
    
    def step(self, joint_action : tuple):
        assert self.action_space.contains(joint_action), f"Jason! Invalid Action {joint_action}"
        
        dir1 = self.action_mapping[joint_action[0]]
        dir2 = self.action_mapping[joint_action[1]]
    
        result = self.tron.tick(dir1, dir2)
        done = result != 0
        state = self._get_state()
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls, you, opp = self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
        return (walls, you, opp), (walls, opp, you)
    
if __name__ == "__main__":
    from environment.wrappers import TronView, TronEgo
    from agents import DeterministicAgent, RandomAgent

    env = TronDualEnv(DeterministicAgent(), width=10, height=10)
    # env = TronEgo(env)
    env = TronView(env, fps=100, scale=70)
    state, _ = env.reset()

    agent = RandomAgent()
    agent.bind_env(env)
    # agent = DeterministicAgent(1)

    done = False
    total_reward = 0.0
    episodes = 1
    while True:
        # TronView.view(state[0], scale=70)
        TronView.view_dual(state, scale=70)
        action = TronView.wait_for_keypress()
        action = env.action_space.sample()

        state, reward, done, _, info = env.step(action)
        if done:
            if reward > 0.9:
                total_reward += reward
            print(f"{episodes}: Avg reward = {round(total_reward / episodes, 2)}", end='\r')
            env.reset()
            agent.reset()
            episodes += 1