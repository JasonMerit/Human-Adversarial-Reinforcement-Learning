import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import gymnasium as gym

from .tron import Tron
from rl_core.agents import Agent


class TronDualEnv(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    reward_mapping = [0, 1, -1, 0]  # draw, bike2 crash, bike1 crash, playing

    def __init__(self, size):
        self.tron = Tron(size)
        self.width, height = size

        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4)))
        self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=2, shape=(height, self.width), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width-1, height-1]), shape=(2,), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width-1, height-1]), shape=(2,), dtype=np.int8)
            ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
            
        return self._get_state(), {'result': 0}
    
    def step(self, joint_action : tuple):
        assert self.action_space.contains(joint_action), f"Jason! Invalid Action {joint_action}"
        
        dir1 = self.action_mapping[joint_action[0]]
        dir2 = self.action_mapping[joint_action[1]]
    
        result = self.tron.tick(dir1, dir2)
        done = result != -1
        state = self._get_state()
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls, you, opp = self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
        return walls, you, opp
    
class TronOppEnv(gym.Env):

    def __init__(self, tron, observation_space):
        self.tron = tron  # Used by TronImage
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = observation_space
        self.current_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.current_obs, {'result': 0}

    def step(self, action):
        return self.current_obs, 0.0, False, False, {'result': 0}
    
    def set_state(self, state):
        self.current_obs = state
    
    def get_state(self):
        return self.current_obs
    
class TronSingleEnv(gym.Env):

    def __init__(self, human : Agent, size):
        self.dual_env = TronDualEnv(size)
        self.tron = self.dual_env.tron

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = self.dual_env.observation_space

        self.human = human
        human.bind_env(self)
        self.last_state = None  # For human to observe the state before its action in step()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.dual_env.reset(seed=seed)
        
        self.human.reset()
        self.last_state = state

        return state, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Jason! Invalid Action {action}"
        
        human_action = self.human(self.last_state)

        state, reward, done, _, info = self.dual_env.step((human_action, action))
        
        self.last_state = state  # Flip state for opponent
        return state, reward, done, False, info

if __name__ == "__main__":
    import yaml
    from rl_core.environment.wrappers import (TronView, TronDualImage, 
                                      TronEgo, TronImage, 
                                      TronDualEgo)
    from rl_core.agents import (DeterministicAgent, RandomAgent, SemiDeterministicAgent, 
                        HeuristicAgent, DQNAgent, DQNSoftAgent)
    from rl_core.utils import StateViewer

    with open("rl_core/config.yml", "r") as f:
        config = yaml.safe_load(f)
    single = config.get("single", True)
    size = tuple(config.get("grid"))

    if single:
        env = TronSingleEnv(HeuristicAgent(), size)
        # env = TronSingleEnv(SemiDeterministicAgent(.5), size)
        # env = TronEgo(TronImage(env))
    else:
        env = TronDualEnv(size)
        env = TronDualEgo(TronDualImage(env))

    env = TronView(env, fps=10, scale=70)
    sv = StateViewer(size, fps=1, single=single)

    # agent = DQNSoftAgent("q_net.pth")
    # agent.eval()
    agent = SemiDeterministicAgent(.6)
    # agent = HeuristicAgent()
    # agent = RandomAgent()
    agent.bind_env(env)
    state, _ = env.reset()

    done = False
    total_reward = 0.0
    episodes = 1
    while True:
        # TronView.view(state[0], scale=70)
        # sv.view_dual(state)
        # action = TronView.wait_for_both_inputs()
        
        if single:
            action = agent(state)
        else:
            action = agent(state[0]), agent(state[1])


        state, reward, done, _, info = env.step(action)
        # sv.view_image(state)
        if done:
            if reward > 0.9:
                total_reward += reward
            state, _ = env.reset()
            agent.reset()
            episodes += 1

        # TronView.wait_for_keypress()