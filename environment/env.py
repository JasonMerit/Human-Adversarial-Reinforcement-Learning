import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import gymnasium as gym

from environment.tron import Tron
from agents import Agent, DQNAgent
from environment.wrappers import TronImage, TronTorch, TronEgo

class TronDualEnv(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    action_flipped = [0, 3, 2, 1]  # Flipping opponent action horizontally
    reward_mapping = [0, -1, 1, 0]  # playing, lose, win, draw

    def __init__(self, width, height):
        self.tron = Tron(width, height)
        self.width = width

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.tron.tick(self.action_mapping[1], self.action_mapping[3])  # First facing
            
        return self._get_state(), {'result': 0}
    
    def step(self, joint_action : tuple):
        assert self.action_space.contains(joint_action), f"Jason! Invalid Action {joint_action}"
        
        dir1 = self.action_mapping[joint_action[0]]
        dir2 = self.action_mapping[self.action_flipped[joint_action[1]]]
    
        result = self.tron.tick(dir1, dir2)
        done = result != 0
        state = self._get_state()
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls, you, opp = self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
        you_ = np.array([self.width - 1 - you[0], you[1]])
        opp_ = np.array([self.width - 1 - opp[0], opp[1]])
        a = np.fliplr(walls).copy()
        a[a != 0] = 3 - a[a != 0]  # Map (1, 2) -> (2, 1)
        return (walls, you, opp), (a, opp_, you_)
    
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

    reward_mapping = [0, -1, 1, .5]  # playing, lose, win, draw

    def __init__(self, opponent : Agent, width, height):
        self.dual_env = TronDualEnv(width, height)
        self.tron = self.dual_env.tron

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space, opp_observation_space = self.dual_env.observation_space.spaces

        self._base_oppenv = TronOppEnv(self.tron, opp_observation_space)
        self.oppenv = self._base_oppenv

        self.opponent = opponent
        if isinstance(opponent, DQNAgent):
            opponent.eval()
            self.oppenv = TronTorch(TronEgo(self.oppenv))
        opponent.bind_env(self.oppenv)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        (state, opponent_state), info = self.dual_env.reset(seed=seed)
        self.opponent.reset()
        self._base_oppenv.set_state(opponent_state)
        self.oppenv.reset(seed=seed)

        return state, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Jason! Invalid Action {action}"
        
        opp_view_state, _, _, _, _ = self.oppenv.step(0)
        opponent_action = self.opponent(opp_view_state)
        print(opponent_action)

        (state, opp_state), reward, done, _, info = self.dual_env.step((action, opponent_action))
        self._base_oppenv.set_state(opp_state)

        # TronView.view(state, scale=70)
        # TronView.view_dual((state, opponent_state), scale=70)
        
        return state, reward, done, False, opp_view_state

if __name__ == "__main__":
    from environment.wrappers import TronView, TronEgo, TronTorch
    from agents import DeterministicAgent, RandomAgent, SemiDeterministicAgent, HeuristicAgent, DQNAgent
    from utils import StateViewer

    # env = TronDualEnv(width=10, height=10)
    # env = TronSingleEnv(DQNAgent("q_net.pth"), width=10, height=10)
    env = TronSingleEnv(SemiDeterministicAgent(.5), width=10, height=10)
    # env = TronEgo(env)
    # env = TronView(env, fps=10, scale=70)

    sv = StateViewer((10, 10), fps=2)

    env = TronTorch(env)
    agent = DQNAgent("q_net.pth")
    agent.eval()

    # agent = SemiDeterministicAgent(.6)
    # agent = HeuristicAgent()
    # agent = RandomAgent()
    agent.bind_env(env)


    state, _ = env.reset()

    done = False
    total_reward = 0.0
    episodes = 1
    kek = 2
    while True:
        # TronView.view(state[0], scale=70)
        # TronView.view_dual(state, scale=70)
        action = agent(state)
        # action = TronView.wait_for_both_inputs()
        # action = TronView.wait_for_keypress()
        # action = env.action_space.sample()
        # action = kek
        kek = 1

        state, reward, done, _, info = env.step(action)
        sv.view_image(state)
        if done:
            kek = 2
            if reward > 0.9:
                total_reward += reward
            # print(f"{episodes}: Avg reward = {round(total_reward / episodes, 2)}", end='\r')
            state, _ = env.reset()
            agent.reset()
            episodes += 1

        # TronView.wait_for_keypress()