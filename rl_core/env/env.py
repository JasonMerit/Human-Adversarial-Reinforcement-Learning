import os, time

from rich import print
import numpy as np
import gymnasium as gym

from .tron import Tron, Result
from rich import print
from . import utils
from .heuristic import get_best_action
# from .wrappers import encode_observation
# from rl_core.agents import Agent

clear = lambda: os.system('cls')


class TronEnv(gym.Env):
    """Wraps TronEnvBase with all the wrappers"""

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    # reward_dict = { Result.DRAW: -.5, Result.BIKE2_CRASH: -1, Result.BIKE1_CRASH: 1, Result.PLAYING: 0 }
    reward_dict = { Result.DRAW: 0, Result.BIKE2_CRASH: -1, Result.BIKE1_CRASH: 1, Result.PLAYING: 0 }
    # reward_dict = { Result.DRAW: -1, Result.BIKE2_CRASH: -1, Result.BIKE1_CRASH: 1, Result.PLAYING: .01 }

    def __init__(self, size=25, adv_policy: callable =None, render=False):
        self.tron = Tron(size)
        self.size = size
        if render:
            os.system('cls')
        self.render = render


        self.obs_shape = (3, size, size)
        self.n_actions = 3  # up, right, down, left
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.heading1, self.heading2 = 1, 3  # First facing eachother, bike1 goes right, bike2 goes left
            
        return TronEnv.encode(self.state), {'result': 0, 'state': self.state}
    
    def step(self, action : int):
        assert isinstance(action, int) or isinstance(action, np.int8), f"expected int, got {type(action)}"
        assert self.action_space.contains(action), f"[bold red]Jason! Invalid Action {action}"
        
        self.heading1 = self._adv_act(*self.state)
        self.heading2 = (self.heading2 + (action - 1)) % 4  # Because (left, forward, right)
    
        result = self.tron.tick(self.action_mapping[self.heading1], self.action_mapping[self.heading2])
        done = result != Result.PLAYING
        reward = self.reward_dict[result]
        info = {'result': result, 'state': self.state}

        if self.render:
            self.view()
        return TronEnv.encode(self.state), reward, done, False, info
    
    def set_state(self, state):
        tron = self.tron
        # tron.walls, tron.bike1.pos, tron.bike2.pos, self.heading2 = state
        walls, pos1, pos2, h1, h2 = state
        tron.walls, tron.pos1, tron.pos2 = walls.copy(), pos1.copy(), pos2.copy()
        self.heading1, self.heading2 = h1, h2

    @staticmethod
    def encode(state):
        assert type(state) == tuple
        assert len(state) == 5

        walls, bike1, bike2, _, heading2 = state
        occ = (walls > 0).astype(np.float32)

        you = np.zeros_like(occ)
        other = np.zeros_like(occ)

        x, y = bike1
        you[y, x] = 1.0

        x, y = bike2
        other[y, x] = 1.0

        obs = np.stack([occ, you, other], axis=0)

        return np.rot90(obs, k=heading2, axes=(1, 2)).copy()

    @property
    def state(self):
        return self.tron.walls.copy(), self.tron.pos1.copy(), self.tron.pos2.copy(), self.heading1, self.heading2
    
    def sample_action(self):
        return np.random.randint(self.n_actions)
    
    def _adv_act(self, walls, bike1, bike2, head1, head2) -> int:
        # Search if action is valid, if not take best action, with some randomness
        candidates = [0, 1, 2, 3]
        candidates.remove((head1 + 2) % 4)  # Can't turn back

        for action in candidates:
            new_pos = bike1 + self.action_mapping[action]
            if new_pos[0] < 0 or new_pos[0] >= self.size or \
                  new_pos[1] < 0 or new_pos[1] >= self.size:
                continue
            if not walls[new_pos[1], new_pos[0]]:  # If not hit
                return action
            
        return 1
    
    def view(self):
        # os.system('cls')
        # print("\033[H", end="")  # move cursor to top (terminal animation)
        board = np.full((self.size, self.size), ".", dtype=str)
        board[self.tron.walls > 0] = "#"
        x1, y1 = self.tron.pos1
        x2, y2 = self.tron.pos2
        board[y1, x1] = "A"
        board[y2, x2] = "B"
        for row in board:
            print(" ".join(row))


        time.sleep(0.4)
    

# class TronEnv(gym.Env):
#     """Wraps TronEnvBase with all the wrappers"""

#     def __init__(self, size=25, render=False):
#         super().__init__()
#         from .wrappers import TronImage, TronEgo
#         env = TronEgo(TronImage(TronEnvBase(size, render=render)))
#         self.env = env
#         self.tron = env.unwrapped.tron
#         self.action_space = env.action_space
#         self.observation_space = env.observation_space

#         self.n_actions = 3

#     def reset(self, seed=None, options=None):
#         return self.env.reset(seed=seed, options=options)

#     def step(self, action):
#         return self.env.step(action)
    
#     def sample_action(self):
#         return np.random.randint(self.n_actions)

class TronEnvBase(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    reward_dict = { Result.DRAW: -1, Result.BIKE2_CRASH: -1, Result.BIKE1_CRASH: 1, Result.PLAYING: .01 }

    def __init__(self, size=25, render=False):
        self.tron = Tron(size)
        self.size = size

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=2, shape=(size, size), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([size-1, size-1]), shape=(2,), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([size-1, size-1]), shape=(2,), dtype=np.int8)
            ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.heading1, self.heading2 = 1, 3  # First facing eachother, bike1 goes right, bike2 goes left
            
        return self._get_state(), {'result': 0}
    
    def step(self, action : int):
        assert self.action_space.contains(action), f"[bold red]Jason! Invalid Action {action}"
        
        # dir1 = self.action_mapping[self._semi_random_act(*self._get_state())]  # Human's action
        dir1 = self.action_mapping[get_best_action(self._get_state())]  # Human's action
        dir2 = self.action_mapping[action]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        state = self._get_state()
        reward = self.reward_dict[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls, you, opp = self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
        return walls, you, opp

    def _semi_random_act(self, walls, bike1, bike2):
        # Search if action is valid, if not take best action, with some randomness
        candidates = [0, 1, 2, 3]
        candidates.remove((self.heading2 + 2) % 4)  # Can't turn back

        for action in candidates:
            new_pos = bike2 + self.action_mapping[action]
            if not self.tron.bike2.is_hit(walls, *new_pos):
                return action
        
        raise Exception("Jason! No valid moves for the human player - this should never happen")
class TronDuoEnv(gym.Env):
    """
    TronEnv with both players controlled by the same agent. Action is a tuple of (action1, action2)
    Rewards assume bike1 is the "main" agent and bike2 is the adversary, so reward is positive if bike1 wins and negative if bike2 wins.
    """
    
    reward_dict = { Result.DRAW: 0, Result.BIKE2_CRASH: 1, Result.BIKE1_CRASH: -1, Result.PLAYING: 0 }

    def __init__(self, size=25):
        super().__init__()
        self.tron = Tron(size)
        self.size = size
        self.action_space = gym.spaces.MultiDiscrete([3, 3])  # (left, forward, right) for each bike relative to their current heading
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 3, size, size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.heading1, self.heading2 = 1, 3  # First facing eachother, bike1 goes right, bike2 goes left
            
        return self._obs(), {'result': 0}
    
    def step(self, action : np.ndarray):
        assert self.action_space.contains(action), f"[bold red]Jason! Invalid Action {action}"

        # a0, a1 = int(action[0]), int(action[1])

        self.heading1 = (self.heading1 + (action[0] - 1)) % 4  # Because (left, forward, right)
        self.heading2 = (self.heading2 + (action[1] - 1)) % 4  # Because (left, forward, right)
        
        dir1 = TronEnvBase.action_mapping[self.heading1]
        dir2 = TronEnvBase.action_mapping[self.heading2]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        state = self._obs()
        reward = self.reward_dict[result]

        info = {"result": result} if done else {}
        return state, reward, done, False, info
    
    def _obs(self):
        # First encode to one hot image
        walls, bike1, bike2 = encode_observation(self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos)
        obs1 = (walls, bike1, bike2)
        obs2 = (walls, bike2, bike1)

        # Then rotate according to heading so that bikes always face up
        obs1 = np.rot90(obs1, k=self.heading1, axes=(1, 2)).copy()  # Copy to remove negative stride
        obs2 = np.rot90(obs2, k=self.heading2, axes=(1, 2)).copy()  # Copy to remove negative stride

        obs = np.stack([obs1, obs2], axis=0)
        assert obs.shape == self.observation_space.shape, utils.red(f"Jason! Obs shape mismatch {obs1.shape} vs {self.observation_space.shape}")
        return obs

class TronCoreEnv(gym.Env):
    """TronEnv agnostic to state representation - for comparing different state representations"""

    def __init__(self, size=25):
        super().__init__()
        self.tron = Tron(size)
        self.size = size
        self.action_space = gym.spaces.MultiDiscrete([3, 3])  # (left, forward, right) for each bike relative to their current heading
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 3, size, size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.heading1, self.heading2 = 1, 3  # First facing eachother, bike1 goes right, bike2 goes left
            
        return self._get_state(), {'result': 0}
    
    def step(self, action : np.ndarray):
        assert self.action_space.contains(action), utils.red(f"Jason! Invalid Action {action}")

        self.heading1 = (self.heading1 + (action[0] - 1)) % 4  # Because (left, forward, right)
        self.heading2 = (self.heading2 + (action[1] - 1)) % 4  # Because (left, forward, right)
        
        dir1 = TronEnvBase.action_mapping[self.heading1]
        dir2 = TronEnvBase.action_mapping[self.heading2]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        state = self._get_state()

        info = {"result": result} if done else {}
        return state, 0, done, False, info
    
    def _get_state(self):
        return self.heading1, self.heading2, self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos

def encode_observation_2channel(walls, bike1, bike2):
    occ = (walls > 0).astype(np.float32)

    bikes = np.zeros_like(occ)

    x, y = bike1
    bikes[y, x] = 1.0

    x, y = bike2
    bikes[y, x] = -1.0

    return np.stack([occ, bikes], axis=0)

class Tron2ChannelEnv(gym.Env):
    """TronEnv with both players controlled by the same agent. Action is a tuple of (action1, action2)"""
    
    reward_dict = { Result.DRAW: 0, Result.BIKE2_CRASH: 1, Result.BIKE1_CRASH: -1, Result.PLAYING: 0 }

    def __init__(self, size=25):
        super().__init__()
        self.tron = Tron(size)
        self.size = size
        self.action_space = gym.spaces.MultiDiscrete([3, 3])  # (left, forward, right) for each bike relative to their current heading
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 2, size, size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.heading1, self.heading2 = 1, 3  # First facing eachother, bike1 goes right, bike2 goes left
            
        return self._get_state(), {'result': 0}
    
    def step(self, action : np.ndarray):
        assert self.action_space.contains(action), utils.red(f"Jason! Invalid Action {action}")

        # a0, a1 = int(action[0]), int(action[1])

        self.heading1 = (self.heading1 + (action[0] - 1)) % 4  # Because (left, forward, right)
        self.heading2 = (self.heading2 + (action[1] - 1)) % 4  # Because (left, forward, right)
        
        dir1 = TronEnvBase.action_mapping[self.heading1]
        dir2 = TronEnvBase.action_mapping[self.heading2]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        state = self._get_state()
        reward = self.reward_dict[result]

        info = {"result": result} if done else {}
        return state, reward, done, False, info
    
    def _get_state(self):
        # First encode to one hot image
        walls, bikes = encode_observation_2channel(self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos)
        obs1 = (walls, bikes)
        obs2 = (walls, -bikes)  # Flip the sign

        # Then rotate according to heading so that bikes always face up
        obs1 = np.rot90(obs1, k=self.heading1, axes=(1, 2)).copy()  # Copy to remove negative stride
        obs2 = np.rot90(obs2, k=self.heading2, axes=(1, 2)).copy()  # Copy to remove negative stride

        obs = np.stack([obs1, obs2], axis=0)
        assert obs.shape == self.observation_space.shape, utils.red(f"Jason! Obs shape mismatch {obs1.shape} vs {self.observation_space.shape}")
        return obs

class PoLEnv(gym.Env):
    """
    Serves as a proof of concept that the agent actually can learn to go to a square
    Spawns top left (0, 0) and must reach bottom right (size-1, size-1). 
    Action is absolute (left, forward, right). 
    Rewards are minus manhatten distance to goal per step and +1 for reaching the goal. 
    No adversary or walls.
    """
    
    reward_dict = { Result.DRAW: 0, Result.BIKE2_CRASH: 1, Result.BIKE1_CRASH: -1, Result.PLAYING: 0 }
    dirs = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left

    def __init__(self, size=25, render=False):
        super().__init__()
        self.size = size
        self.render = render
        if render:
            os.system('cls')
        self.action_space = gym.spaces.Discrete(4)  # (left, forward, right) for each bike relative to their current heading
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, size, size), dtype=np.float32)
        self.obs_shape = (3, size, size)
        self.n_actions = 4  # up, right, down, left
        
        self.goal = np.array([size-1, size-1], dtype=np.int8)
        self.walls = np.zeros((size, size), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0], dtype=np.int8)
        self.walls.fill(0)
        return self._get_state(), {"result": 0, "state": self.state} 
    
    def step(self, action : int):
        assert self.action_space.contains(action), utils.red(f"Jason! Invalid Action {action} not in {self.action_space}")
        # print(f"Action: {action} from pos {self.pos}")

        old_dist = self._manhattan_distance(self.pos, self.goal)
        self.walls[self.pos[1], self.pos[0]] = 1  # Mark current position as wall

        self.pos += self.dirs[action]
        self.pos = np.clip(self.pos, 0, self.size-1)

        new_dist = self._manhattan_distance(self.pos, self.goal)

        crash = self.walls[self.pos[1], self.pos[0]]
        goal = np.array_equal(self.pos, self.goal)
        progress = (old_dist - new_dist) * .1

        reward = 1.0 if goal else -1.0 if crash else progress
        done = goal or crash

        if self.render:
            self.view()  # render
        info = {"result": int(done), "state": (self.walls.copy(), self.pos.copy())} 
        return self._get_state(), reward, done, False, info
    
    def _get_state(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) # 3, size, size
        obs[0] = self.walls
        obs[1, self.pos[1], self.pos[0]] = 1.0
        obs[2, self.goal[1], self.goal[0]] = 1.0
        return obs

    def _manhattan_distance(self, pos1, pos2):
        return np.abs(pos1 - pos2).sum()
    
    def peek_reward(self, action):
        assert self.action_space.contains(action), utils.red(f"Jason! Invalid Action {action}")

        old_dist = self._manhattan_distance(self.pos, self.goal)
        new_pos = self.pos + self.dirs[action]
        new_pos = np.clip(new_pos, 0, self.size-1)
        new_dist = self._manhattan_distance(new_pos, self.goal)

        done = np.array_equal(new_pos, self.goal)
        oob = -0.1 if np.any(new_pos < 0) or np.any(new_pos >= self.size) else 0
        reward = 1.0 if done else (old_dist - new_dist) * .1 + oob
        return reward

    def view(self):
        clear()
        board = np.full((self.size, self.size), ".", dtype=str)
        board[self.walls == 1] = "#"
        board[self.goal[1], self.goal[0]] = "G"
        board[self.pos[1], self.pos[0]] = "A"

        for row in board:
            print(" ".join(row))
        time.sleep(0.1)

    def set_state(self, state):
        assert isinstance(state, tuple) and len(state) == 2, utils.red(f"Jason! Invalid state {state}, expected tuple of (walls, pos)")
        assert state[0].shape == (self.size, self.size), utils.red(f"Jason! Invalid walls shape {state[0].shape}, expected {(self.size, self.size)}")
        assert state[1].shape == (2,), utils.red(f"Jason! Invalid pos shape {state[1].shape}, expected {(2,)}")
        # print(f"Setting state {self.pos} => {state[1]}")
        self.walls = state[0].copy()
        self.pos = state[1].copy()

    @property
    def state(self):
        return self.walls.copy(), self.pos.copy()

if __name__ == "__main__":
    env = TronEnv(7)
    from .wrappers import TronView
    env = TronView(env, fps=40)

    obs, info = env.reset()
    np.random.seed(21)
    steps = 0
    while True:
        steps += 1
        # action = env.unwrapped.sample_action()
        action = 1
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, info = env.reset()