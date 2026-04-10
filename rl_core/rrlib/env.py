from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np

from rl_core.env.tron import Tron, Result

def encode_observation(walls, bike1, bike2):
    occ = (walls > 0).astype(np.float32)

    you = np.zeros_like(occ)
    other = np.zeros_like(occ)

    x, y = bike1
    you[y, x] = 1.0

    x, y = bike2
    other[y, x] = 1.0

    return np.stack([occ, you, other], axis=-1)

class TronMultiAgentEnv(MultiAgentEnv):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    reward_dict = {Result.DRAW: 0, Result.BIKE2_CRASH: 1, Result.BIKE1_CRASH: -1, Result.PLAYING: 0}

    OBS = gym.spaces.Box(0, 1, (25, 25, 3), dtype=np.float32)
    ACT = gym.spaces.Discrete(3)

    def __init__(self, kek=None):
        super().__init__()
        self.tron = Tron(25)

        self.agents = ["bike1", "bike2"]
        self.possible_agents = self.agents

        self.observation_spaces = {"bike1": self.OBS, "bike2": self.OBS}
        self.action_spaces = {"bike1": self.ACT, "bike2": self.ACT}

    def reset(self, *, seed=None, options=None):
        self.tron.reset()

        self.heading1 = 1
        self.heading2 = 3

        obs1, obs2 = self._get_state()
        obs = {"bike1": obs1, "bike2": obs2}

        return obs, {}

    def step(self, action_dict):
        a1 = int(action_dict["bike1"])
        a2 = int(action_dict["bike2"])

        self.heading1 = (self.heading1 + (a1 - 1)) % 4
        self.heading2 = (self.heading2 + (a2 - 1)) % 4

        dir1 = self.action_mapping[self.heading1]
        dir2 = self.action_mapping[self.heading2]

        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING

        obs1, obs2 = self._get_state()
        obs = {"bike1": obs1, "bike2": obs2}

        reward = self.reward_dict[result]
        rewards = {"bike1": reward, "bike2": -reward}

        terminated = {"bike1": done, "bike2": done, "__all__": done}
        truncated = {"bike1": False, "bike2": False, "__all__": False}
        infos = {"bike1": {"result": result} if done else {}, "bike2": {"result": result} if done else {}}

        return obs, rewards, terminated, truncated, infos

    def _get_state(self):
        obs = encode_observation(self.tron.walls, self.tron.bike1.pos ,self.tron.bike2.pos)

        walls = obs[..., 0]
        bike1 = obs[..., 1]
        bike2 = obs[..., 2]

        obs1 = np.stack([walls, bike1, bike2], axis=-1)
        obs2 = np.stack([walls, bike2, bike1], axis=-1)

        obs1 = np.rot90(obs1, k=self.heading1)
        obs2 = np.rot90(obs2, k=self.heading2)

        return obs1.astype(np.float32), obs2.astype(np.float32)
    

if __name__ == "__main__":
    env = TronMultiAgentEnv()
    obs, info = env.reset()
    for i in range(100):
        action_dict = {"bike1": env.action_spaces["bike1"].sample(), "bike2": env.action_spaces["bike2"].sample()}
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        if terminated['bike1']:
            print("Episode ended with result:", infos["bike1"]['result'])
            break
