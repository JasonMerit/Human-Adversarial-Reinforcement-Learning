import torch
from torch.utils.data import Dataset
import numpy as np
import gymnasium as gym

from rl_core.env import TronDuoEnv, TronView, StateViewer

class TorchObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, obs):
        return torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

class TronDataset(Dataset):

    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):

        states, actions = zip(*self.episodes[idx])

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)

        return states, actions
    
    def save(self, path):
        torch.save(self.episodes, path)

    @classmethod
    def load(cls, path):
        episodes = torch.load(path, weights_only=False)
        return cls(episodes)

        
    
def heading_to_action(prev_heading, new_heading):
    delta = (new_heading - prev_heading) % 4

    if delta == 3:
        return 0  # left
    elif delta == 0:
        return 1  # forward
    elif delta == 1:
        return 2  # right
    else:
        raise ValueError("Invalid heading change")

def populate_data():
    env = TronDuoEnv()
    # env = TorchObservationWrapper(env, device="cpu")
    # env = TronView(env)
    
    # Load trajectories from file
    with open("rl_core/player_modelling/trajectories.txt", "r") as f:
        trajectories = [eval(line.strip()) for line in f.readlines()]

    episodes = []
    for trajectory in trajectories:
        obs, _ = env.reset()
        episode = []
        for joint_action in trajectory:
            a0, a1 = joint_action  # Absolute actions from the trajectory
            x = heading_to_action(env.unwrapped.heading1, a0)
            y = heading_to_action(env.unwrapped.heading2, a1)
            obs, _, done, _, _ = env.step([x, y])
            # print(obs.shape)
            # print(obs[:, 1].shape)
            # quit()
            episode.append([obs[1], y])  # We want to predict the adversary's action based on their observation
        episodes.append(episode)
    
    data = TronDataset(episodes)
    data.save("rl_core/player_modelling/tron_dataset.pt")

def iterate_data():
    env = TronDuoEnv()
    env = TronView(env)
    
    # Load trajectories from file
    data = TronDataset.load("rl_core/player_modelling/tron_dataset.pt")

    sv = StateViewer(25, 20)

    for states, actions in data:
        for state in states:
            sv.view(state)
    

if __name__ == "__main__":
    # populate_data()
    iterate_data()




