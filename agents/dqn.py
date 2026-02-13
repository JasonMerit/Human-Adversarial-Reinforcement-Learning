import torch
import torch.nn as nn

from agents.base import Agent
from environment.wrappers import TronTorch
from utils.helper import has_wrapper, bcolors

class QNet(nn.Module):
    def __init__(self, input_shape=(3,10,10), num_actions=3):
        super().__init__()
        c,h,w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 32 * h * w
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    @staticmethod
    def load(path):
        model = QNet()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

class DQNAgent(Agent):
    def __init__(self, qnet_path: str):
        self.qnet = QNet.load(qnet_path)
    
    def eval(self):
        self.qnet.eval()

    def __call__(self, state):
        with torch.no_grad():
            q_values = self.qnet(state)
            return q_values.argmax().item()
    
    def reset(self):
        pass

    def _check_env(self, env):
        if not has_wrapper(env, TronTorch):
            raise ValueError(f"{bcolors.FAIL}DQNAgent requires TronTorch wrapper{bcolors.ENDC}")