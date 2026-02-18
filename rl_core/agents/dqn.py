import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent
from rl_core.utils.helper import has_wrapper, bcolors

class QNet(nn.Module):
    def __init__(self, input_shape, num_actions=3):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
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
        weights = torch.load(path, weights_only=True)
        conv_out = weights['fc.0.weight'].shape[1]  # flattened input to first FC layer
        size = int((conv_out // 32) ** 0.5)
        input_shape = (weights['conv.0.weight'].shape[1], size, size)  # Channels, Height, Width
        num_actions = weights['fc.2.weight'].shape[0]
        print(f"Loading QNet with input shape: {bcolors.OKGREEN}{input_shape}{bcolors.ENDC} and num actions: {bcolors.OKGREEN}{num_actions}{bcolors.ENDC}")
        
        model = QNet(input_shape, num_actions)
        model.load_state_dict(weights)
        return model

class DQNAgent(Agent):
    def __init__(self, qnet_path: str):
        self.qnet = QNet.load(qnet_path)
    
    def eval(self):
        self.qnet.eval()

    def __call__(self, state):
        with torch.no_grad():
            q_values = self.qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return q_values.argmax().item()
    
    def reset(self):
        pass

    def _check_env(self, env):
        pass

class DQNSoftAgent(DQNAgent):
    
    def __call__(self, state, temperature=1.0):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.qnet(state)

            probs = F.softmax(q_values / temperature, dim=1)
            action = torch.multinomial(probs, num_samples=1)

            return action.item()
