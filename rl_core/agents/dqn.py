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

class QNetFlat(nn.Module):
    def __init__(self, input_size, num_actions=3):
        super().__init__()
        self.input_size = input_size  # Total flattened size: 3*11*11 = 363
        self.num_actions = num_actions
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # x is already flat: [batch, 363]
        return self.fc(x)


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

if __name__ == "__main__":
    shape = (3, 11, 11)
    model = QNet(input_shape=shape, num_actions=3)
    # model = QNetFlat(input_size=3*11*11)
    
    # input = torch.randn(1, 3*11*11)
    input = torch.randn(1, *shape)

    torch.onnx.export(
    model,                      # your trained PyTorch model
    input,                      # example input
    "rl_core/model_static.onnx",       # output file
    export_params=True,         # store trained weights
    opset_version=17,           # ONNX opset (higher is more compatible with newer features)
    input_names=['state'],      # input tensor name
    output_names=['output'],    # output tensor name
    dynamic_axes=None
    # dynamic_axes={'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}}  # allow variable batch sizes
)
    print(model(input))