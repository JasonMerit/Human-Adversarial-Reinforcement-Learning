import torch
import torch.nn as nn


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

