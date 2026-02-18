from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Return a batch as NumPy arrays."""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def sample_torch(self, batch_size, device="cpu"):
        """
        Return a batch as torch tensors on the specified device.
        Automatically converts to float32 for states/rewards and long for actions.
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)

        # Convert to arrays first to handle inhomogeneous nested states
        s = torch.tensor(np.array(s), dtype=torch.float32, device=device).squeeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device).squeeze(1)
        a = torch.tensor(a, dtype=torch.long, device=device)
        r = torch.tensor(r, dtype=torch.float32, device=device)
        d = torch.tensor(d, dtype=torch.float32, device=device)

        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)
