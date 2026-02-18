import numpy as np
import torch

from rl_core.agents.dqn import QNet
from rl_core.utils.helper import bcolors

model = QNet.load("rl_core/q_net.pth")  # Load your trained model
dummy_state = np.zeros(model.input_shape, dtype=np.float32)  # float32 preferred
dummy_tensor = torch.from_numpy(dummy_state).unsqueeze(0)  # batch dimension


torch.onnx.export(
    model,                      # your trained PyTorch model
    dummy_tensor,               # example input
    "rl_core/agent.onnx",       # output file
    export_params=True,         # store trained weights
    opset_version=17,           # ONNX opset (higher is more compatible with newer features)
    input_names=['state'],      # input tensor name
    output_names=['action'],    # output tensor name
    dynamic_axes={'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}}  # allow variable batch sizes
)

print(f"{bcolors.OKGREEN}Model successfully exported to ONNX format at 'rl_core/agent.onnx'{bcolors.ENDC}")

