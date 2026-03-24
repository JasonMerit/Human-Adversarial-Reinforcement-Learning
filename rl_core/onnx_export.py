import numpy as np
import torch
import torch.nn as nn

from rl_core.agents.dqn import QNetwork
from rl_core.utils.helper import bcolors

obs_shape, n_actions = (3, 25, 25), 3

name = "test"
checkpoint_path = f"runs/self_train_4/human.pth"
export_path = f"tron_unity/Assets/{name}.onnx"
dummy_input = torch.rand(1, 25, 25, 3)  # single observation

model = QNetwork.from_checkpoint(checkpoint_path, obs_shape, n_actions)  # Load your trained model
model.eval()  # Set to evaluation mode for export

class UnityExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)  # NHWC → NCHW
        q = self.model(x)
        return q.view(-1, 1, 1, q.shape[-1])  # (N,1,1,C)

wrapped_model = model
# wrapped_model = UnityExportWrapper(model)

torch.onnx.export(
    wrapped_model,              # your trained PyTorch model
    dummy_input,                # example input
    export_path,                # output file
    export_params=True,         # store trained weights
    opset_version=17,           # ONNX opset (higher is more compatible with newer features)
    input_names=['state'],      # input tensor name
    output_names=['q_values'],    # output tensor name
    dynamic_axes={'state': {0: 'batch'}, 'q_values': {0: 'batch'}}  # allow variable batch sizes
)

print(f"Model {bcolors.OKGREEN}successfully{bcolors.ENDC} exported to ONNX format at {bcolors.OKCYAN}'{export_path}'{bcolors.ENDC}")


from onnx import onnx
onnx_model = onnx.load(export_path)
onnx.checker.check_model(onnx_model)

# Test with dummy input
import onnxruntime as ort
ort_session = ort.InferenceSession(export_path)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print(f"ONNX model output shape: {ort_outs[0].shape}")
print(ort_outs[0])

