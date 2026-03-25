# upload.py torch module to supabase storage
import subprocess
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from supabase import create_client

from rl_core.agents.dqn import QNetwork
from rl_core.utils.helper import bcolors

# ===================
# === pth -> onnx ===
# ===================
obs_shape, n_actions = (3, 25, 25), 3

name = "adversary"
checkpoint_path = f"runs/self_train_4/adversary.pth"
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

wrapped_model = UnityExportWrapper(model)

torch.onnx.export(
    wrapped_model,              # your trained PyTorch model
    dummy_input,                # example input
    export_path,                # output file
    export_params=True,         # store trained weights
    opset_version=15,           # ONNX opset (higher is more compatible with newer features)
    input_names=['state'],      # input tensor name
    output_names=['q_values'],    # output tensor name
    dynamic_axes={'state': {0: 'batch'}, 'q_values': {0: 'batch'}}  # allow variable batch sizes
)

print(f"Exported to ONNX format at {bcolors.OKCYAN}'{export_path}'{bcolors.ENDC}")


# ======================
# === onnx -> sentis ===
# ======================
onnx2sentis_folder = os.getcwd() + "/tools/onnx2sentis_windows/"  # absolute path

sentis_path = onnx2sentis_folder + f"{name}.sentis"
result = subprocess.run([onnx2sentis_folder + "onnx2sentis.exe", export_path], capture_output=True, text=True)

if result.returncode != 0:
    print("Error converting ONNX to Sentis:", result.stderr)
    raise RuntimeError("Conversion failed")

print(f"Converted to Sentis format at {bcolors.OKCYAN}'{sentis_path}'{bcolors.ENDC}")


# ==========================
# === sentis -> supabase ===
# ==========================
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

storage_path = "adversary.sentis"

bucket = "onnx-models"
extension = ".sentis"

# Rename existing file
files = supabase.storage.from_(bucket).list()
count = len([f for f in files if f["name"].startswith(name) and f["name"].endswith(extension)])
response = supabase.storage.from_(bucket).move(
        name + extension,  # Assuming one already exists with this name
        name + str(count) + extension
    )
print(f"Renamed existing file to {bcolors.OKCYAN}{name + str(count) + extension}{bcolors.ENDC}")

# Upload new file
with open(sentis_path, "rb") as f:
    response = supabase.storage.from_(bucket).upload(
        path=storage_path,
        file=f,
        file_options={"content-type": "application/octet-stream"}
    )
print(f"Model {bcolors.OKGREEN}successfully{bcolors.ENDC} uploaded to {bcolors.OKCYAN}'{response.fullPath}'{bcolors.ENDC}")