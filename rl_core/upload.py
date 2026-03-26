# upload.py torch module to supabase storage
import subprocess
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from supabase import create_client

from .agents.dqn import QNetwork
from .env import utils

class UnityExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x_nhwc):
            x = x_nhwc.permute(0, 3, 1, 2)  # NHWC → NCHW
            q = self.model(x)
            return q.reshape(q.shape[0], 1, 1, q.shape[-1])  # (N,1,1,C)
        
def pth2onnx(checkpoint_path, export_path):
    obs_shape, n_actions = (3, 25, 25), 3
    dummy_input = torch.rand(1, 25, 25, 3)

    model = QNetwork.from_checkpoint(checkpoint_path, obs_shape, n_actions)  # Load your trained model
    model.eval()  # Set to evaluation mode for export

    torch.onnx.export(UnityExportWrapper(model), dummy_input, export_path, opset_version=15)

    print(f"Exported to .onnx at {utils.cyan(export_path)}")

def onnx2sentis(onnx2sentis_exe, onnx_path):
    result = subprocess.run([onnx2sentis_exe, onnx_path], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error converting ONNX to Sentis:", result.stderr)
        raise RuntimeError("Conversion failed")

def upload_sentis(name, bucket, sentis_path):
    load_dotenv()
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

    extension = ".sentis"
    storage_path = name + extension

    # Rename existing file
    files = supabase.storage.from_(bucket).list()
    count = len([f for f in files if f["name"].startswith(name) and f["name"].endswith(extension)])
    if count > 0:
        response = supabase.storage.from_(bucket).move(
                storage_path,  # Assuming one already exists with this name
                name + str(count) + extension
            )
        print(f"Renamed existing file to {utils.cyan(name + str(count) + extension)}")

    # Upload new file
    with open(sentis_path, "rb") as f:
        response = supabase.storage.from_(bucket).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "application/octet-stream"}
        )
    print(f"Uploaded to {utils.cyan(response.fullPath)}")

def upload(checkpoint_path, name):
    onnx_path = f"rl_core/{name}.onnx"  # Temp location for ONNX file
    onnx2sentis_folder = os.getcwd() + "/tools/onnx2sentis_windows/"  # absolute path for subprocess
    sentis_path = onnx2sentis_folder + f"{name}.sentis"
    bucket = "onnx-models"

    pth2onnx(checkpoint_path, onnx_path)
    onnx2sentis(onnx2sentis_folder + "onnx2sentis.exe", onnx_path)
    print("Converted to .sentis at "+ utils.cyan('tools/onnx2sentis_windows/' + name + '.sentis'))
    upload_sentis(name, bucket, sentis_path)
    print(utils.green("Upload complete!!!"))

if __name__ == "__main__":
    # Get name from system args
    import argparse
    parser = argparse.ArgumentParser(description="Upload trained model.")
    parser.add_argument("path", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--name", type=str, default="adversary", help="Name of uploaded file.")
    args = parser.parse_args()

    upload(args.path, args.name)
    
    