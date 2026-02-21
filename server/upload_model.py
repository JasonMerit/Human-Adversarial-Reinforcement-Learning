import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

file_path = "rl_core/model.onnx"
storage_path = "model_v1.onnx"

with open(file_path, "rb") as f:
    file_size = os.path.getsize(file_path) / 1000
    print(f"Uploading file: {file_path} ({file_size} bytes)")
    response = supabase.storage.from_("onnx-models").upload(
        path=storage_path,
        file=f,
        file_options={"content-type": "application/octet-stream"}
    )

print(response)