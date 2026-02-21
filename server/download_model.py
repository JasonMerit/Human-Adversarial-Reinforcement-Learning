import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

file_bytes = supabase.storage.from_("onnx-models").download(
    "model_v1.onnx"
)

with open("server/downloaded.onnx", "wb") as f:
    f.write(file_bytes)