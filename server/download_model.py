import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(url, key)

file_bytes = supabase.storage.from_("onnx-models").download(
    "model_v1.onnx"
)

with open("server/downloaded.onnx", "wb") as f:
    f.write(file_bytes)