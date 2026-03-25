# import os
# from dotenv import load_dotenv
# from supabase import create_client

# load_dotenv()
# supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

# file_bytes = supabase.storage.from_("onnx-models").download(
#     "model_v1.onnx"
# )

# with open("server/downloaded.onnx", "wb") as f:
#     f.write(file_bytes)

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SERVICE_ROLE_KEY")
BUCKET_NAME = "onnx-models"
MODEL_FILE = "model_v1.onnx"

# Supabase Storage URL for private bucket
url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{MODEL_FILE}"

# Use Service Role Key to authenticate
headers = {
    "apikey": SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SERVICE_ROLE_KEY}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("server/downloaded.onnx", "wb") as f:
        f.write(response.content)
    print("ONNX model downloaded successfully!")
else:
    print(f"Failed to download: {response.status_code} {response.text}")