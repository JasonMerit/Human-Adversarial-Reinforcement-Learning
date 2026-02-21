import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("ANON_KEY"))

trajectory = [[1, 1], [0, 2], [1, 3]]
winner = 1

response = supabase.functions.invoke(
    "upload-episode",
    invoke_options={
        "body": {
            "trajectory": trajectory,
            "winner": winner,
        }
    },
)
print(response)

