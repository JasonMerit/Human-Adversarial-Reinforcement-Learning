import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SERVICE_ROLE_KEY"))

major = 1
minor = 2
patch = 0

response = (
    supabase
    .table("episodes")
    .select("trajectory, winner, trapped")
    .eq("version_major", major)
    .eq("version_minor", minor)
    .eq("version_patch", patch)
    .execute()
)

episodes = response.data

print(f"Downloaded {len(episodes)} episodes")

# Convert trajectories to python lists
trajectories = [ep["trajectory"] for ep in episodes]

# Save to file
with open("rl_core/player_modelling/trajectories.txt", "w") as f:
    for traj in trajectories:
        f.write(str(traj) + "\n")
print("Saved to trajectories.txt")