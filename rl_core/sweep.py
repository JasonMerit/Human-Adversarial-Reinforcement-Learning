import subprocess
import sys

# configurations to test
NUM_ENVS_LIST = [1, 2, 4, 8, 16, 32, 64]
# Results [304, 416, 440, 486, 903, 1149, 1436]

BASE_CMD = [sys.executable, "-m", "rl_core.self_train"]

for n in NUM_ENVS_LIST:
    print(f"\n===== RUNNING num_envs={n} =====\n")

    cmd = BASE_CMD + ["--num_envs", str(n)]

    process = subprocess.run(cmd)

    if process.returncode != 0:
        print(f"Run failed for num_envs={n}")
        sys.exit(process.returncode)