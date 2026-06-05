# import os, time, random
# from pathlib import Path
# import yaml
# import numpy as np
# from rich import print
# import gymnasium as gym

# from rl_core.env import TronDuoEnv, TronView

from pathlib import Path
import argparse

import yaml

def read_second_line(file_path: Path):
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            next(f)  # skip first line
            return next(f).rstrip("\n")
    except StopIteration:
        return None
    except FileNotFoundError:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("X", type=str, help="Prefix of team folders to analyze (e.g., 'NN' to analyze 'NN0', 'NN1', etc.)")
    args = parser.parse_args()

    base = Path("runs")
    # pattern = f"{args.X}*/Out.out"

    # files = sorted(base.glob(pattern))
    # lines = set()
    # for f in files:
    #     second_line = read_second_line(f)
    #     run_id = f.parent.name
    #     print(f"{run_id}\t{second_line}")
    #     if second_line in lines:
    #         raise ValueError(f"Duplicate second line found: {second_line} in file {f}")
    #     if second_line is not None:
    #         lines.add(second_line)

    # Do something similar but instead read the "training_time_mins" found within "results.yml" in each folder
    pattern = f"{args.X}*/results.yml"
    files = sorted(base.glob(pattern))
    times = []
    for f in files:
        run_id = f.parent.name
        with f.open("r") as y:
            data = yaml.safe_load(y)
            training_time = data.get("training_time_mins")
            print(f"{run_id}\t{training_time}")
            if training_time is not None:
                times.append(training_time)

    print(f"Max training time: {max(times) / 60:.2f} mins")

if __name__ == "__main__":
    main()