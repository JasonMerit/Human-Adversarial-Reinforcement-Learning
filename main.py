# import os, time, random
# from pathlib import Path
# import yaml
# import numpy as np
# from rich import print
# import gymnasium as gym

# from rl_core.env import TronDuoEnv, TronView

from pathlib import Path
import argparse

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
    parser.add_argument("--X", required=True, help="Setup name (X in runs/X_n/)")
    args = parser.parse_args()

    base = Path("runs")
    pattern = f"{args.X}*/Out.out"

    files = sorted(base.glob(pattern))
    lines = set()
    for f in files:
        second_line = read_second_line(f)
        run_id = f.parent.name
        print(f"{run_id}\t{second_line}")
        if second_line in lines:
            raise ValueError(f"Duplicate second line found: {second_line} in file {f}")
        if second_line is not None:
            lines.add(second_line)

if __name__ == "__main__":
    main()