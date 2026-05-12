import os
from pathlib import Path

def clean(folder):
    """Accidentally used folder name infront of each file in the folder, so remove folder name from each item that prefixes folder"""
    absolute_folder = Path('runs') / folder
    print(f"Cleaning folder {absolute_folder}...")
    input("Press Enter to continue...")
    for filename in os.listdir(absolute_folder):
        if filename.startswith(folder):
            new_name = filename[len(folder):]
            print(f"{filename} ==> {new_name}")
            os.rename(absolute_folder / filename, absolute_folder / new_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean the folder.")
    parser.add_argument("folder", type=str, help="Path folder of trained model checkpoints.")
    clean(parser.parse_args().folder)