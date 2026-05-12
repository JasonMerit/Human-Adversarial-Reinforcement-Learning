import os
from pathlib import Path
import csv

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

def sort_glossary():
    print()
    rows = []
    with open("glossary.csv") as f:
        reader = csv.reader(f)
        next(reader)
        rows = sorted(reader, key=lambda x: x[0].lower())

    for t, d in rows:
        print(f"{t} & {d} \\\\ \\hline")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Clean the folder.")
    # parser.add_argument("folder", type=str, help="Path folder of trained model checkpoints.")
    # clean(parser.parse_args().folder)

    sort_glossary()