import os
import pandas as pd
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def clean_key(path: str) -> str:
    """
    Convert path like 'nextera energy - Social.csv' to 'nextera_energy_Social',
    avoiding repeated company name.
    """
    path = path.replace(DATA_DIR + os.sep, "")  # remove base dir
    path = os.path.splitext(path)[0]            # remove .csv

    parts = re.split(r'[/\\]', path)  # split by folder separators
    folder = parts[0].strip().replace(" ", "_").lower()

    # Extract meaningful filename part
    filename_part = parts[-1]
    filename_cleaned = filename_part.replace(folder.replace("_", " "), "", 1).strip(" -_")

    # Final key: folder_company + cleaned file part
    return f"{folder}_{filename_cleaned.replace(' ', '_')}"

def load_all_csvs() -> dict:
    """Recursively load all CSVs from company subdirectories into a dict of DataFrames."""
    print(f"\n📂 Scanning directory: {DATA_DIR}")
    data_frames = {}

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                key = clean_key(full_path)

                print(f"➡️  Loading file: {file}")
                print(f"    ↳ Key: {key}")

                try:
                    df = pd.read_csv(full_path)
                    data_frames[key] = df
                    print(f"    ✔ Success: {df.shape[0]} rows")
                except Exception as e:
                    print(f"    ❌ Failed to load {file}: {e}")
    
    return data_frames
