import os
import pandas as pd
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def clean_key(path: str) -> str:
    """
    Extract a composite key by combining the company name (from the parent folder)
    and the aspect (everything after the first hyphen in the filename).
    
    For example, if the file is:
      data/nextera_energy/nextera energy - SASB Metrics.csv
    then:
      - Company = "nextera_energy"
      - Aspect  = "SASB_Metrics"
      and the key becomes:
          "nextera_energy_SASB_Metrics"
    """
    # Get the company name from the parent folder.
    company = os.path.basename(os.path.dirname(path)).strip().replace(" ", "_")
    
    # Get only the filename (without directory and extension)
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    
    # Extract the aspect: take everything after the first hyphen if it exists.
    if "-" in filename:
        key_part = filename.split("-", 1)[1].strip()
    else:
        key_part = filename.strip()
    
    key_part = key_part.replace(" ", "_")
    
    # Return the composite key.
    return f"{company}_{key_part}"

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

if __name__ == "__main__":
    sample1 = os.path.join(DATA_DIR, "nextera_energy", "nextera energy - SASB Metrics.csv")
    sample2 = os.path.join(DATA_DIR, "nextera_energy", "nextera energy - Social.csv")
    print("Generated key 1:", clean_key(sample1))
    print("Generated key 2:", clean_key(sample2))