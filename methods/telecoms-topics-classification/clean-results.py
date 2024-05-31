import os
import sys
import pandas as pd
import json

# Define the allowed keys
ALLOWED_KEYS = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]


def filter_topics(row, allowed_keys):
    """
    Filter the topics dictionary in the given row to include only allowed keys.
    """
    if 'topics' in row and isinstance(row['topics'], dict):
        return {k: v for k, v in row['topics'].items() if k in allowed_keys}
    return row['topics']

def process_file(filepath, allowed_keys):
    """
    Process a single JSON file, filtering the topics column based on allowed keys.
    """

    print(filepath)
    df = pd.read_json(filepath, lines=True)
    if 'topics' in df.columns:
        df['topics'] = df.apply(lambda row: filter_topics(row, allowed_keys), axis=1)
        df.to_json(filepath, orient='records', lines=True)
        print(f"Processed and updated file: {filepath}")

def process_folder(folder_path, allowed_keys):
    """
    Process all JSON files in the specified folder.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            process_file(filepath, allowed_keys)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"The specified path is not a directory: {folder_path}")
        sys.exit(1)

    process_folder(folder_path, ALLOWED_KEYS)
    print("Processing complete.")
