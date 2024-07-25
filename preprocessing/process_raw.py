import pandas as pd
import os
import ast
import numpy as np
import json
import argparse


PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../'))

def main(create_full_binary):

    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'raw_data', 'data.csv'))

    # Read in the text representations of arrays
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else [])

    df = df.drop_duplicates(subset=['description', 'name'])
    df = df.dropna(subset=['description'])

    if create_full_binary:
        full_binary = df[['description', 'name']].copy()
        full_binary["text"] = "Title: " + full_binary["name"] + " Abstract: " + full_binary["description"]
        full_binary.drop(columns=["name", "description"], inplace=True)
        full_binary.to_csv(os.path.join(PROJECT_ROOT, 'binary', 'data', 'all_samples.csv'), index=False)

    # Extract labelled data
    df = df[df['topics'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]

    df = df[df['dataSource'] == 'CORDIS']
    # Filter down to description, name and topics
    df = df[['description', 'name', 'topics']]

    # Collate Abstract title and content into single datapoint
    df["text"] = "Title: " + df["name"] + " Abstract: " + df["description"]
    df.drop(columns=["name", "description"], inplace=True)

    # List of topics we use as classes
    topics = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]
   
    # If the data was labelled with a pre-determined telecoms topic, we label it as telecoms
    df["isTelecoms"] = df["topics"].apply(lambda s:  np.any([x in topics for x in s]))

    telecoms = df[df["isTelecoms"]].copy()

    # Remove any topics that aren't directly signifying telecoms
    telecoms["topics"] = telecoms["topics"].apply(lambda x: [y for y in x if y in topics])

    # Process MuliLabel data
    multilabel = telecoms.copy()
    multilabel.drop(columns=["isTelecoms"], inplace=True)
    multilabel = multilabel[["text", "topics"]]
    multilabel['topics'] = multilabel['topics'].apply(json.dumps)
    multilabel.to_csv(os.path.join(PROJECT_ROOT, 'multilabel', 'data', 'cordis-multilabel-telecoms.csv'), index=False)

    # Process Binary Data
    non_telecoms = df[~df["isTelecoms"]].sample(n=len(telecoms), random_state=42)  
    binary = pd.concat([telecoms, non_telecoms])
    binary.drop(columns=["topics"], inplace=True)
    binary.to_csv(os.path.join(PROJECT_ROOT, 'binary', 'data', 'cordis-binary-telecoms.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('create_full_binary', default=False, help='Flag to create full binary dataset')
    args = parser.parse_args()
    main(args.create_full_binary)
