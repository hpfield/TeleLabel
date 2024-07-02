import pandas as pd
import os
import ast
import numpy as np
import json
import hydra
from omegaconf import DictConfig

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))

@hydra.main(config_path="../../conf", config_name="config", version_base='1.3.2')
def main(cfg: DictConfig):

    df = pd.read_csv(os.path.join(PROJECT_ROOT, cfg.data.raw, 'data.csv'))

    # Read in the text representations of arrays
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else [])

    df = df.drop_duplicates(subset=['description', 'name'])
    df = df.dropna(subset=['description'])

    # Extract labelled data
    df = df[df['topics'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]

    df = df[df['dataSource'] == 'CORDIS']
    # Filter down to description, name and topics
    df = df[['description', 'name', 'topics']]

    # Collate Abstract title and content into single datapoint
    df["text"] = "Title: " + df["name"] + " Abstract: " + df["description"]
    df.drop(columns=["name", "description"], inplace=True)

    # List of topics we use as classes
    topics = list(cfg.labels.topics)
   
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
    multilabel.to_csv(os.path.join(PROJECT_ROOT, cfg.data.processed, 'cordis-multilabel-telecoms.csv'), index=False)

    # Process Binary Data
    non_telecoms = df[~df["isTelecoms"]].sample(n=len(telecoms), random_state=42)  
    binary = pd.concat([telecoms, non_telecoms])
    binary.drop(columns=["topics"], inplace=True)
    binary.to_csv(os.path.join(PROJECT_ROOT, cfg.data.processed, 'cordis-binary-telecoms.csv'), index=False)


if __name__ == "__main__":
    main()
