import pandas as pd
import dask.dataframe as dd
import os
import ast
import numpy as np
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    import cupy as cp

    x = cp.arange(10)
    print(x)

    df = pd.read_csv(os.path.join(to_absolute_path(cfg.data.raw), 'data.csv'))

    # Read in the text representations of arrays
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else [])

    df = df.drop_duplicates(subset=['description', 'name'])
    df = df.dropna(subset=['description'])

    # Extract labelled data
    df = df[df['topics'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]

    df = df[df['dataSource'] == 'CORDIS']
    # Filter down to description, name and topics
    df = df[['description', 'name', 'topics']]

    # List of topics we use as classes
    topics = cfg.labels.topics
    df["isTelecoms"] = df["topics"].apply(lambda s:  np.any([x in topics for x in s]))

    telecoms = df[df["isTelecoms"]]

    # Remove any topics that aren't directly signifying telecoms
    telecoms["topics"] = telecoms["topics"].apply(lambda x: [y for y in x if y in topics])

    telecoms.to_csv(to_absolute_path('../data/cordis-telecoms.csv'), index=False)

    multilabel = telecoms.copy()
    multilabel.drop(columns=["isTelecoms"], inplace=True)

    # Add a column called text that is the concatenation of the name and description, where the name is Title: {name} and the description is Abstract: {description}
    multilabel["text"] = "Title: " + multilabel["name"] + " Abstract: " + multilabel["description"]
    multilabel.drop(columns=["name", "description"], inplace=True)
    multilabel = multilabel[["text", "topics"]]
    multilabel['topics'] = multilabel['topics'].apply(json.dumps)

    multilabel.to_csv(to_absolute_path(cfg.data.processed, 'cordis-multilabel-telecoms.csv'), index=False)


if __name__ == "__main__":
    main()
