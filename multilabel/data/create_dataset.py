# prepare_data.py

import pandas as pd
import os

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))

# Load the dataset
df = pd.read_csv(os.path.join(MULTILABEL_ROOT, 'data', 'cordis-multilabel-telecoms.csv'))

# Split the dataset into train and test sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save datasets as CSV files
train_df.to_csv(os.path.join(MULTILABEL_ROOT, 'data', 'train.csv'), index=False)
test_df.to_csv(os.path.join(MULTILABEL_ROOT, 'data', 'test.csv'), index=False)
