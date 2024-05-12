# %% [markdown]
# # Extract Cordis rows from df

# %%
# Import libraries
import pandas as pd
import ast
import numpy as np

import os

# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the directory of the current file
os.chdir(current_dir)

# Now you can start working with files in this directory


# %%

# Cordis data will be in the labelled data
file_path = '../data/data.csv'

df = pd.read_csv(file_path)

# Formatting strings back into arrays
df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else [])

#df.head()

# %% [markdown]
# ### Using practices established in memory-test.csv

# %%
df = df.drop_duplicates(subset=['description', 'name'])
df = df.dropna(subset=['description'])
# Extract labelled data
df = df[df['topics'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
#df.shape

# %%
# Extract rows where the 'dataSource' column is 'CORDIS'
df = df[df['dataSource'] == 'CORDIS']
# Filter down to description, name and topics
df = df[['description', 'name', 'topics']]
#df.shape

# %%
#df.head()

# %%
df.to_csv('../data/cordis-desc-name-topics.csv', index=False)

# %%
tel_topic_match = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]

df["isTelecoms"] = df["topics"].apply(lambda s:  np.any([x in tel_topic_match for x in s]))

# %%
telecoms = df[df["isTelecoms"]].copy()
#telecoms.shape

# %%
# Remove all topics that are not in tel_topic_match
telecoms["topics"] = telecoms["topics"].apply(lambda x: [y for y in x if y in tel_topic_match])

# %%
#telecoms.head()

# %%
telecoms.to_csv('../data/cordis-telecoms.csv', index=False)

# %%



