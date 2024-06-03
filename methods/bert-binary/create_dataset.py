import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../../data/cordis-binary-telecoms.csv')

# Combine 'description' and 'name' into a single column
df['combined_text'] = 'Title: ' + df['name'] + ' Abstract: ' + df['description']

# Drop the original 'description' and 'name' columns
df.drop(['description', 'name'], axis=1, inplace=True)

# Split the data into train and test sets with stratification to maintain class balance
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['isTelecoms'], random_state=42)

# Randomize the order of the datapoints in both sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the train and test sets to CSV files
train_df.to_csv('../../data/ml/binary/train.csv', index=False)
test_df.to_csv('../../data/ml/binary/test.csv', index=False)
