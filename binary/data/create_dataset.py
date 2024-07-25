import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('cordis-binary-telecoms.csv')

# Split the data into train and test sets with stratification to maintain class balance
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['isTelecoms'], random_state=42)

# Randomize the order of the datapoints in both sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the train and test sets to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
