import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the project root and result paths
MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
comparison_path = os.path.join(MULTILABEL_ROOT, 'results', 'comparison')

# Create the comparison directory if it doesn't exist
os.makedirs(comparison_path, exist_ok=True)

# Define file paths
file1 = os.path.join(MULTILABEL_ROOT, 'results', 'llama3-8B', 'llama-3-multilabel-classification-best.csv')
file2 = os.path.join(MULTILABEL_ROOT, 'results', 'bert', 'best.csv')

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge dataframes on the Metric column
df1 = df1[['Metric', 'Best Score']].rename(columns={'Best Score': 'Best Score llama3-8B'})
df2 = df2[['Metric', 'Best Score']].rename(columns={'Best Score': 'Best Score bert'})
merged_df = pd.merge(df1, df2, on='Metric')

# Print the merged dataframe to debug
print("\nMerged DataFrame:")
print(merged_df)

# Plotting the results
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(merged_df['Metric']))

# Plot bars
plt.bar([i - bar_width/2 for i in index], merged_df['Best Score llama3-8B'], bar_width, label='llama3-8B')
plt.bar([i + bar_width/2 for i in index], merged_df['Best Score bert'], bar_width, label='bert')

# Add labels, title, and legend
plt.xlabel('Metrics')
plt.ylabel('Best Score')
plt.title('Comparison of Optimal Performance Metrics')
plt.xticks(index, merged_df['Metric'], rotation=45)
plt.legend()

# Save the plot
plot_path = os.path.join(comparison_path, 'performance_comparison.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
