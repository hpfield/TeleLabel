import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the project root and result paths for binary problem
BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))
comparison_path = os.path.join(BINARY_ROOT, 'results', 'comparison')

# Create the comparison directory if it doesn't exist
os.makedirs(comparison_path, exist_ok=True)

# Define file paths
file1 = os.path.join(BINARY_ROOT, 'results', 'bert', 'bert_metrics.csv')
file2 = os.path.join(BINARY_ROOT, 'results', 'llama3-8B', 'best.csv')

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Convert df1 to long format to match df2
df1_long = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_score'],
    'Best Score bert': [df1['Accuracy'][0], df1['Precision'][0], df1['Recall'][0], df1['F1_score'][0]]
})

# Rename columns to match the expected format
df2 = df2.rename(columns={'Best Score': 'Best Score llama3-8B'})

# Merge dataframes on the Metric column
merged_df = pd.merge(df2[['Metric', 'Best Score llama3-8B']], df1_long, on='Metric', how='outer')

# Set the order of the metrics
order = ['Accuracy', 'Precision', 'Recall', 'F1_score']
merged_df['Metric'] = pd.Categorical(merged_df['Metric'], categories=order, ordered=True)
merged_df = merged_df.sort_values('Metric')

# Plotting the results
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(merged_df['Metric']))

# Plot bars with llama3-8B on the left (deep blue) and bert on the right (teal)
plt.bar([i - bar_width/2 for i in index], merged_df['Best Score llama3-8B'], bar_width, label='llama3-8B', color='#1f77b4')  # deep blue
plt.bar([i + bar_width/2 for i in index], merged_df['Best Score bert'], bar_width, label='bert', color='#2ca02c')  # teal

# Add labels, title, and legend
plt.xlabel('Metrics')
plt.ylabel('Best Score')
plt.title('Comparison of Optimal Performance Metrics (Binary Problem)')
plt.xticks(index, merged_df['Metric'], rotation=45)
plt.legend()

# Save the plot
plot_path = os.path.join(comparison_path, 'binary_performance_comparison.png')
plt.tight_layout()
plt.savefig(plot_path)
print(f'Plot saved to {plot_path}')
plt.show()
