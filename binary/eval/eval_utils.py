import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os


def get_results(data_path):

    # Evaluate the ability of the model to predict topics with the current confidence threshold with all chunk sizes
    # file_path = data_path + f'cordis-telecoms-binary.json'
    df = pd.read_json(data_path, lines=True)

    gt = df['gt']
    scores = df['conf']

    results = []

    # Evaluate the performance of the LLM with each threshold from 0.1 to 1.0, incrementing by 0.1 each time
    for t in range(1, 11):
        thresh = t / 10.0

        # Process predictions based on threshold
        predictions = scores >= thresh

        # Calculate metrics
        precision = precision_score(gt, predictions)
        recall = recall_score(gt, predictions)
        f1 = f1_score(gt, predictions)
        accuracy = accuracy_score(gt, predictions)

        # Append the results
        results.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1,
            'Accuracy': accuracy
        })

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)

    return results_df

def save_best(metrics, results_df, best_results_file):
    best_rows = []

    for metric in metrics:
        best_row = results_df.loc[results_df[metric].idxmax()] 
        best_rows.append({
            'Metric': metric,
            'Best Score': best_row[metric],
            'Threshold': best_row['Threshold'],
        })

    best_df = pd.DataFrame(best_rows)

    best_df = best_df[['Metric', 'Best Score', 'Threshold']]
    best_df.to_csv(best_results_file, index=False)

def plot_metrics(metrics, results_df, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in metrics:
        plt.figure()
        plt.plot(results_df['Threshold'], results_df[metric], marker='o')
        plt.xlabel('Threshold')
        plt.ylabel(metric.capitalize())
        plt.title(f'Threshold vs {metric.capitalize()}')
        plt.grid(True)
        
        # Save the plot with "binary" in the filename
        plot_filename = f'{metric}.png'
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()