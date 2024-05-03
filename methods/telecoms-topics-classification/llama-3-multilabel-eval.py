import pandas as pd

POSSIBLE_TOPICS = 22
data_path = '/home/rz20505/Documents/ask-jgi/data/labelled/llama-3-multilabel-classification/'
results_file = '/home/rz20505/Documents/ask-jgi/eval/results/llama-3-multilabel-classification.csv'
best_results_file = '/home/rz20505/Documents/ask-jgi/eval/results/llama-3-multilabel-classification-best.csv'
tel_topic_match = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]

results = []

# Evaluate the performance of the LLM with each threshold from 0.1 to 1.0, incrementing by 0.1 each time
for t in range(1, 11):
    thresh = t / 10.0

    # Evaluate the ability of the model to predict topics with the current confidence threshold with all chunk sizes
    for chunk_size in range(1, POSSIBLE_TOPICS):
        file_path = data_path + f'cordis-telecoms-chunk_size-{chunk_size}.json'
        df = pd.read_json(file_path, lines=True)

        gt = df['gt']
        scores = df['topics']

        # Process predictions based on threshold
        predictions = scores.apply(lambda d: [k for k, v in d.items() if v >= thresh])

        # Compute metrics for each datapoint
        metrics_list = []
        for gt_set, pred_set in zip(gt, predictions):
            true_positives = len(set(gt_set) & set(pred_set))
            false_positives = len(set(pred_set) - set(gt_set))
            false_negatives = len(set(gt_set) - set(pred_set))
            true_negatives = len(set(tel_topic_match) - set(gt_set) - set(pred_set))

            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

            metrics_list.append((true_positives, false_positives, false_negatives, true_negatives, precision, recall, f1, accuracy))

        total_true_positives = sum(x[0] for x in metrics_list)
        total_false_positives = sum(x[1] for x in metrics_list)
        total_false_negatives = sum(x[2] for x in metrics_list)
        total_true_negatives = sum(x[3] for x in metrics_list)

        total_accuracy = sum(x[7] for x in metrics_list) / len(metrics_list)
        aggregated_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) else 0
        aggregated_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) else 0
        aggregated_f1 = 2 * (aggregated_precision * aggregated_recall) / (aggregated_precision + aggregated_recall) if (aggregated_precision + aggregated_recall) else 0

        row = [thresh, chunk_size, aggregated_precision, aggregated_recall, aggregated_f1, total_accuracy]
        results.append(row)

results_df = pd.DataFrame(results, columns=['Threshold', 'Chunk Size', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
results_df.to_csv(results_file)

# Get best performance
metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy'] 
best_rows = []

for metric in metrics:
    best_row = results_df.loc[results_df[metric].idxmax()] 
    best_rows.append({
        'Metric': metric,
        'Best Score': best_row[metric],
        'Threshold': best_row['Threshold'],
        'Chunk Size': best_row['Chunk Size']
    })

best_df = pd.DataFrame(best_rows)

best_df = best_df[['Metric', 'Best Score', 'Threshold', 'Chunk Size']]
best_df.to_csv(best_results_file)

print('Evaluation completed.')
