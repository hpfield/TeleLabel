import pandas as pd
import os
import ast
import numpy as np

POSSIBLE_TOPICS = 22
data_path = '/home/rz20505/Documents/ask-jgi/data/labelled/llama-3-multilabel-classification/'
results_file = '/home/rz20505/Documents/ask-jgi/eval/results/llama-3-multilabel-classification.csv'
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
        
        # If the predicted label is equal to or over the threshold, add it to the final predicted labels for this datapoint
        predictions = scores.apply(lambda d: [k for k, v in d.items() if v > thresh])

        # Binary representation of ground truth and predictions
        y_true = np.array([[(topic in gt_i) for topic in tel_topic_match] for gt_i in gt])
        y_pred = np.array([[(topic in pred_i) for topic in tel_topic_match] for pred_i in predictions])
        
        # Element-wise
        accuracy = (y_true == y_pred).mean()
        # Row-wise
        accuracy_datapoints = (y_true.all(axis=1) == y_pred.all(axis=1)).mean()

        # All others are element-wise
        precision = (np.logical_and(y_true, y_pred).sum(axis=0) / y_pred.sum(axis=0)).mean() if y_pred.sum(axis=0).any() else 0
        recall = (np.logical_and(y_true, y_pred).sum(axis=0) / y_true.sum(axis=0)).mean() if y_true.sum(axis=0).any() else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        # Add metrics to row
        row = [thresh, chunk_size, accuracy, accuracy_datapoints, precision, recall, f1_score]
        results.append(row)

results_df = pd.DataFrame(results, columns=['Threshold', 'Chunk Size', 'Accuracy', 'Accuracy Datapoints', 'Precision', 'Recall', 'F1 Score'])
results_df.to_csv(results_file)
print('Finished')