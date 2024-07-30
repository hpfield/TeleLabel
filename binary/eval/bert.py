from transformers import BertForSequenceClassification, BertTokenizer

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))

# Load the trained model and tokenizer
model_path = os.path.join(BINARY_ROOT, 'train', 'bert', 'bert_final')
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to classify new data points in batches
def classify_texts(texts, batch_size=16):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions.extend(batch_predictions)
    return predictions

# Load the test data
df = pd.read_csv(os.path.join(BINARY_ROOT, 'data', 'test.csv'))

# Get texts and true labels
texts = df['text'].to_list()
true_labels = df['isTelecoms'].astype(int).to_list()  # Convert True/False to 1/0

# Get predictions
predictions = classify_texts(texts)  # Convert to numpy array for sklearn functions

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1_score: {f1:.4f}')

# Save the metrics to a CSV file
metrics = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_score': [f1]
}

metrics_df = pd.DataFrame(metrics)
metrics_file_path = os.path.join(BINARY_ROOT, 'results', 'bert', 'bert_metrics.csv')
os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
metrics_df.to_csv(metrics_file_path, index=False)
print(f'Metrics saved to {metrics_file_path}')
