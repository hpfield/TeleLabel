# from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer, EarlyStoppingCallback

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model and tokenizer
model_path = './roberta_final_model'  # Ensure this path matches the output_dir in your training script
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

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
df = pd.read_csv('../../data/ml/binary/test.csv')

# Get texts and true labels
texts = df['combined_text'].to_list()
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
print(f'F1 Score: {f1:.4f}')
