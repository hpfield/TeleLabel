import pandas as pd
import numpy as np
import torch
import os
import argparse
import json
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))

# Default file paths
DEFAULT_DATA_PATH = os.path.join(MULTILABEL_ROOT, 'data', 'test.csv')
BEST_CSV_PATH = os.path.join(MULTILABEL_ROOT, 'results', 'bert', 'best.csv')
OUTPUT_DIR = os.path.join(MULTILABEL_ROOT, 'outputs', 'bert')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hard-code the list of available labels
available_labels = ["teleology", "telecommunications", "radio frequency", "radar", "mobile phones", "bluetooth",
                    "WiFi", "data networks", "optical networks", "microwave technology", "radio technology",
                    "mobile radio", "4G", "LiFi", "mobile network", "radio and television", "satellite radio",
                    "telecommunications networks", "5G", "fiber-optic network", "cognitive radio",
                    "fixed wireless network"]

# Dataset class
class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

def get_best_threshold(metric):
    best_df = pd.read_csv(BEST_CSV_PATH)
    best_row = best_df[best_df['Metric'] == metric]
    if best_row.empty:
        raise ValueError(f"No best threshold found for metric {metric}")
    return best_row['Threshold'].values[0]

def run_inference(data_path, metric):
    # Get the best threshold for the given metric
    threshold = get_best_threshold(metric)
    print(f'Threshold: {threshold}')
    threshold=0.5

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(os.path.join(MULTILABEL_ROOT, 'train', 'bert', 'bert_final'))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read the input CSV file
    input_df = pd.read_csv(data_path)

    # Extract the 'text' column
    texts = input_df['text'].tolist()

    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    # Create the dataset and dataloader
    dataset = InferenceDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=8)

    # Run inference
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            probs = torch.sigmoid(outputs.logits)
            all_preds.append(probs.cpu().numpy())

    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Convert predictions to topic lists
    topics = []
    for pred in all_preds:
        topic_list = [available_labels[i] for i, p in enumerate(pred) if p > threshold]
        topics.append(topic_list)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the output JSON structure
    output_data = [{'text': text, 'topics': topic_list} for text, topic_list in zip(texts, topics)]

    # Save the predictions to a JSON file
    output_file_path = os.path.join(OUTPUT_DIR, f'optimising_{metric}_{timestamp}.json')
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Predictions saved to {output_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a dataset and store the predictions.')
    parser.add_argument('data_file', type=str, nargs='?', default=DEFAULT_DATA_PATH, help='Path to the data file for inference')
    parser.add_argument('--metric', type=str, default='F1_score', help='Performance metric to determine the threshold (e.g., F1_score, Precision, Recall, Accuracy)')
    args = parser.parse_args()
    
    run_inference(args.data_file, args.metric)
