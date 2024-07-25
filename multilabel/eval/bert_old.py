# evaluate_model.py

import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import os
from sklearn.preprocessing import MultiLabelBinarizer

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))

# Load validation data
test_df = pd.read_csv(os.path.join(MULTILABEL_ROOT, 'data', 'test.csv'))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts
test_texts = test_df['text'].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Encode the labels
mlb = MultiLabelBinarizer()
test_labels = mlb.fit_transform(test_df['topics'].apply(eval))

# Dataset class
class AbstractDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

test_dataset = AbstractDataset(test_encodings, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Load the model
model = BertForSequenceClassification.from_pretrained(os.path.join(MULTILABEL_ROOT, 'train', 'bert', 'bert_final'))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model at different thresholds
def evaluate_at_thresholds(model, dataloader, thresholds, device):
    model.eval()
    metrics = []
    
    for threshold in thresholds:
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            # Move batch data to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                probs = torch.sigmoid(outputs.logits)
                preds = probs > threshold
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        f1 = f1_score(all_labels, all_preds, average='micro')
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
        
        metrics.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        })
    
    return metrics

# Define thresholds to evaluate
thresholds = np.arange(0.1, 1.0, 0.1)

# Evaluate the model at the defined thresholds
metrics_list = evaluate_at_thresholds(model, test_dataloader, thresholds, device)

# Save the metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(MULTILABEL_ROOT, 'results', 'bert', 'threshold_sweep.csv'), index=False)

# Print the metrics
print(metrics_df)
