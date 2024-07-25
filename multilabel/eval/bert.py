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

# Hard-code the list of available labels
available_labels = ["teleology", "telecommunications", "radio frequency", "radar", "mobile phones", "bluetooth",
                    "WiFi", "data networks", "optical networks", "microwave technology", "radio technology",
                    "mobile radio", "4G", "LiFi", "mobile network", "radio and television", "satellite radio",
                    "telecommunications networks", "5G", "fiber-optic network", "cognitive radio",
                    "fixed wireless network"]

# Encode the labels
mlb = MultiLabelBinarizer(classes=available_labels)
test_labels = mlb.fit_transform(test_df['topics'].apply(eval))

# Ensure label shape consistency
num_labels = test_labels.shape[1]
print(f"Number of labels: {num_labels}")

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
model = BertForSequenceClassification.from_pretrained(os.path.join(MULTILABEL_ROOT, 'train', 'bert', 'bert_final'), num_labels=num_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model at different thresholds
def evaluate_at_thresholds(model, dataloader, thresholds, device):
    model.eval()
    metrics = []
    best_scores = {
        'Precision': {'score': 0, 'threshold': 0},
        'Recall': {'score': 0, 'threshold': 0},
        'F1_score': {'score': 0, 'threshold': 0},
        'Accuracy': {'score': 0, 'threshold': 0}
    }
    
    for threshold in thresholds:
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            # Move batch data to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        
        # Check consistency before concatenating
        print(f"Preds shapes: {[p.shape for p in all_preds]}")
        print(f"Labels shapes: {[l.shape for l in all_labels]}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Ensure consistency in the number of labels
        if all_preds.shape[1] != all_labels.shape[1]:
            raise ValueError("Mismatch in the number of labels between predictions and true labels")
        
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
        
        metrics.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        })

        # Update best scores
        if precision > best_scores['Precision']['score']:
            best_scores['Precision']['score'] = precision
            best_scores['Precision']['threshold'] = threshold
        if recall > best_scores['Recall']['score']:
            best_scores['Recall']['score'] = recall
            best_scores['Recall']['threshold'] = threshold
        if f1 > best_scores['F1_score']['score']:
            best_scores['F1_score']['score'] = f1
            best_scores['F1_score']['threshold'] = threshold
        if accuracy > best_scores['Accuracy']['score']:
            best_scores['Accuracy']['score'] = accuracy
            best_scores['Accuracy']['threshold'] = threshold
    
    return metrics, best_scores

# Define thresholds to evaluate
thresholds = np.arange(0.1, 1.0, 0.1)

# Evaluate the model at the defined thresholds
metrics_list, best_scores = evaluate_at_thresholds(model, test_dataloader, thresholds, device)

# Save the metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(MULTILABEL_ROOT, 'results', 'bert', 'threshold_sweep.csv'), index=False)

# Prepare and save the best scores
best_scores_df = pd.DataFrame([
    {'Metric': 'Precision', 'Best Score': best_scores['Precision']['score'], 'Threshold': best_scores['Precision']['threshold']},
    {'Metric': 'Recall', 'Best Score': best_scores['Recall']['score'], 'Threshold': best_scores['Recall']['threshold']},
    {'Metric': 'F1_score', 'Best Score': best_scores['F1_score']['score'], 'Threshold': best_scores['F1_score']['threshold']},
    {'Metric': 'Accuracy', 'Best Score': best_scores['Accuracy']['score'], 'Threshold': best_scores['Accuracy']['threshold']}
])
best_scores_df.to_csv(os.path.join(MULTILABEL_ROOT, 'results', 'bert', 'best.csv'), index=False)

# Print the metrics
print(metrics_df)
print(best_scores_df)
