import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import os
import json

# Load the dataset
df = pd.read_csv('../../data/cordis-multilabel-telecoms.csv')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the abstracts
tokenized_data = tokenizer(list(df['text'].values), padding='max_length', truncation=True, max_length=512, return_tensors="pt")

# Build the label list
unique_labels = sorted(set(label for sublist in df['topics'].apply(eval).tolist() for label in sublist))
label_map = {label: i for i, label in enumerate(unique_labels)}

# Encode the labels
def encode_labels(labels):
    label_ids = [0] * len(label_map)
    for label in labels:
        if label in label_map:
            label_ids[label_map[label]] = 1
    return label_ids

encoded_labels = df['topics'].apply(lambda x: encode_labels(eval(x)))

# Convert to lists
input_ids = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']
labels = torch.tensor(encoded_labels.tolist()).float()

# Dataset class
class AbstractDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# Create the dataset
dataset = AbstractDataset(input_ids, attention_masks, labels)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map), problem_type="multi_label_classification")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 100

# Define the training arguments
training_args = TrainingArguments(
    output_dir=f'./results_{epochs}',
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'./logs_{epochs}',
    logging_steps=10,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

# Function to compute metrics for evaluation
def compute_metrics(p):
    logits = torch.tensor(p.predictions)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float().cpu().numpy()
    labels = p.label_ids

    f1 = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro')
    accuracy = accuracy_score(labels, preds)  # Calculate accuracy

    # Log detailed metrics
    print("Detailed Metrics:")
    print(f"F1: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")

    for i in range(5):  # Print some sample predictions
        print(f"Sample {i}:")
        print(f"Predicted: {preds[i]}")
        print(f"Labels: {labels[i]}")

    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

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
metrics_list = evaluate_at_thresholds(model, val_dataloader, thresholds, device)

# Save the metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f'metrics_threshold_tuning_{epochs}_epochs.csv', index=False)

# Save the model
model.save_pretrained(f'./model_{epochs}')

# Save the tokenizer
tokenizer.save_pretrained(f'./model_{epochs}')

# Print the metrics
print(metrics_df)
