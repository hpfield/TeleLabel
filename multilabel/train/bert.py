import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))

# Load train and test data
train_df = pd.read_csv(os.path.join(MULTILABEL_ROOT, 'data', 'train.csv'))
test_df = pd.read_csv(os.path.join(MULTILABEL_ROOT, 'data', 'test.csv'))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts
train_texts = train_df['text'].tolist()
test_texts = test_df['text'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Hard-code the list of available labels
available_labels = ["teleology", "telecommunications", "radio frequency", "radar", "mobile phones", "bluetooth",
                    "WiFi", "data networks", "optical networks", "microwave technology", "radio technology",
                    "mobile radio", "4G", "LiFi", "mobile network", "radio and television", "satellite radio",
                    "telecommunications networks", "5G", "fiber-optic network", "cognitive radio",
                    "fixed wireless network"]

# Encode the labels using the hard-coded list
mlb = MultiLabelBinarizer(classes=available_labels)
train_labels = mlb.fit_transform(train_df['topics'].apply(eval))
test_labels = mlb.transform(test_df['topics'].apply(eval))

# Ensure label shape consistency
num_labels = train_labels.shape[1]
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

train_dataset = AbstractDataset(train_encodings, train_labels)
test_dataset = AbstractDataset(test_encodings, test_labels)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, problem_type="multi_label_classification")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(MULTILABEL_ROOT, 'train', 'bert'),
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(MULTILABEL_ROOT, 'train', 'bert', 'logs'),
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

    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.join(MULTILABEL_ROOT, 'train', 'bert', 'bert_final'))
