from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, EarlyStoppingCallback

from datasets import load_dataset
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))

# Function to compute additional metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Load datasets from CSV files
dataset = load_dataset('csv', data_files={'train': os.path.join(BINARY_ROOT, 'data', 'train.csv'), 'test': os.path.join(BINARY_ROOT, 'data', 'test.csv')})

# Rename columns to fit our script
dataset = dataset.rename_column('isTelecoms', 'label')

# Convert boolean labels to integers (True -> 1, False -> 0)
def preprocess_labels(example):
    example['label'] = int(example['label'])
    return example

dataset = dataset.map(preprocess_labels)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = []

# Cross-validation loop
for train_index, val_index in kf.split(tokenized_datasets['train']):
    # Initialize model for each fold to ensure independence
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    train_dataset = tokenized_datasets['train'].select(train_index)
    val_dataset = tokenized_datasets['train'].select(val_index)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(BINARY_ROOT, 'train', 'bert'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(BINARY_ROOT, 'train', 'bert', 'logs'),
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    # Initialize Trainer with Early Stopping and Metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()
    results.append(eval_result)

# Average results over all folds
avg_results = {metric: np.mean([result[metric] for result in results]) for metric in results[0].keys()}

print(f'Average results: {avg_results}')

# At the end of the training script
final_model_dir = os.path.join(BINARY_ROOT, 'train', 'bert', 'bert_final')
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
