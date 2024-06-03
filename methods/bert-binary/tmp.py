from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, EarlyStoppingCallback

# Assuming tokenizer and model are defined and trained
model_path = './results/checkpoint-420'

# Load model and tokenizer from checkpoint directory
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use base tokenizer if not saved explicitly

# Save final model and tokenizer to a new directory
final_model_dir = './final_model'
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
