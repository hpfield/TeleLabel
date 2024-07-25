from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
import os
import argparse
from datetime import datetime

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))

def main(data_file):

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

    # Load the input data
    df = pd.read_csv(data_file)

    # Get texts
    texts = df['text'].to_list()

    # Get predictions
    predictions = classify_texts(texts)

    # Create a new DataFrame with the texts and predicted labels
    output_df = pd.DataFrame({'text': texts, 'isTelecoms': predictions})

    # Generate a readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure the output directory exists
    output_dir = os.path.join(BINARY_ROOT, 'outputs', 'bert')
    os.makedirs(output_dir, exist_ok=True)

    # Save the predictions to a CSV file with timestamp
    output_file = os.path.join(output_dir, f'{timestamp}.csv')
    output_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a dataset and store the predictions.')
    parser.add_argument('data_file', type=str, nargs='?', default=os.path.join(BINARY_ROOT, 'data', 'test.csv'), help='The file path to the data to run inference on')
    args = parser.parse_args()

    main(args.data_file)
