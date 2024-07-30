from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
import os
import argparse
from datetime import datetime
from tqdm import tqdm  # Importing tqdm for progress bar

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))

def main(args):

    # Load the trained model and tokenizer
    model_path = os.path.join(BINARY_ROOT, 'train', 'bert', 'bert_final')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Function to classify new data points in batches
    def classify_texts(texts, batch_size=16):
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):  # Adding tqdm progress bar
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
            outputs = model(**inputs)
            batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
        return predictions

    # Load the input data
    df = pd.read_csv(args.data_file)

    # Get texts
    texts = df['text'].to_list()

    # Get predictions
    predictions = classify_texts(texts)

    # Create a new DataFrame with the texts and predicted labels
    output_df = pd.DataFrame({'text': texts, 'isTelecoms': predictions})    

    # Save the predictions to a CSV file with timestamp
    output_df.to_csv(args.output_file, index=False)

    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(BINARY_ROOT, 'outputs', 'bert')
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description='Run inference on a dataset and store the predictions.')
    parser.add_argument('--data_file', type=str, nargs='?', default=os.path.join(BINARY_ROOT, 'data', 'test.csv'), help='The file path to the data to run inference on')
    parser.add_argument('--output_file', type=str, nargs='?', default=os.path.join(output_dir, f'{timestamp}.csv'), help='The path to and name of desired output')
    args = parser.parse_args()

    main(args)
