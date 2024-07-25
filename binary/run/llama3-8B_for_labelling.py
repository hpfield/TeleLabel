import pandas as pd
from llama import Llama
import fire
from typing import Optional
import os
from datetime import datetime, timedelta
from tqdm import tqdm

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))

def save_checkpoint(data, file_path):
    headers = ['text', 'isTelecoms']
    df_new = pd.DataFrame(data, columns=headers)
    if not os.path.exists(file_path):
        df_new.to_json(file_path, orient='records', lines=True)
    else:
        df_existing = pd.read_json(file_path, lines=True)
        df_concatenated = pd.concat([df_existing, df_new], ignore_index=True)
        df_concatenated.to_json(file_path, orient='records', lines=True)

def main(
    metric: str = 'F1_score',
    ckpt_dir: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct'),
    tokenizer_path: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct', 'tokenizer.model'),
    file_path: str = os.path.join(BINARY_ROOT, 'data', 'all_samples.csv'),
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    metrics_path = os.path.join(BINARY_ROOT, 'results', 'llama3-8B', 'best.csv')
    base_checkpoint_path = os.path.join(BINARY_ROOT, 'data', 'labelled', 'llama3-8B')
    os.makedirs(base_checkpoint_path, exist_ok=True)
    df = pd.read_csv(file_path)
    metrics_df = pd.read_csv(metrics_path)

    # Get the confidence threshold for the desired metric
    threshold = metrics_df.loc[metrics_df['Metric'] == metric, 'Threshold'].values[0]
    print(f'Threshold: {threshold}')

    with open(os.path.join(BINARY_ROOT, 'run', 'prompts', 'system-prompt.txt')) as f:
        system_prompt = f.read()

    with open(os.path.join(BINARY_ROOT, 'run', 'prompts', 'user-prompt.txt')) as f:
        user_prompt = f.read()    

    # Generate a readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = os.path.join(base_checkpoint_path, f'{timestamp}.json')
    batch_data = []

    start_time = datetime.now()

    positive_count = 0
    negative_count = 0

    with tqdm(total=len(df), desc="Processing samples", unit="sample") as pbar:
        for idx, row in df.iterrows():
            text = row['text']
            
            # Generate confidence scores for each chunk using LLM
            dialog = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(text=text)}
            ]
            
            try:
                results = generator.chat_completion(
                    [dialog],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                generation_content = results[0]['generation']['content']

                try:
                    score = float(generation_content)
                except ValueError:
                    score = -1  

            except Exception as e:
                score = -1

            is_telecoms = 1 if score >= threshold else 0
            if is_telecoms:
                positive_count += 1
            else:
                negative_count += 1

            batch_data.append([text, is_telecoms])
            
            if (idx + 1) % 10 == 0:
                elapsed_time = datetime.now() - start_time
                avg_time_per_sample = elapsed_time / (idx + 1)
                samples_remaining = len(df) - (idx + 1)
                estimated_time_remaining = avg_time_per_sample * samples_remaining
                hours, remainder = divmod(estimated_time_remaining.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                pbar.set_postfix({
                    'Positives': positive_count,
                    'Negatives': negative_count,
                    'ETA': f'{int(hours)}h {int(minutes)}m {int(seconds)}s'
                })
            
            if (idx + 1) % 100 == 0:
                save_checkpoint(batch_data, checkpoint_path)
                batch_data = []  # Reset the batch data list

            pbar.update(1)

    if batch_data:
        save_checkpoint(batch_data, checkpoint_path)

    print("Data labelling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
