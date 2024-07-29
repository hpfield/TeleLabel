import torch
import pandas as pd
from llama import Llama
import fire
from typing import Optional
import time
import re
import ast
import os
from tqdm import tqdm
from datetime import datetime, timedelta

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '..', '..', '..'))

def save_checkpoint(data, file_path):
    headers = ['text', 'gt', 'preds']
    df_new = pd.DataFrame(data, columns=headers)
    if not os.path.exists(file_path):
        df_new.to_json(file_path, orient='records', lines=True)
    else:
        df_existing = pd.read_json(file_path, lines=True)
        df_concatenated = pd.concat([df_existing, df_new], ignore_index=True)
        df_concatenated.to_json(file_path, orient='records', lines=True)

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        df_existing = pd.read_json(file_path, lines=True)
        return df_existing
    else:
        return pd.DataFrame(columns=['text', 'gt', 'preds'])

def main(
    ckpt_dir: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct'),
    tokenizer_path: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct', 'tokenizer.model'),
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
    incomplete_output_file: Optional[str] = None
):

    # Ensure the model checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Model checkpoint directory not found: {ckpt_dir}")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    file_path = os.path.join(MULTILABEL_ROOT, 'data', 'cordis-multilabel-telecoms.csv')
    base_checkpoint_path = os.path.join(MULTILABEL_ROOT, 'outputs', 'llama3-8B')
    # Create directory if it doesn't exist
    if not os.path.exists(base_checkpoint_path):
        os.makedirs(base_checkpoint_path)

    df = pd.read_csv(file_path)
    df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else [])
    num_items = df.shape[0]

    # Determine the start index based on the incomplete output file
    start_idx = 0
    if incomplete_output_file:
        df_checkpoint = load_checkpoint(incomplete_output_file)
        start_idx = len(df_checkpoint)
        processed_data = df_checkpoint.values.tolist()
        checkpoint_path = incomplete_output_file
        print(f"Resuming from checkpoint: {incomplete_output_file}, starting at index {start_idx}")
    else:
        processed_data = []
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = os.path.join(base_checkpoint_path, f'cordis-telecoms_{timestamp}.json')
        print(f"Starting new output file: {checkpoint_path}")

    with open(os.path.join(MULTILABEL_ROOT, 'run', 'prompts', 'system-prompt.txt'), 'r') as f:
        system_prompt = f.read()

    with open(os.path.join(MULTILABEL_ROOT, 'run', 'prompts', 'user-prompt.txt')) as f:
        user_prompt = f.read()    

    tel_topic_match = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]
    tel_topic_match = [f"'{label}'" for label in tel_topic_match]
    total_topics_num = len(tel_topic_match)

    outer_start_time = time.time()

    try:
        # Break up the topics list into chunks, starting at a single topic and working all the way up to all the topics
        with tqdm(total=total_topics_num, desc="Overall Progress", unit='chunk') as outer_pbar:
            for topics_num in range(1, total_topics_num + 1):
                num_topics_chunks = total_topics_num // topics_num
                remainder = total_topics_num % topics_num
                topics_chunks = [tel_topic_match[i * topics_num:(i + 1) * topics_num] for i in range(num_topics_chunks)]
                if remainder:
                    topics_chunks.append(tel_topic_match[num_topics_chunks * topics_num:])

                start_time = time.time()

                with tqdm(total=num_items, desc=f"Processing chunks of size {topics_num}", unit='sample', initial=start_idx) as pbar:
                    batch_data = []
                    for idx, row in df.iterrows():
                        if idx < start_idx:
                            pbar.update(1)
                            continue

                        text = row['text']
                        gt = row['topics']
                        full_scores = {}

                        for topics_chunk in topics_chunks:
                            dialog = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt.format(text=text, labels_list=", ".join(topics_chunk))}
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
                                    scores = ast.literal_eval(generation_content)
                                    if isinstance(scores, dict):
                                        # Filter scores to include only valid labels
                                        valid_scores = {k: v for k, v in scores.items() if k in topics_chunk}
                                        full_scores.update(valid_scores)
                                    else:
                                        scores = {"format_error": 0.0}
                                except ValueError:
                                    scores = {"value_error": 0.0}

                            except Exception as e:
                                scores = {"gen_error": 0.0}

                        batch_data.append([text, gt, full_scores])
                        processed_data.append([text, gt, full_scores])

                        if len(batch_data) >= 5:
                            save_checkpoint(batch_data, checkpoint_path)
                            batch_data = []

                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / (idx + 1 - start_idx)) * (num_items - (idx + 1))
                        remaining_td = timedelta(seconds=int(remaining_time))
                        pbar.set_postfix({
                            'Elapsed': str(timedelta(seconds=int(elapsed_time))),
                            'Remaining': str(remaining_td)
                        })
                        pbar.update(1)

                    if batch_data:
                        save_checkpoint(batch_data, checkpoint_path)

                outer_elapsed_time = time.time() - outer_start_time
                outer_remaining_time = (outer_elapsed_time / topics_num) * (total_topics_num - topics_num)
                outer_remaining_td = timedelta(seconds=int(outer_remaining_time))
                outer_pbar.set_postfix({
                    'Elapsed': str(timedelta(seconds=int(outer_elapsed_time))),
                    'Remaining': str(outer_remaining_td)
                })
                outer_pbar.update(1)
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Data labeling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
