import torch
import pandas as pd
from llama import Llama
import fire
from typing import Optional
import time
import re
import ast
import os

def save_checkpoint(data, file_path, mode='a'):
    """ Helper function to save data to a CSV file """
    df = pd.DataFrame(data, columns=['description', 'name', 'topics'])
    df.to_csv(file_path, mode=mode, header=mode=='w', index=False)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    file_path = '/home/rz20505/Documents/ask-jgi/data/unlabelled-desc-name-topics.csv'
    checkpoint_path = '/home/rz20505/Documents/ask-jgi/data/labelled/llama-3-zero-shot.csv'
    df = pd.read_csv(file_path)
    
    start_index = 0
    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_index = len(checkpoint_df)
        print(f"Resuming from index {start_index}")

    with open('/home/rz20505/Documents/ask-jgi/methods/utils/prompts/system-prompt-llama-3.txt', 'r') as f:
        system_prompt = f.read()

    total_time = 0
    batch_data = []
    topic_pattern = re.compile(r'\["[^"]*"(?:, "[^"]*")*\]') 

    for idx, row in df.iloc[start_index:].iterrows():
        name = row['name']
        description = row['description']
        user_input = f"{system_prompt} \nTitle: \n{name} \nDescription: \n{description}"
        
        dialog = [{"role": "user", "content": user_input}]
        
        start_time = time.time()
        try:
            results = generator.chat_completion(
                [dialog],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            generation_content = results[0]['generation']['content']
            match = topic_pattern.search(generation_content)
            if match:
                topics_list = ast.literal_eval(match.group())
            else:
                topics_list = ["Parsing Failed", generation_content]

        except Exception as e:
            print(f"An error occurred: {e}")
            topics_list = ["Error in generation"]
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        batch_data.append([description, name, topics_list])
        print(f"Processed item {idx+1}/{len(df)}: Time taken = {elapsed_time:.2f}s")

        # Batch save every 100 entries
        if (idx+1) % 100 == 0:
            save_checkpoint(batch_data, checkpoint_path, mode='a')
            batch_data = []  # Reset the batch data list

        torch.cuda.empty_cache()

    # Save remaining entries if any
    if batch_data:
        save_checkpoint(batch_data, checkpoint_path, mode='a')

    print(f"Data labeling complete and file saved. Average generation time: {total_time/len(df):.2f}s")

if __name__ == "__main__":
    fire.Fire(main)
