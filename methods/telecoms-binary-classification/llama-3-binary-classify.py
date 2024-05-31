import torch
import pandas as pd
from llama import Llama
import fire
from typing import Optional
import time
import re
import ast
import os

def save_checkpoint(data, file_path):
    headers = ['description', 'name', 'gt', 'conf']
    df_new = pd.DataFrame(data, columns=headers)
    print('Saving Checkpoint')
    if not os.path.exists(file_path):
        df_new.to_json(file_path, orient='records', lines=True)
    else:
        df_existing = pd.read_json(file_path, lines=True)
        df_concatenated = pd.concat([df_existing, df_new], ignore_index=True)
        df_concatenated.to_json(file_path, orient='records', lines=True)
        print(f'{df_concatenated.shape[0]} processed in {"/".split(file_path)[-1]}')
    

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
):
    
    os.chdir('../..')

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    file_path = 'data/cordis-binary-telecoms.csv'
    base_checkpoint_path = 'data/labelled/llama-3-binary-classification/'
    df = pd.read_csv(file_path)
    # df = df.iloc[:10]
    print("Starting!")
    num_items = df.shape[0]
    print('Num items: ',num_items)

    with open('methods/telecoms-binary-classification/prompts/system-prompt.txt', 'r') as f:
        system_prompt = f.read()

    with open('methods/telecoms-binary-classification/prompts/user-prompt.txt') as f:
        user_prompt = f.read()    


    # Save a separate json file for each round of predictions
    checkpoint_path = base_checkpoint_path + f'cordis-telecoms-binary.json'
    batch_data = []
    
    for idx, row in df.iterrows():
        name = row['name']
        description = row['description']
        gt = row['isTelecoms']
        # Generate confidence scores for each chunk using LLM
                    
        dialog = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(title=name, abstract=description)}
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
                print(f'Processed score: {score}')
            except ValueError:
                print(f"Value error: {generation_content}")
                score = -1  

        except Exception as e:
            print(f"Results generation error: {e}")
            print(generation_content)
            score = -1

        batch_data.append([description, name, gt, score])
    save_checkpoint(batch_data, checkpoint_path)
    batch_data = []  # Reset the batch data list


    print("Data labeling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
