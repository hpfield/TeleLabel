import pandas as pd
from llama import Llama
import fire
from typing import Optional
import os
from datetime import datetime

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))

def save_checkpoint(data, file_path):
    headers = ['text', 'gt', 'conf']
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
    ckpt_dir: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct'),
    tokenizer_path: str = os.path.join(PROJECT_ROOT, 'Meta-Llama-3-8B-Instruct', 'tokenizer.model'),
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

    file_path = os.path.join(BINARY_ROOT, 'data', 'cordis-binary-telecoms.csv')
    base_checkpoint_path = os.path.join(BINARY_ROOT, 'outputs', 'llama3-8B')
    df = pd.read_csv(file_path)
    # df = df.iloc[:10]
    print("Starting!")
    num_items = df.shape[0]
    print('Num items: ',num_items)

    with open(os.path.join(BINARY_ROOT, 'run', 'prompts', 'system-prompt.txt')) as f:
        system_prompt = f.read()

    with open(os.path.join(BINARY_ROOT, 'run', 'prompts', 'user-prompt.txt')) as f:
        user_prompt = f.read()    

    # Generate a readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save a separate json file for each round of predictions
    checkpoint_path = os.path.join(base_checkpoint_path, f'{timestamp}.json')
    batch_data = []
    
    for idx, row in df.iterrows():
        text = row['text']
        gt = row['isTelecoms']
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
                print(f'Processed score: {score}')
            except ValueError:
                print(f"Value error: {generation_content}")
                score = -1  

        except Exception as e:
            print(f"Results generation error: {e}")
            print(generation_content)
            score = -1

        batch_data.append([text, gt, score])
    save_checkpoint(batch_data, checkpoint_path)
    batch_data = []  # Reset the batch data list


    print("Data labeling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
