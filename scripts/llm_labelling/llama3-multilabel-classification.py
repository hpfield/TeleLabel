import torch
import pandas as pd
from llama import Llama
import fire
from typing import Optional
import time
import re
import ast
import os
import hydra
from omegaconf import DictConfig
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))


def save_checkpoint(data, file_path):
    headers = ['description', 'name', 'gt', 'topics']
    df_new = pd.DataFrame(data, columns=headers)
    print('Saving Checkpoint')
    if not os.path.exists(file_path):
        df_new.to_json(file_path, orient='records', lines=True)
    else:
        df_existing = pd.read_json(file_path, lines=True)
        df_concatenated = pd.concat([df_existing, df_new], ignore_index=True)
        df_concatenated.to_json(file_path, orient='records', lines=True)
        print(f'{df_concatenated.shape[0]} processed in {"/".split(file_path)[-1]}')
    
@hydra.main(config_path="../../conf", config_name="config", version_base='1.3.2')
def main(
    cfg: DictConfig,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
    prompt_filename: str = 'user-and-system-prompt.txt'
):
    
    os.chdir('../..')
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # file_path = 'data/cordis-telecoms.csv'
    # base_checkpoint_path = 'data/labelled/llama-3-multilabel/'
    base_checkpoint_path = os.path.join(PROJECT_ROOT, cfg.data.labelled, f'llama3_multilabel_{current_time}')
    # df = pd.read_csv(file_path)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, cfg.data.multilabel))
    # df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if x else []) #! Hopefully not needed
    print("Starting!")
    num_items = df.shape[0]
    print('Num items: ',num_items)

    with open(os.path.join(PROJECT_ROOT, cfg.llm.promts, 'llama3/multilabel', prompt_filename), 'r') as f:
        prompt = f.read()

    with open('methods/telecoms-topics-classification/prompts/system-prompt-max-recall.txt', 'r') as f:
        system_prompt = f.read()

    with open('methods/telecoms-topics-classification/prompts/user-prompt.txt') as f:
        user_prompt = f.read()    

    tel_topic_match = ["teleology","telecommunications","radio frequency","radar","mobile phones","bluetooth","WiFi","data networks","optical networks","microwave technology","radio technology","mobile radio","4G","LiFi","mobile network","radio and television","satellite radio","telecommunications networks","5G","fiber-optic network","cognitive radio","fixed wireless network",]
    tel_topic_match = [f"'{label}'" for label in tel_topic_match]
    total_topics_num = len(tel_topic_match)

    print(f'Total possible topics: {total_topics_num}')

    # Break up the topics list into chunks, starting at a single topic and working all the way up to all the topics
    for topics_num in range(1, total_topics_num+1):
        num_topics_chunks = total_topics_num // topics_num
        remainder = total_topics_num % topics_num
        topics_chunks = [tel_topic_match[i*topics_num:(i+1)*topics_num] for i in range(num_topics_chunks)]
        if remainder:
            topics_chunks.append(tel_topic_match[num_topics_chunks*topics_num:])
    
        # Save a separate json file for each round of predictions
        checkpoint_path = base_checkpoint_path + f'cordis-telecoms-chunk_size-{topics_num}.json'
        batch_data = []
        
        for idx, row in df.iterrows():
            name = row['name']
            description = row['description']
            gt = row['topics']
            full_scores = {}
            # Generate confidence scores for each chunk using LLM
            
            for topics_chunk in topics_chunks:
                
                dialog = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(title=name, abstract=description, labels_list=", ".join(topics_chunk))}
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
                            # Add list of assigned scores to full_scores dictionary without deleting existing scores
                            full_scores.update(scores)
                            
                            print(f'Processed {len(scores)}: {scores}')
                        else:
                            # Flag that the output was incorrect
                            print(f"Format error: {scores}")
                            scores = {"format_error": 0.0}
                    except ValueError:
                        print(f"Value error: {scores}")

                except Exception as e:
                    print(f"Results generation error: {e}")
                    print(generation_content)
                    scores = {"gen_error": 0.0}

            batch_data.append([description, name, gt, full_scores])
        save_checkpoint(batch_data, checkpoint_path)
        batch_data = []  # Reset the batch data list

    batch_data = []

    print("Data labeling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
