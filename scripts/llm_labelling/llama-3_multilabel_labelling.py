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


PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))

def save_checkpoint(data, file_path, mode='a'):
    """ Helper function to save data to a CSV file """
    df = pd.DataFrame(data, columns=['text', 'gt', 'topics'])
    df.to_csv(file_path, mode=mode, header=mode=='w', index=False)

@hydra.main(config_path="../../cfg", config_name="config", version_base='1.3.2')
def main(
    cfg: DictConfig,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0, # for consistency
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

    # file_path = '/home/rz20505/Documents/ask-jgi/data/unlabelled-desc-name-topics.csv'
    # checkpoint_path = '/home/rz20505/Documents/ask-jgi/data/labelled/llama-3-zero-shot.csv'
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = os.path.join(PROJECT_ROOT, cfg.outputs.path, f"{cfg.llm.model}", f"{datetime}") #! Check the LLM dir exists first
    # df = pd.read_csv(file_path)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, cfg.data.multilabel))

    
    with open(os.path.join(PROJECT_ROOT, cfg.llm.promts, cfg.llm.model, 'multilabel', cfg.llm.system_prompt), 'r') as f:
        system_prompt = f.read()

    with open(os.path.join(PROJECT_ROOT, cfg.llm.promts, cfg.llm.model, 'multilabel', cfg.llm.user_prompt), 'r') as f:
        user_prompt = f.read()
    
    start_index = 0
    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_index = len(checkpoint_df)
        print(f"Resuming from index {start_index}")

    # with open('/home/rz20505/Documents/ask-jgi/methods/utils/prompts/system-prompt-llama-3.txt', 'r') as f:
    #     system_prompt = f.read()


    topics = cfg.labels.topics
    num_topics = len(topics)

    total_time = 0
    batch_data = []
    topic_pattern = re.compile(r'\["[^"]*"(?:, "[^"]*")*\]') 


    # Break up the topics list into chunks, starting at a single topic and working all the way up to all the topics
    for topics_num in range(1, num_topics+1):
        num_topics_chunks = num_topics // topics_num
        remainder = num_topics % topics_num
        topics_chunks = [topics[i*topics_num:(i+1)*topics_num] for i in range(num_topics_chunks)]
        if remainder:
            topics_chunks.append(topics[num_topics_chunks*topics_num:])
    
        # Save a separate json file for each round of predictions
        checkpoint_path = os.path.join(output_dir, f'{topics_num}_topics.json')
        batch_data = []
        
        for idx, row in df.iterrows():
            data = row['text']
            gt = row['topics']
            full_scores = {}
            # Generate confidence scores for each chunk using LLM
            
            for topics_chunk in topics_chunks:
                
                dialog = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(data=data, labels_list=", ".join(topics_chunk))}
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

            batch_data.append([data, gt, full_scores])
        save_checkpoint(batch_data, checkpoint_path)
        batch_data = []  # Reset the batch data list

    batch_data = []

    print("Data labeling complete and files saved")

if __name__ == "__main__":
    fire.Fire(main)
