#!/bin/bash

# cd scripts/llm_labelling

# torchrun --nproc_per_node 1 llama-3_multilabel_labelling.py \
#     --ckpt_dir ../../models/llama3/Meta-Llama-3-8B-Instruct/ \
#     --tokenizer_path ../../models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
#     --max_seq_len 1024 --max_batch_size 2

torchrun --nproc_per_node 1 llama-3_multilabel_labelling.py
