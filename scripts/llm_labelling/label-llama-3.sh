#!/bin/bash

torchrun --nproc_per_node 1 label-llama.py \
    --ckpt_dir ../../models/llama3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path ../../models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 2
