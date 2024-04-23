#!/bin/bash

torchrun --nproc_per_node 1 label-llama.py \
    --ckpt_dir /home/rz20505/Documents/ask-jgi/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path /home/rz20505/Documents/ask-jgi/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 2
