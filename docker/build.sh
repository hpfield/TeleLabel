#!/bin/bash

docker run -d \
    --name telelabel \
    -p 2223:23 \
    --gpus all \
    --mount type=bind,source=/home/harry/TeleLabel,target=/home/ubuntu/TeleLabel \
    --mount type=bind,source=/home/harry/mamba,target=/home/ubuntu/mamba \
    --mount type=bind,source=/data/llama,target=/home/ubuntu/.llama \
    harry-telelabel