#!/bin/sh
set -e -x
nvidia-smi
pip list

export WANDB_MODE=offline
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_fsdp.py \
    --model_name meta-llama/Llama-3.2-1B \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --sequence_length 2048 \
    --dtype bfloat16 \
    --train_data_cache_dir ./data/findweb-edu-1000000-1000/train \
    --val_data_cache_dir ./data/findweb-edu-1000000-1000/validation \
    --micro_batch_size 16 \
    --attn_impl sdpa
