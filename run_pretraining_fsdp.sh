#!/bin/sh
set -e -x
nvidia-smi
pip list

# export WANDB_MODE=offline
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

unset GCC_HOME
unset OSC_GCC_DIR
unset CXX
unset OSC_CC
unset CC
unset MPICC
unset MPICXX

torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    train_fsdp.py \
    --model_name meta-llama/Llama-3.2-1B \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --sequence_length 2048 \
    --acc_steps 4 \
    --micro_batch_size 16 \
    --val_batch_size 16 \
    --val_steps 100 \
    --warmup_steps 200 \
    --dtype bfloat16 \
    --train_data_cache_dir ./data/findweb-edu-1000000-1000/train \
    --val_data_cache_dir ./data/findweb-edu-1000000-1000/validation \
    --output_dir ./outputs \
    --attn_impl sdpa \
    --wandb_project speed-llama-1b-debug
