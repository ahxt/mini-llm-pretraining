#!/bin/sh
set -e -x
nvidia-smi
pip list

echo "System Information:"
echo "Hostname: $(hostname)"
echo "Operating System: $(uname -a)"
echo "CPU Info: $(lscpu | grep 'Model name')"
echo "Memory Info: $(free -h)"
echo "GPU Info: $(nvidia-smi -L)"
echo "Python Version: $(python3 --version)"
echo "Pip Version: $(pip --version)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')"


export WANDB_MODE=offline
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
    --dtype bfloat16 \
    --train_data_cache_dir ./data/findweb-edu-1000000-1000/train \
    --val_data_cache_dir ./data/findweb-edu-1000000-1000/validation \
    --micro_batch_size 16 \
    --attn_impl sdpa
