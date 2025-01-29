#!/bin/bash
#SBATCH --job-name=fsdp
#SBATCH --nodes=1       
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=PDS0352

# export WANDB_MODE=offline
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

unset GCC_HOME
unset OSC_GCC_DIR
unset CXX
unset OSC_CC
unset CC
unset MPICC
unset MPICXX

cd $HOME/workspace/speed_llama3
nvidia-smi

singularity exec --nv --bind $HOME:$HOME --bind /tmp:/tmp $HOME/sif/llm_pt25_0.3.sif torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    train_fsdp.py \
    --model_name meta-llama/Llama-3.2-1B \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --sequence_length 2048 \
    --dtype bfloat16 \
    --train_data_cache_dir ./data/findweb-edu-1000000-1000/train \
    --val_data_cache_dir ./data/findweb-edu-1000000-1000/validation \
    --micro_batch_size 16 \
    --attn_impl flash_attention_3

