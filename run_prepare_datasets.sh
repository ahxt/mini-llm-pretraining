#!/bin/sh
set -e -x
cat ./modeling_llama.py
cat ./train_fsdp.py
cat ./run_pretraining_fsdp.sh
pip list
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

python3 preprocess_findweb.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --name sample-10BT \
    --split train \
    --sequence_length 2048 \
    --val_spilt_num 1000 \
    --train_spilt_num 1000000 \
    --num_proc 16 \
    --output ./data/findweb-edu-1000000-1000