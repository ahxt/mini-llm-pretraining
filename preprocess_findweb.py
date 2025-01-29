# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
from itertools import chain
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    sequence_length: int
) -> Dict[str, List[List[int]]]:
    
    text = examples['text']
    input_ids = tokenizer(text)['input_ids']
    input_ids = list(chain(*input_ids))
    total_length = len(input_ids)-1
    total_length = (total_length // sequence_length) * sequence_length
    return {'input_ids': [input_ids[i:i+sequence_length+1] for i in range(0, total_length, sequence_length)]} # why +1? for input and for label in the same sequence


def preprocess(
    dataset_name: str,
    name: Optional[str] = None,
    split: str = 'train',
    output: str = 'data',
    tokenizer_name: str = 'meta-llama/Llama-3.2-1B',
    num_proc: int = 64,
    sequence_length: int = 2048,
    val_spilt_num: int = 1000,
    train_spilt_num: int = 0
) -> None:

    logging.info(f'Initializing tokenizer of {tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    logging.info(f'Tokenizer initialized: {tokenizer}')


    ####### training dataset
    logging.info(f'Loading dataset: {dataset_name}')
    ori_dataset = load_dataset(dataset_name, name=name, split=split)

    tokenized_path = f'{output}/train'
    if train_spilt_num > 0:
        train_dataset = ori_dataset.select(range(0, train_spilt_num))
    else:
        train_dataset = ori_dataset.select(range(0, len(ori_dataset) - val_spilt_num))

    remove_columns = list(next(iter(train_dataset)).keys())
    
    logging.info('Tokenizing and processing train dataset')
    train_dataset = train_dataset.map(
        lambda examples: tokenize(examples, tokenizer, sequence_length),
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Tokenizing training data"
    )

    logging.info(f'Saving processed training dataset to {tokenized_path}')
    train_dataset.save_to_disk(tokenized_path, num_proc=num_proc)

    ####### Validation dataset
    tokenized_path = f'{output}/validation'
    logging.info(f'Preparing validation dataset from {dataset_name}')
    
    # Select validation samples from the end of the dataset
    val_dataset = ori_dataset.select(range(len(ori_dataset) - val_spilt_num, len(ori_dataset)))
    logging.info(f'Validation dataset size: {len(val_dataset)}')

    # Get column names to remove after tokenization
    remove_columns = list(next(iter(val_dataset)).keys())
    
    # Tokenize and process validation data
    logging.info('Tokenizing and processing validation dataset')
    val_dataset = val_dataset.map(
        lambda examples: tokenize(examples, tokenizer, sequence_length),
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Tokenizing validation data"
    )

    # Save processed validation dataset
    logging.info(f'Saving processed validation dataset to {tokenized_path}')
    val_dataset.save_to_disk(tokenized_path, num_proc=num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu", help="Path or name of the dataset")
    parser.add_argument("--name", default="sample-10BT", help="Name of the dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--output", default="data/fineweb-edu_10BT_mistral", help="Output directory")
    parser.add_argument("--tokenizer_name", default="meta-llama/Llama-3.2-1B", help="Model name for tokenizer")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for parallel processing")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Context length for tokenization")
    parser.add_argument("--val_spilt_num", type=int, default=1000, help="Number of validation split")
    parser.add_argument("--train_spilt_num", type=int, default=0, help="Number of training split")
    args = parser.parse_args()
    print(args)

    preprocess(
        dataset_name=args.dataset_name,
        name=args.name,
        split=args.split,
        output=args.output,
        tokenizer_name=args.tokenizer_name,
        num_proc=args.num_proc,
        sequence_length=args.sequence_length,
        val_spilt_num=args.val_spilt_num,
        train_spilt_num=args.train_spilt_num
    )
