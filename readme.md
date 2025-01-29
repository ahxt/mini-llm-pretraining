# How to speed up LLaMA 3 training?

A clean and efficient implementation for training LLaMA 3 models, focused on simplicity and performance. This codebase provides a streamlined version of LLaMA 3 pretraining based on Hugging Face's implementation and uses FSDP for distributed training.



## Files

- `modeling_llama.py`: minimal LLaMA 3 model implementation
- `train_fsdp.py`: FSDP distributed training script
- `preprocess_findweb.py`: data preprocessing and cache





## How to run

1. Recommend to use Docker to run the code
```
docker push xiaotian99/llm_pt25:v0.3
```
or use the [sglang official image](https://hub.docker.com/r/lmsysorg/sglang/tags).



2. Prepare the dataset:
```bash
./run_prepare_datasets.sh
```

3. Start training:
```bash
./run_pretraining_fsdp.sh
```



## Key Features
- Minimal codebase with clear structure
- Optimized for modern GPU architectures
- Support for multiple attention implementations
- Clean and easy to use code
