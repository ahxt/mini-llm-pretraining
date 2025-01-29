import argparse
import functools
import math
import os
import time
import uuid
from datetime import datetime

import datasets
import torch
import transformers
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import DefaultDataCollator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from modeling_llama import LlamaForCausalLM


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def evaluation(model, eval_dataloader, wandb, local_rank, current_step):
    if global_rank == 0:
        print("Running evaluation......")

    torch.cuda.empty_cache()
    model.eval()
    val_losses = 0
    total_steps = len(eval_dataloader)
    for step, batch in enumerate(eval_dataloader):
        inputs = batch["input_ids"].to(model.device)
        x, y = inputs[::, :args.sequence_length], inputs[::, -args.sequence_length:]

        with torch.no_grad():
            loss = model(input_ids=x, labels=y)

        val_losses += loss.float()
        if global_rank == 0:
            print(f"{(step+1)/total_steps*100:.2f} % done", end="\r")

    val_losses = val_losses / total_steps
    val_loss = get_all_reduce_mean(val_losses.detach().clone()).item()
    if global_rank == 0:
        if wandb:
            wandb.log({"val_loss": val_loss,}, step=current_step)
        print(f"\nValidation Loss: {val_loss:.4f}")

    model.train()
    torch.cuda.empty_cache()
    return val_loss


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=True
    )


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def clip_model_gradients(model, max_grad_norm):
    return model.clip_grad_norm_(max_grad_norm).item()


def get_scheduler(global_rank, scheduler_type, optimizer, max_steps):
    warmup_ratio = 0.01
    warmup_steps = math.ceil(max_steps * warmup_ratio)
    warmup_steps = min(warmup_steps, 1000)
    warmup_steps = max(warmup_steps, 100)
    return transformers.get_scheduler(name=scheduler_type, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,)


def save_model(global_rank, model, tokenizer, outpath, current_step):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if global_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"/step_{current_step}"
        model.save_pretrained(outpath, state_dict=cpu_state)
        tokenizer.save_pretrained(outpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_3", help="flash_attention_2|sdpa|flex|eager|flash_attention_3")
    parser.add_argument("--train_data_cache_dir", type=str, default="data/fineweb-edu_10BT/train")
    parser.add_argument("--val_data_cache_dir", type=str, default="data/fineweb-edu_10BT/validation")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=12)
    parser.add_argument("--acc_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000000000)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=14159)

    args = parser.parse_args()


    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    print(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}")

    torch.set_float32_matmul_precision('high')
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", rank=global_rank, world_size=world_size)

    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    scheduler_type = "cosine"
    transformers.set_seed(args.seed)

    args.total_batch_size = args.micro_batch_size * world_size * args.acc_steps

    run_id = str(uuid.uuid4().hex)[:8]
    model_output_path = f"{args.output_dir}/{args.model_name}/{run_id}"
    args.model_output_path = model_output_path

    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    args.date_of_run = date_of_run

    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True
    gradient_clipping = 1.0

    if global_rank == 0:
        run = wandb.init(project="llama-1b", name=run_id, config=vars(args),)
        print(args)
        for key, value in vars(args).items():
            print(f"{key}{'-' * max(0, 80 - len(key) - len(str(value)))}{value}")
        

    # config = transformers.AutoConfig.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    config.qk_bias = True
    config.sequence_length = args.sequence_length
    config.attn_impl = args.attn_impl

    if global_rank == 0:
        print(config)

    # model = LlamaForCausalLM._from_config(config=config, torch_dtype=torch.bfloat16, token=os.environ["HF_TOKEN"])
    # tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=args.sequence_length, padding_side="right", use_fast=False, token=os.environ["HF_TOKEN"])
    model = LlamaForCausalLM._from_config(config=config, torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=args.sequence_length, padding_side="right", use_fast=False)
    
    tokenizer.pad_token_id = 128004  # pad token id for LLaMA 3.1 base
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.stop_tokens = [tokenizer.eos_token_id]
    model.resize_token_embeddings(len(tokenizer))

    num_params = sum([p.numel() for p in model.parameters()])
    if global_rank == 0:
        print(model)
        print(f"Number of parameters: {num_params}")
        print(f"Model data type: {model.dtype}")
        print(f"Model device: {model.device}")


    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls={LlamaDecoderLayer} 
        ) 

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # sharding_strategy=ShardingStrategy.NO_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, 
            reduce_dtype=torch.bfloat16, 
            buffer_dtype=torch.bfloat16,),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=None,
        use_orig_params=True
    )
    model = torch.compile(model)
    # model = torch.compile(model, mode="max-autotune")
    model = FSDP(model, **fsdp_config)
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True
    )

    dataset = datasets.load_from_disk(args.train_data_cache_dir)
    data_collator = DefaultDataCollator()
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=shuffle, seed=args.seed)
    train_loader = DataLoader(dataset, shuffle=False, pin_memory=True, drop_last=False, batch_size=args.micro_batch_size, collate_fn=data_collator, sampler=train_sampler, num_workers=8, prefetch_factor=8)

    val_dataset = datasets.load_from_disk(args.val_data_cache_dir)
    data_collator = DefaultDataCollator()
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, seed=args.seed)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, drop_last=False, batch_size=args.val_batch_size, collate_fn=data_collator, sampler=val_sampler)

    total_steps_per_epoch = len(train_loader)


    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    torch.distributed.barrier()
    current_acc_steps = 0  

    losses = 0

    validation_loss = evaluation(model, val_loader, None, global_rank, 0)

    args.max_steps = min(args.max_steps, len(train_loader)//args.acc_steps)

    scheduler = get_scheduler(global_rank, scheduler_type, optimizer, args.max_steps)

    step_start_time = time.time()
    training_start_time = time.time()


    for step, batch in enumerate(train_loader):
        current_step = (step+1) // args.acc_steps
        if current_step > args.max_steps:
            break
    
        inputs = batch["input_ids"].to(model.device)
        x, y = inputs[::, :args.sequence_length], inputs[::, -args.sequence_length:]

        if (step+1) % args.acc_steps != 0:
            with model.no_sync():
                loss = model(input_ids=x, labels=y)
                loss.backward()
                losses += loss.float()
        else:
            loss = model(input_ids=x, labels=y)
            loss.backward()
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            losses += loss.float()
            train_loss = get_all_reduce_mean(losses.detach().clone()).item()
            losses = 0

            if current_step % args.log_steps == 0:

                if global_rank == 0:
                    duration_time = time.time() - step_start_time
                    throughput = (args.sequence_length * args.total_batch_size*args.log_steps) / duration_time
                    seen_tokens = current_step * args.total_batch_size * args.sequence_length
                    last_lr = scheduler.get_last_lr()[0]
                    status_dict = {"current_loss": train_loss/args.acc_steps, "learning_rate": last_lr, "throughput": throughput}
                    wandb.log(status_dict, step=current_step)
                    remaining_time = (args.max_steps - current_step) * duration_time / args.log_steps
                    print(f"Step: {current_step}/{args.max_steps}, Loss:{train_loss/args.acc_steps:.4f}, LR: {last_lr:.6f}, Thruput: {throughput:.2f}, Seen Tokens: {seen_tokens}, Duration: {duration_time:.3f}s, Remaining: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}",)
                    losses = 0
                    step_start_time = time.time()

            if current_step % args.val_steps == 0:
                validation_loss = evaluation(model, val_loader, wandb, global_rank, current_step)
                if current_step % args.save_steps == 0:
                    save_model(global_rank, model, tokenizer, args.model_output_path, current_step)
            
    # save final model
    save_model(global_rank, model, tokenizer, args.model_output_path, "final")
    # prof.export_chrome_trace(f"./profiling/trace_break_{local_rank}_{time.time()}.json")
