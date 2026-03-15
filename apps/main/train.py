# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from copy import deepcopy
import gc
import logging
import math
import os
import sys
import time
import json
import hashlib
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    get_global_rank,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    tp_parallelize,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

logger = logging.getLogger()


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    pure_adam_update: bool = False
    update_norms_freq: int = 10  # Log weight update norms every N steps
    log_adamw: bool = False  # Log AdamW update stats
    histogram_freq: int = 10  # Log AdamW update histograms every N steps

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None
    
    # Validation evaluation during training
    validation: Optional[Dict] = None  # Just needs max_steps
    validation_freq: int = 1000  # Run validation every N steps
    
    # Data injection
    injection_steps: List[int] = field(default_factory=list)  # Steps to inject data
    injection_file: str = "injection_2025.jsonl"  # File containing injection data


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

    if args.logging.tensorboard is not None:
        args.logging.tensorboard.comment = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def evaluate_injection_loss(model, tokenizer, args):
    """Evaluate model on injection data and return loss and factual token accuracy."""
    import re
    source_dir = list(args.data.sources.keys())[0]
    injection_path = Path(args.data.root_dir) / source_dir / args.injection_file
    
    if not injection_path.exists():
        logger.warning(f"Injection file not found: {injection_path}")
        return {"loss": 0.0, "factual_loss": 0.0, "factual_accuracy": 0.0}
    
    with open(injection_path, 'r') as f:
        injection_data = [json.loads(line) for line in f]
    
    if not injection_data:
        return {"loss": 0.0, "factual_loss": 0.0, "factual_accuracy": 0.0}
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    factual_loss = 0
    factual_correct = 0
    factual_total = 0
    
    with torch.no_grad():
        for item in injection_data:
            text = item.get('text', '')
            if not text:
                continue
                
            tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
            max_tokens = 2 * args.data.seq_len
            tokens = tokens[:max_tokens + 1]
            
            token_texts = [tokenizer.decode([t]) for t in tokens]
            # State acronyms to exclude from factual tokens
            state_acronyms = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC', 'He','She', 'It'}
            is_factual = []
            for text in token_texts:
                text_clean = text.strip()
                # Check if it's a state acronym
                if text_clean in state_acronyms:
                    is_factual.append(False)
                else:
                    # Check for uppercase letters or digits (proper nouns/numbers)
                    is_factual.append(bool(re.search(r'[A-Z]|\d', text)))
            
            for i in range(0, len(tokens) - 1, args.data.seq_len):
                chunk = tokens[i:i+args.data.seq_len+1]
                chunk_factual = is_factual[i:i+args.data.seq_len+1]
                if len(chunk) < 2:
                    continue
                    
                input_ids = torch.tensor(chunk[:-1]).unsqueeze(0).cuda()
                labels = torch.tensor(chunk[1:]).unsqueeze(0).cuda()
                
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1),
                    reduction='none'
                )
                
                total_loss += loss.sum().item()
                total_tokens += labels.numel()
                
                factual_mask = torch.tensor(chunk_factual[1:], dtype=torch.bool).cuda()
                if factual_mask.any():
                    factual_loss += loss[factual_mask].sum().item()
                    factual_correct += (logits.argmax(dim=-1).view(-1)[factual_mask] == labels.view(-1)[factual_mask]).sum().item()
                    factual_total += factual_mask.sum().item()
            break
    
    model.train()
    return {
        "loss": total_loss / total_tokens if total_tokens > 0 else 0.0,
        "factual_loss": factual_loss / factual_total if factual_total > 0 else 0.0,
        "factual_accuracy": factual_correct / factual_total if factual_total > 0 else 0.0
    }


def log_adamw_update_histograms(model, optimizer, tb_writer, step: int, n_layers: int, lr: float, eps: float = 1e-8):
    """Log histograms of AdamW parameter updates to Tensorboard.
    
    Computes pure AdamW updates as: -lr * exp_avg / (sqrt(exp_avg_sq) + eps)
    This excludes weight decay effects.
    """
    if tb_writer is None:
        return
    
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if id(param) not in param_to_name:
                continue
            
            name = param_to_name[id(param)]
            
            layer_idx = None
            for i in range(0, n_layers):
                if f"layers.{i}." in name:
                    layer_idx = i
                    break
            
            if layer_idx is None:
                continue
            
            state = optimizer.state.get(param, None)
            if state is None or 'exp_avg' not in state:
                continue
            
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            # Convert DTensor to local if needed
            if hasattr(exp_avg, 'to_local'):
                exp_avg = exp_avg.to_local()
            if hasattr(exp_avg_sq, 'to_local'):
                exp_avg_sq = exp_avg_sq.to_local()
            
            # Compute AdamW update: exp_avg / (sqrt(exp_avg_sq) + eps)
            update = exp_avg / (torch.sqrt(exp_avg_sq) + eps)
            param_name = name.split(f"layers.{layer_idx}.")[-1]

            logger.info('AdamW Step {}: Layer {}, Param {}, update shape: {}'.format(step, layer_idx, param_name, update.shape))
            tb_writer.add_histogram(f"adamw_updates/layer_{layer_idx}/{param_name}", update, step)
            tb_writer.add_scalar(f"adamw_updates_mean/layer_{layer_idx}/{param_name}", update.mean().item(), step)
            tb_writer.add_scalar(f"adamw_updates_std/layer_{layer_idx}/{param_name}", update.std().item(), step)


def compute_update_norms(model, n_layers: int, weight_decay: float, lr: float) -> Dict[str, float]:
    """Compute norm(W_t - W_{t-1}) / norm(W_t) for every 3rd layer using probe's _get_stats.
    
    Requires that model parameters have a '_prev_weight' attribute set before optimizer step.
    """
    from lingua.probe import _get_stats
    
    update_norms = {}
    for name, param in model.named_parameters():
        if hasattr(param, '_prev_weight'):
            # Only log every 3rd layer
            layer_idx = None
            for i in range(0, n_layers, 3):
                if f"layers.{i}." in name:
                    layer_idx = i
                    break
            
            if layer_idx is not None:
                # Convert to regular tensors to avoid DTensor sharding issues
                param_data = param.data.to_local() if hasattr(param.data, 'to_local') else param.data
                prev_weight = param._prev_weight.to_local() if hasattr(param._prev_weight, 'to_local') else param._prev_weight
                
                update = param_data - prev_weight

                # get the pure adam update
                # update_no_wd = update + lr*weight_decay*prev_weight

                update_stats = _get_stats(update)
                # update_no_wd_stats = _get_stats(update_no_wd)
                param_stats = _get_stats(param_data)
                
                if len(tuple(param_data.shape)) == 2:
                    # Compute wrt specrtal RMS norm
                    logger.info("**JOE: Computing update spec RMS**")
                    fan_out_upd, fan_in_upd = update.shape 
                    spec_rms_delta = update.new_tensor(math.sqrt(fan_in_upd / fan_out_upd)) * torch.linalg.matrix_norm(update.float(), ord=2).to(update.dtype)
                    logger.info("**JOE: DONE computing update spec RMS**")
                    
                    logger.info("**JOE: Computing spec RMS of param**")
                    fan_out_param, fan_in_param = param_data.shape 
                    spec_rms_param = param_data.new_tensor(math.sqrt(fan_in_param / fan_out_param)) * torch.linalg.matrix_norm(param_data.float(), ord=2).to(param_data.dtype)
                    logger.info("**JOE: DONE computing spec RMS of param**")

                    rel_upd_spec_rms = spec_rms_delta/spec_rms_param


                if param_stats.get('rms_norm', 0) > 0:
                    param_name = name.split(f"layers.{layer_idx}.")[-1]
                    update_norm_val = update_stats['rms_norm'] / param_stats['rms_norm']
                    # update_no_wd_norm_val = update_no_wd_stats['rms_norm'] / param_stats['rms_norm']
                    update_norms[f"layer_{layer_idx}_upd/{param_name}"] = (
                        update_norm_val.item() if hasattr(update_norm_val, 'item') else float(update_norm_val)
                    )

                    update_norms[f"layer_{layer_idx}_upd_spec_rms/{param_name}"] = (
                        rel_upd_spec_rms.item() if hasattr(rel_upd_spec_rms, 'item') else float(rel_upd_spec_rms)
                    )

                    # update_norms[f"updates_no_wd/layer_{layer_idx}/{param_name}"] = (
                    #     update_no_wd_norm_val.item() if hasattr(update_no_wd_norm_val, 'item') else float(update_no_wd_norm_val)
                    # )

                
                # Clean up to free memory
                del param._prev_weight
    return update_norms


def inject_data_batch(args: TrainArgs, original_batch: torch.Tensor, step: int) -> torch.Tensor:
    """Inject specific data from injection file into the batch at predefined steps."""
    # Use the first data source directory
    source_dir = list(args.data.sources.keys())[0]
    injection_path = Path(args.data.root_dir) / source_dir / args.injection_file
    
    if not injection_path.exists():
        logger.warning(f"Injection file not found: {injection_path}")
        return original_batch
    
    with open(injection_path, 'r') as f:
        injection_data = [json.loads(line) for line in f]
    
    if not injection_data:
        return original_batch
    
    tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
    injection_item = injection_data[step % len(injection_data)]
    injection_text = injection_item.get('text', '')
    
    if injection_text:
        tokens = tokenizer.encode(injection_text, add_bos=False, add_eos=False)
        seq_len = original_batch.shape[1]
        batch_size = original_batch.shape[0]
        
        # Fill batch items with available tokens
        if len(tokens) >= 2:  # Need at least 2 tokens for input/label pair
            num_items = 0
            for i in range(batch_size):
                start_idx = i * seq_len
                if start_idx >= len(tokens) - 1:  # Not enough tokens for this batch item
                    break
                    
                end_idx = min(start_idx + seq_len, len(tokens) - 1)
                input_ids = torch.tensor(tokens[start_idx:end_idx], dtype=torch.long)
                labels = torch.tensor(tokens[start_idx+1:end_idx+1], dtype=torch.long)
                
                original_batch[i, :len(input_ids), 0] = input_ids
                original_batch[i, :len(labels), 1] = labels
                num_items += 1
        else:
            num_items = 0
        
        logger.info(f"Injected {num_items} batch items ({len(tokens)} tokens) at step {step}")
    
    return original_batch


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path and not args.checkpoint.continue_training_from_init:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        # Note: if continue_training_from_init=True, checkpoint.load() will overwrite these weights
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        log_every = 100
        
        # Run initial validation evaluation at step 0
        if args.validation is not None:
            logger.info("Running initial validation evaluation...")
            model.eval()
            
            val_data_args = deepcopy(args.data)
            val_data_loader_state = init_dataloader_state_from_args(
                val_data_args, dp_rank, dp_degree, file_pattern="*.val.jsonl"
            )
            
            val_losses = []
            val_steps = 0
            max_val_steps = args.validation.get("max_steps", 100) if args.validation else 100
            
            with build_dataloader_from_args(val_data_args, state=val_data_loader_state) as val_data_loader:
                with torch.no_grad():
                    for val_batch, _ in val_data_loader:
                        if val_steps >= max_val_steps:
                            break
                            
                        val_batch = torch.tensor(val_batch, dtype=torch.long)
                        val_input_ids = val_batch[:, :, 0].cuda()
                        val_labels = val_batch[:, :, 1].cuda()
                        
                        val_loss = model(val_input_ids, val_labels)
                        val_losses.append(val_loss.item())
                        val_steps += 1
            
            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_metrics = dist_mean_dict({"validation/loss": avg_val_loss})
                
                if get_is_master():
                    logger.info(f"Initial validation loss: {val_metrics['validation/loss']:.4f}")
                    metric_logger.log({"global_step": 0, **val_metrics})
            
            model.train()
            logger.info("Initial validation evaluation completed")
        
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )
            
            # Inject specific data at predefined steps (only on rank 0)
            if train_state.step+1 in args.injection_steps and get_global_rank() == 0:
                batch = inject_data_batch(args, batch, train_state.step)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            # if train_state.step % log_every == 0:
            #     # Hash the tokens for this rank and batch
            #     token_hash = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
                
            #     # Save hash to file
            #     hash_data = {
            #         "step": train_state.step,
            #         "rank": get_global_rank(),
            #         "token_hash": token_hash
            #     }

            #     os.makedirs(Path(args.dump_dir)/"token_hashes", exist_ok=True)
                
            #     with open(Path(args.dump_dir) /"token_hashes"/f"token_hashes_rank_{get_global_rank()}.jsonl", "a") as f:
            #         f.write(json.dumps(hash_data) + "\n")
                
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()
                    
                    # Extract probe statistics INSIDE the context before store is cleared
                    probe_metrics = {}
                    for key, stats in probe.store.items():
                        if isinstance(stats, dict) and 'rms_norm' in stats:
                            # Extract parameter RMS norms for weights only (not gradients)
                            if '::w' in key and '::w.g' not in key:
                                # Log final output layer
                                if 'output' in key:
                                    probe_metrics['output_layer/weight_rms'] = stats['rms_norm'].item()
                                # Only log every 3rd layer
                                for i in range(0, args.model.n_layers, 3):
                                    if f"layers.{i}." in key:
                                        # Extract parameter name (attention.wq, feed_forward.w2, etc.)
                                        param_name = key.split(f"layers.{i}.")[-1].split("::")[0]
                                        clean_key = f"layer_{i}/{param_name}_w"
                                        probe_metrics[clean_key] = stats['rms_norm'].item()

                                        # clean_key_spec_rms = f"layer_{i}_spec_rms/{param_name}"
                                        # probe_metrics[clean_key_spec_rms] = stats['spec_rms'].item()
                                        break
                    
                    # Store probe metrics for later use in logging
                    setattr(train_state, 'probe_metrics', probe_metrics)

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            loss = model(input_ids, labels)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                # Snapshot weights before optimizer step for update norm computation
                for name, param in model.named_parameters():
                    for i in range(0, args.model.n_layers, 3):
                        if f"layers.{i}." in name:
                            param._prev_weight = param.data.clone()
                            break
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1
                
                # Log AdamW update histograms (after step increment)
                if args.log_adamw and train_state.step % args.histogram_freq == 0 and get_is_master():
                    if metric_logger.tb_writer is not None:
                        log_adamw_update_histograms(
                            model, optimizer, metric_logger.tb_writer, 
                            train_state.step, args.model.n_layers, curr_lr, args.optim.epsilon
                        )
                        metric_logger.tb_writer.flush()
            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()
            
            val_metrics = None
            # Run validation evaluation
            if args.validation is not None and every_n_steps(train_state, args.validation_freq, acc_step=0):
                logger.info("Running validation evaluation...")
                model.eval()
                
                # Use same dataloader setup as training but for validation files
                val_data_args = deepcopy(args.data)
                val_data_loader_state = init_dataloader_state_from_args(
                    val_data_args, dp_rank, dp_degree, file_pattern="*.val.jsonl"
                )
                
                val_losses = []
                val_steps = 0
                max_val_steps = args.validation.get("max_steps", 100) if args.validation else 100
                
                with build_dataloader_from_args(val_data_args, state=val_data_loader_state) as val_data_loader:
                    with torch.no_grad():
                        
                        for val_batch, _ in val_data_loader:
                            if val_steps >= max_val_steps:
                                break
                                
                            val_batch = torch.tensor(val_batch, dtype=torch.long)
                            val_input_ids = val_batch[:, :, 0].cuda()
                            val_labels = val_batch[:, :, 1].cuda()
                            
                            val_loss = model(val_input_ids, val_labels)
                            val_losses.append(val_loss.item())
                            val_steps += 1
                
                # Average validation loss across all steps and GPUs
                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_metrics = dist_mean_dict({"validation/loss": avg_val_loss})
                    
                    if get_is_master():
                        logger.info(f"Validation loss: {val_metrics['validation/loss']:.4f}")
                
                model.train()
                logger.info("Validation evaluation completed")

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                # Add probe metrics if available
                if hasattr(train_state, 'probe_metrics') and train_state.probe_metrics:
                    logger.info('Adding probe metrics to metrics')
                    metrics.update(train_state.probe_metrics)

                # Compute and log weight update norms
                if every_n_steps(train_state, args.update_norms_freq, acc_step=0):
                    update_norms = compute_update_norms(model, args.model.n_layers, args.optim.weight_decay, curr_lr)
                    metrics.update(update_norms)

                if val_metrics:
                    metrics.update(val_metrics)

                # Evaluate on injection data even if injection steps are not configured
                injection_loss = evaluate_injection_loss(model, tokenizer, args)
                injection_metrics = dist_mean_dict({
                    "injection/loss": injection_loss['loss'],
                    "injection/facts_acc": injection_loss['factual_accuracy'],
                    "injection/facts_loss": injection_loss['factual_loss']
                })
                if get_is_master():
                    metric_logger.log({
                        "global_step": train_state.step,
                        **injection_metrics
                    })
                    logger.info(f"Injection loss at step {train_state.step}: {injection_metrics['injection/loss']:.4f}")

                if get_is_master():
                    metric_logger.log(metrics)
                
                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )
                
            if args.eval is not None and (every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ) or every_n_steps(train_state, args.steps, acc_step=0)):
                from apps.main.eval import (
                    launch_eval,
                    EVAL_FOLDER_NAME,
                    EvalArgs,
                )

                eval_args = dataclass_from_dict(EvalArgs, args.eval)

                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_eval(eval_args)
                elif get_is_master():
                    if metric_logger.tb_writer is not None and args.logging.tensorboard is not None:
                        eval_args.tensorboard = deepcopy(args.logging.tensorboard)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.eval",
                                copy_code=False,
                                nodes=args.async_eval_gpus // 8,
                                qos="lowest",
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

        if not saved:
            checkpoint.save(
                model,
                optimizer,
                train_state,
                args,
                device_mesh=world_mesh,
            )
        
        # Eval injection on final step
        injection_loss = evaluate_injection_loss(model, tokenizer, args)
        injection_metrics = dist_mean_dict({
            "injection/loss": injection_loss['loss'],
            "injection/facts_acc": injection_loss['factual_accuracy'],
            "injection/facts_loss": injection_loss['factual_loss']
        })
        if get_is_master():
            metric_logger.log({
                "global_step": train_state.step,
                **injection_metrics
            })
            if metric_logger.tb_writer is not None:
                metric_logger.tb_writer.flush()
            logger.info(f"Final injection loss at step {train_state.step}: {injection_metrics['injection/loss']:.4f}")
    
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
