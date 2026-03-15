#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import json
import os
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from timeit import default_timer as timer

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.tokenizer import build_tokenizer
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import DataArgs, build_dataloader_from_args, init_dataloader_state_from_args, init_dataloader_state_from_args
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    parallelize_model, 
    get_global_rank,
    get_is_master,
    get_world_size,
    setup_torch_distributed, 
    get_device_mesh
)
from lingua.logger import init_logger
from .transformer import LMTransformerArgs, LMTransformer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.activation_hooks import ActivationHook



@dataclass
class ValidateArgs:
    dump_dir: str = ""
    checkpoint_path: str = ""
    model: LMTransformerArgs = None
    data: DataArgs = None
    log_every: int = 100
    steps: int = 95367  # Number of steps to replay
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    
    def __post_init__(self):
        if self.model is None:
            self.model = LMTransformerArgs()
        if self.data is None:
            self.data = DataArgs()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    
    default_cfg = OmegaConf.structured(ValidateArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    args = OmegaConf.to_object(cfg)
    
    # Setup distributed - use the distributed args from config
    setup_torch_distributed(args.distributed)
    
    # Auto-calculate dp_replicate if needed (same logic as train.py)
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
    
    device_mesh = get_device_mesh(args.distributed)
    
    # Setup
    if get_is_master():
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_config(args, Path(args.dump_dir) / "config.yaml")
    init_logger(Path(args.dump_dir) / "train.log")
    logger = logging.getLogger()
    
    # Load model
    model = LMTransformer(args.model)
    model = parallelize_model(model, device_mesh, args.model, args.distributed)
    
    # Load checkpoint
    load_from_checkpoint(args.checkpoint_path, model)
    model.eval()
    
    # # Setup activation hooks
    # hook = ActivationHook()
    # hook.register_hooks(model)
    
    # TensorBoard writer
    writer = None
    if get_global_rank() == 0:
        writer = SummaryWriter(Path(args.dump_dir) / "tensorboard")
    model.eval()
    
    # Setup data - get dp_rank and dp_degree from device_mesh like in train.py
    dp_rank = device_mesh.get_local_rank(mesh_dim=0) if device_mesh.ndim > 0 else 0
    dp_degree = device_mesh.size(mesh_dim=0) if device_mesh.ndim > 0 else 1
    data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)

    # For decoding the token ids corresponding to high loss tokens
    # Build tokenizer using the same config as the dataloader
    tokenizer = build_tokenizer(name=args.data.tokenizer.name, path=args.data.tokenizer.path)
    
    with build_dataloader_from_args(args.data, state=data_loader_state) as dataloader:
        # Validation loop
        with torch.no_grad():
            # Add this right before the for loop
            logger.info(f"Starting replay with {args.steps} target steps")
            logger.info(f"Batch size: {args.data.batch_size}, Seq len: {args.data.seq_len}")
            logger.info(f"World size: {get_world_size()}")
            for step, (batch_data, _) in enumerate(dataloader):
                if step >= args.steps:
                    break
                    
                if step % args.log_every != 0:
                    continue
                
                # Start timing
                iter_start = timer()
                
                # Convert to tensor like in train.py
                batch = torch.tensor(batch_data, dtype=torch.long)
                    
                # Extract input_ids and labels like in train.py
                input_ids = batch[:, :, 0].cuda()
                labels = batch[:, :, 1].cuda()
                
                # Hash the tokens for this rank and batch
                token_hash = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
                
                # Save hash to file
                hash_data = {
                    "step": step,
                    "rank": get_global_rank(),
                    "token_hash": token_hash
                }

                os.makedirs(Path(args.dump_dir)/"token_hashes", exist_ok=True)
                
                with open(Path(args.dump_dir) /"token_hashes"/f"token_hashes_rank_{get_global_rank()}.jsonl", "a") as f:
                    f.write(json.dumps(hash_data) + "\n")
                
                # Get logits and batch loss in single forward pass
                logits = model(input_ids)
                
                # Per-token loss
                token_losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    reduction='none'
                )
                
                # Batch loss from same logits
                batch_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # End timing
                curr_iter_time = timer() - iter_start
                
                # Log to TensorBoard
                if get_global_rank() == 0:
                    # Batch loss
                    writer.add_scalar("replay/batch_loss", batch_loss.item(), step)
                    
                    # Timing
                    writer.add_scalar("speed/curr_iter_time", curr_iter_time, step)
                    
                    # Per-token loss percentiles
                    percentiles = torch.quantile(token_losses.float(), torch.tensor([0.5, 0.7, 0.9, 0.95, 0.98, 1.0]).cuda())
                    writer.add_scalar("replay/loss_p50", percentiles[0].item(), step)
                    writer.add_scalar("replay/loss_p70", percentiles[1].item(), step)
                    writer.add_scalar("replay/loss_p90", percentiles[2].item(), step)
                    writer.add_scalar("replay/loss_p95", percentiles[3].item(), step)
                    writer.add_scalar("replay/loss_p98", percentiles[4].item(), step)
                    writer.add_scalar("replay/loss_p100", percentiles[5].item(), step)

                    max_loss = percentiles[3].item()
                    tok_ids = input_ids.view(-1).cpu().tolist()
                    token_id_max_loss = tok_ids[torch.argmax(token_losses).item()]
                    word = tokenizer.decode([token_id_max_loss])

                    # Get context tokens (10 before and 10 after)
                    max_loss_idx = torch.argmax(token_losses).item()
                    context_start = max(0, max_loss_idx - 10)
                    context_end = min(len(tok_ids), max_loss_idx + 11)
                    context_tokens = tok_ids[context_start:context_end]
                    context_words = [tokenizer.decode([tid]) for tid in context_tokens]
                    
                    logger.info("\n*** Batch {}: Max loss: {}, Token: {}.".format(step, max_loss, word))
                    # Highlight the token of interest in context
                    context_with_highlight = []
                    for i, cword in enumerate(context_words):
                        if context_start + i == max_loss_idx:
                            context_with_highlight.append(f">>>{cword}<<<")
                        else:
                            context_with_highlight.append(cword)
                    logger.info("*** Context: {}".format(' '.join(context_with_highlight)))
                    # Get logits for just the high-loss token
                    token_logits = logits.view(-1, logits.size(-1))[max_loss_idx]
                    token_probs = F.softmax(token_logits, dim=-1)
                    
                    # Calculate entropy of the probability distribution
                    entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10))
                    
                    top_indices = torch.topk(token_probs, 10).indices
                    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
                    logger.info("*** Top predicted tokens: {}".format(top_tokens))
                    logger.info("*** Entropy: {:.4f}\n".format(entropy.item()))
                    
                    
                    # Histogram of per-token losses
                    writer.add_histogram("replay/token_loss_distribution", token_losses.cpu(), step)
                    
                    # Save per-token losses and token IDs
                    token_data = {
                        "batch_idx": step,
                        "token_ids": input_ids.view(-1).cpu().tolist(),
                        "token_losses": token_losses.cpu().tolist(),
                        "max_loss": max_loss,
                        "max_loss_idx": max_loss_idx,
                        "max_loss_token": token_id_max_loss,
                        "max_loss_word": word,
                        "entropy": entropy.item(),
                        "top_tokens": top_tokens,
                        "max_context": context_with_highlight
                    }
                    
                    with open(Path(args.dump_dir) / "per_token_losses.jsonl", "a") as f:
                        f.write(json.dumps(token_data) + "\n")
                    
                    logger.info(f"Step {step}: batch_loss={batch_loss.item():.4f}")
                    logger.info(f"Step {step}")
            
            logger.info('*** JOE: for loop terminated at step: {}'.format(step))
    
    # # Save activations to json
    # all_activations = hook.activations
    # torch.save(all_activations, Path(args.dump_dir) / "activations.pt")
    
    # # Compute and save RMS norms
    # rms_norms = hook.compute_rms_norms()
    # if get_global_rank() == 0:
    #     # Log to TensorBoard
    #     for name, value in rms_norms.items():
    #         writer.add_scalar(f"rms_norms/{name}", value, step)
        
    #     with open(Path(args.dump_dir) / "activation_rms_norms.json", "w") as f:
    #         json.dump(rms_norms, f, indent=2)
    #     logger.info("Saved activation RMS norms")
    
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
