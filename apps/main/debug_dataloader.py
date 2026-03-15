#!/usr/bin/env python3

import torch
import json
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import OmegaConf

from lingua.data import DataArgs, build_dataloader_from_args, init_dataloader_state_from_args, PackTokensState
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_device_mesh,
    get_global_rank,
    setup_torch_distributed,
    get_world_size
)

def main():
    # Load your config
    cfg = OmegaConf.load("apps/main/configs/replay_train_wsd_new_tok.yaml")
    data_args = DataArgs(**cfg.data)
    distributed_args = DistributedArgs(**cfg.distributed)
    
    # Setup distributed - exact same as replay_training.py
    setup_torch_distributed(distributed_args)

    # Auto-calculate dp_replicate if needed (same logic as train.py)
    if (
        distributed_args.dp_replicate
        * distributed_args.dp_shard
        * distributed_args.tp_size
        != get_world_size()
    ):
        assert get_world_size() % distributed_args.dp_shard == 0
        distributed_args.dp_replicate = get_world_size() // distributed_args.dp_shard

        assert distributed_args.dp_replicate % distributed_args.tp_size == 0
        distributed_args.dp_replicate = (
            distributed_args.dp_replicate // distributed_args.tp_size
        )
    device_mesh = get_device_mesh(distributed_args)
    
    dp_rank = device_mesh.get_local_rank(mesh_dim=0) if device_mesh.ndim > 0 else 0
    dp_degree = device_mesh.size(mesh_dim=0) if device_mesh.ndim > 0 else 1
    
    # Create fresh dataloader state
    fresh_state = init_dataloader_state_from_args(data_args, dp_rank, dp_degree)
    
    # Load checkpoint state from JSON
    ckpt_path = "/checkpoints/checkpoints/jsheno/jsheno-llama-1b-wsd-new-tok/checkpoints/0000095000"
    with open(f"{ckpt_path}/train_state_00000.json", "r") as f:
        train_state_data = json.load(f)
    
    if get_global_rank() == 0:
        print(f"Checkpoint step: {train_state_data['step']}")
        print(f"Checkpoint dataloader seq_idx: {train_state_data['data_loader_state']['seq_idx']}")
    
    # Reconstruct checkpoint dataloader state
    ckpt_state = PackTokensState(**train_state_data['data_loader_state'])
    
    # Test if checkpoint dataloader can continue from step 95000
    if get_global_rank() == 0:
        print(f"\nTesting checkpoint dataloader from step 95000:")
    with build_dataloader_from_args(data_args, state=ckpt_state) as ckpt_loader:
        try:
            for i in range(367):  # Try to get remaining steps to 95367
                batch, state = next(ckpt_loader)
                if i % 50 == 0 and get_global_rank() == 0:
                    print(f"Step {95000 + i}: state = {state}")
            if get_global_rank() == 0:
                print("Successfully reached step 95367!")
        except StopIteration:
            if get_global_rank() == 0:
                print(f"Checkpoint dataloader stopped at step {95000 + i}")
    
    # Test fresh dataloader
    if get_global_rank() == 0:
        print(f"\nTesting fresh dataloader:")
    with build_dataloader_from_args(data_args, state=fresh_state) as fresh_loader:
        try:
            for i in range(95367):  # Try all steps
                batch, state = next(fresh_loader)
                if i % 10000 == 0 and get_global_rank() == 0:
                    print(f"Step {i}: state = {state}")
            if get_global_rank() == 0:
                print("Fresh dataloader completed all 95367 steps!")
        except StopIteration:
            if get_global_rank() == 0:
                print(f"Fresh dataloader stopped at step {i}")

if __name__ == "__main__":
    main()
