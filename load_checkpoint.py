#!/usr/bin/env python3
"""
Standalone script to load Meta Lingua model checkpoints for inspection.
Usage: python load_checkpoint.py /path/to/dump_dir/checkpoints/0000001000
"""

import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

def load_model_checkpoint(ckpt_dir):
    """Load Meta Lingua checkpoint and return model state dict"""
    ckpt_path = Path(ckpt_dir)
    
    # Check if consolidated version exists and is valid
    consolidated_path = ckpt_path / "consolidated"
    consolidated_file = consolidated_path / "consolidated.pth"
    
    needs_consolidation = (
        not consolidated_file.exists() or 
        consolidated_file.stat().st_size == 0
    )
    
    if needs_consolidation:
        print(f"Consolidating checkpoint from {ckpt_path} (this may take several minutes...)")
        consolidated_path.mkdir(exist_ok=True)
        
        import time
        start_time = time.time()
        
        # Convert distributed checkpoint to regular PyTorch format
        dcp_to_torch_save(str(ckpt_path), str(consolidated_file))
        
        # Verify consolidation succeeded
        if not consolidated_file.exists() or consolidated_file.stat().st_size == 0:
            raise RuntimeError(f"Consolidation failed - {consolidated_file} is missing or empty")
        
        elapsed = time.time() - start_time
        size_mb = consolidated_file.stat().st_size / (1024 * 1024)
        print(f"Consolidation completed in {elapsed:.1f}s, file size: {size_mb:.1f}MB")
        
        # Copy config
        config_src = ckpt_path / "params.json"
        config_dst = consolidated_path / "params.json"
        if config_src.exists():
            config_dst.write_text(config_src.read_text())
    
    # Load the consolidated checkpoint
    print(f"Loading checkpoint from {consolidated_file}")
    checkpoint = torch.load(consolidated_file, map_location='cpu', weights_only=True)
    
    # Load config if available
    config_path = consolidated_path / "params.json"
    config = None
    if config_path.exists():
        config = OmegaConf.load(config_path)
    
    return checkpoint["model"], config

def inspect_weights(state_dict):
    """Inspect model weights - focus on every 3rd layer"""
    
    # Print embedding and output layer norms first
    print("\nEmbedding and Output Layer Norms:")
    
    # Embedding layer
    emb_key = "tok_embeddings.weight"
    if emb_key in state_dict:
        emb_norm = torch.norm(state_dict[emb_key]).item()
        print(f"Embedding layer norm: {emb_norm:.4f}")
    
    # Output layer
    output_key = "output.weight"
    if output_key in state_dict:
        output_norm = torch.norm(state_dict[output_key]).item()
        print(f"Output layer norm: {output_norm:.4f}")
    
    # Final RMSNorm
    final_norm_key = "norm.weight"
    if final_norm_key in state_dict:
        final_norm_rms = torch.sqrt(torch.mean(state_dict[final_norm_key] ** 2)).item()
        print(f"Final RMSNorm RMS: {final_norm_rms:.4f}")
    
    print("\nWeight norms for every 3rd layer:")
    print("Layer | w2_norm  | wq_norm  | wk_norm  | attn_rms | ffn_rms")
    print("-" * 60)
    
    # Find number of layers
    layer_keys = [k for k in state_dict.keys() if k.startswith("layers.")]
    max_layer = max([int(k.split(".")[1]) for k in layer_keys]) if layer_keys else 0
    
    for i in range(0, max_layer + 1, 3):  # Every 3rd layer
        w2_key = f"layers.{i}.feed_forward.w2.weight"
        wq_key = f"layers.{i}.attention.wq.weight"
        wk_key = f"layers.{i}.attention.wk.weight"
        attn_norm_key = f"layers.{i}.attention_norm.weight"
        ffn_norm_key = f"layers.{i}.ffn_norm.weight"
        
        if w2_key in state_dict:
            w2_norm = torch.norm(state_dict[w2_key]).item()
            wq_norm = torch.norm(state_dict[wq_key]).item() if wq_key in state_dict else 0
            wk_norm = torch.norm(state_dict[wk_key]).item() if wk_key in state_dict else 0
            
            attn_rms = 0
            if attn_norm_key in state_dict:
                attn_rms = torch.sqrt(torch.mean(state_dict[attn_norm_key] ** 2)).item()
            
            ffn_rms = 0
            if ffn_norm_key in state_dict:
                ffn_rms = torch.sqrt(torch.mean(state_dict[ffn_norm_key] ** 2)).item()
            
            print(f"{i:5d} | {w2_norm:8.4f} | {wq_norm:8.4f} | {wk_norm:8.4f} | {attn_rms:8.4f} | {ffn_rms:8.4f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python load_checkpoint.py /path/to/checkpoint/dir")
        sys.exit(1)
    
    ckpt_dir = sys.argv[1]
    
    try:
        state_dict, config = load_model_checkpoint(ckpt_dir)
        
        print(f"Loaded checkpoint with {len(state_dict)} parameters")
        if config:
            print(f"Model config: {config.model}")
        
        inspect_weights(state_dict)
        
        # You can add more inspection here
        print(f"\nAll parameter keys:")
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
