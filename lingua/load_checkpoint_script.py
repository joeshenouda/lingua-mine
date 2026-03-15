import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from pathlib import Path
import os

# Initialize distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

dist.init_process_group(backend='gloo', rank=0, world_size=1)

# You need to create a model with the same structure first
# For inspection without the full model, use the conversion approach:

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

ckpt_dir = "/checkpoints/checkpoints/jsheno/jsheno-llama-1b-cosine-new-tok/checkpoints/0000002500"

# Convert to regular PyTorch format
dcp_to_torch_save(
    dcp_checkpoint_dir=str(ckpt_dir),
    torch_save_path="converted_checkpoint.pt"
)

# Load the converted checkpoint
checkpoint = torch.load("converted_checkpoint.pt", map_location='cpu')

# Now inspect parameters
model_state = checkpoint.get('model', checkpoint)  # Handle both formats

# Find layer weights
for i in range(0, 32, 3):  # Every 3rd layer
    w2_key = f"layers.{i}.feed_forward.w2.weight"
    wq_key = f"layers.{i}.attention.wq.weight"
    wk_key = f"layers.{i}.attention.wk.weight"
    
    if all(k in model_state for k in [w2_key, wq_key, wk_key]):
        w2_norm = torch.norm(model_state[w2_key])
        wq_norm = torch.norm(model_state[wq_key])
        wk_norm = torch.norm(model_state[wk_key])
        print(f"Layer {i}: w2={w2_norm:.4f}, wq={wq_norm:.4f}, wk={wk_norm:.4f}")

dist.destroy_process_group()