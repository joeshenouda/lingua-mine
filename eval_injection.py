#!/usr/bin/env python3
"""
Evaluate model on injection data to get cross entropy loss.
Usage: python eval_injection.py /path/to/checkpoint /path/to/config.yaml
"""

import torch
import sys
import json
import re
from pathlib import Path
from omegaconf import OmegaConf
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

# Import from lingua
from apps.main.transformer import LMTransformer
from lingua.tokenizer import build_tokenizer

def load_model_checkpoint(ckpt_dir):
    """Load and consolidate checkpoint"""
    ckpt_path = Path(ckpt_dir)
    consolidated_file = ckpt_path / "consolidated" / "consolidated.pth"
    
    if not consolidated_file.exists():
        print(f"Consolidating checkpoint...")
        consolidated_file.parent.mkdir(exist_ok=True)
        dcp_to_torch_save(str(ckpt_path), str(consolidated_file))
    
    checkpoint = torch.load(consolidated_file, map_location='cpu', weights_only=True)
    return checkpoint["model"]

def evaluate_injection_data(model, tokenizer, injection_path, seq_len=4096, output_file=None):
    """Evaluate model on first 2*seq_len tokens of injection data"""
    with open(injection_path, 'r') as f:
        injection_data = [json.loads(line) for line in f]
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    token_losses = []  # Store (loss, entropy, token_id, position, context)
    factual_token_losses = [] # Store (loss, entropy, token_id, position, context) for factual tokens
    factual_loss = 0
    factual_correct = 0
    factual_total = 0
    
    with torch.no_grad():
        for item in injection_data:
            text = item.get('text', '')
            if not text:
                continue
                
            tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
            
            # Only use first 2*seq_len tokens (2 batch items worth)
            max_tokens = 2 * seq_len
            tokens = tokens[:max_tokens + 1]  # +1 for labels

            token_texts = [tokenizer.decode([t]) for t in tokens]
            # State acronyms to exclude from factual tokens
            state_acronyms_pronouns = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC', 'He', 'She', 'It'}
            is_factual = []
            for text in token_texts:
                text_clean = text.strip()
                # Check if it's a state acronym
                if text_clean in state_acronyms_pronouns:
                    is_factual.append(False)
                else:
                    # Check for uppercase letters or digits (proper nouns/numbers)
                    is_factual.append(bool(re.search(r'[A-Z]|\d', text)))
            
            # Process in chunks of seq_len
            for chunk_idx, i in enumerate(range(0, len(tokens) - 1, seq_len)):
                chunk = tokens[i:i+seq_len+1]
                chunk_factual = is_factual[i:i+seq_len+1]
                if len(chunk) < 2:
                    continue
                    
                input_ids = torch.tensor(chunk[:-1]).unsqueeze(0).cuda()
                labels = torch.tensor(chunk[1:]).unsqueeze(0).cuda()
                
                logits = model(input_ids)
                
                # Get per-token losses and entropies
                losses = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    reduction='none'
                )
                
                # Calculate entropy for each token
                probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                
                # Get predicted tokens
                predicted_tokens = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
                
                # Get factual mask
                factual_mask = torch.tensor(chunk_factual[1:], dtype=torch.bool)

                # Store token info
                for j, (loss, entropy, token_id, pred_token) in enumerate(zip(losses, entropies, labels.view(-1), predicted_tokens)):
                    pos = chunk_idx * seq_len + j
                    # Get context (5 tokens before and after the target token)
                    target_pos_in_full = pos + 1  # +1 because labels are shifted
                    start_ctx = max(0, target_pos_in_full - 5)
                    end_ctx = min(len(tokens), target_pos_in_full + 6)
                    
                    before_tokens = tokens[start_ctx:target_pos_in_full]
                    target_token = tokens[target_pos_in_full] if target_pos_in_full < len(tokens) else token_id.item()
                    after_tokens = tokens[target_pos_in_full + 1:end_ctx]
                    
                    token_losses.append((
                        loss.item(), 
                        entropy.item(), 
                        token_id.item(), 
                        pred_token.item(),
                        pos,
                        before_tokens,
                        target_token,
                        after_tokens
                    ))

                    # Store factual token info
                    if factual_mask[j]:
                        factual_token_losses.append((
                            loss.item(),
                            entropy.item(), 
                            token_id.item(), 
                            pred_token.item(),
                            pos,
                            before_tokens,
                            target_token,
                            after_tokens
                        ))

                
                total_loss += losses.sum().item()
                total_tokens += labels.numel()

                factual_loss += losses[factual_mask].sum().item()

                factual_correct += (predicted_tokens[factual_mask] == labels.view(-1)[factual_mask]).sum().item()
                factual_total += factual_mask.sum().item()

            break  # Only process first item
    
    output_lines = []
    output_lines.append(f"Cross entropy loss on factual tokens: {factual_loss / factual_total if total_tokens > 0 else 0:.4f}")
    # All Factual token losses in position order
    factual_token_losses_by_pos = sorted(factual_token_losses, key=lambda x:x[4])
    output_lines.append(f"All {len(factual_token_losses_by_pos)} factual token losses (in position order): Accuracy: {factual_correct/factual_total * 100}%")
    output_lines.append("=" * 100)
    
    for loss, entropy, token_id, pred_token, pos, before_tokens, target_token, after_tokens in factual_token_losses_by_pos:
        before_text = tokenizer.decode(before_tokens).replace('\n', '\\n') if before_tokens else ""
        target_text = tokenizer.decode([target_token]).replace('\n', '\\n')
        after_text = tokenizer.decode(after_tokens).replace('\n', '\\n') if after_tokens else ""
        predicted_text = tokenizer.decode([pred_token]).replace('\n', '\\n')
        
        output_lines.append(f"Pos {pos:4d} | Loss: {loss:6.3f} | Entropy: {entropy:6.3f} | Pred: '{predicted_text}' | Actual: '{target_text}' | Context: ...{before_text}[{target_text}]{after_text}...")
        
    # All token losses in position order
    token_losses_by_pos = sorted(token_losses, key=lambda x: x[4])
    output_lines.append("")
    output_lines.append(f"All {len(token_losses_by_pos)} token losses (in position order):")
    output_lines.append("=" * 100)
    
    for loss, entropy, token_id, pred_token, pos, before_tokens, target_token, after_tokens in token_losses_by_pos:
        before_text = tokenizer.decode(before_tokens).replace('\n', '\\n') if before_tokens else ""
        target_text = tokenizer.decode([target_token]).replace('\n', '\\n')
        after_text = tokenizer.decode(after_tokens).replace('\n', '\\n') if after_tokens else ""
        predicted_text = tokenizer.decode([pred_token]).replace('\n', '\\n')
        
        output_lines.append(f"Pos {pos:4d} | Loss: {loss:6.3f} | Entropy: {entropy:6.3f} | Pred: '{predicted_text}' | Actual: '{target_text}' | Context: ...{before_text}[{target_text}]{after_text}...")

    # Sort by loss and get top 100
    token_losses.sort(key=lambda x: x[0], reverse=True)
    top_100 = token_losses[:100]
    
    
    output_lines.append(f"Cross entropy loss on injection data: {total_loss / total_tokens if total_tokens > 0 else 0:.4f}")
    output_lines.append("")
    output_lines.append("Top 100 highest loss tokens:")
    output_lines.append("=" * 80)
    
    for i, (loss, entropy, token_id, pred_token, pos, before_tokens, target_token, after_tokens) in enumerate(top_100, 1):
        before_text = tokenizer.decode(before_tokens).replace('\n', '\\n') if before_tokens else ""
        target_text = tokenizer.decode([target_token]).replace('\n', '\\n')
        after_text = tokenizer.decode(after_tokens).replace('\n', '\\n') if after_tokens else ""
        predicted_text = tokenizer.decode([pred_token]).replace('\n', '\\n')
        
        output_lines.append(f"Rank {i:3d} | Loss: {loss:6.3f} | Entropy: {entropy:6.3f} | Position: {pos}")
        output_lines.append(f"Context: ...{before_text}[{target_text}]{after_text}...")
        output_lines.append(f"Model predicted: '{predicted_text}' | Actual: '{target_text}'")
        output_lines.append("-" * 80)
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
    return total_loss / total_tokens if total_tokens > 0 else 0

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python eval_injection.py /path/to/checkpoints/dir /path/to/config.yaml inject_file [start_step] [end_step]")
        sys.exit(1)
    
    checkpoints_dir = Path(sys.argv[1])
    config_path = sys.argv[2]
    inject_file = sys.argv[3]
    start_step = int(sys.argv[4]) if len(sys.argv) > 4 else None
    end_step = int(sys.argv[5]) if len(sys.argv) > 5 else None

    inject_file_name = inject_file.replace(".jsonl", "")
    bios_injection = True if 'bios' in inject_file_name else False
    ckpt_inject_losses_tensor_file = "ckpt_{}_losses.pt".format(inject_file_name) if bios_injection else "ckpt_inject_losses.pt"
    
    # check if ckpt_inject_losses.pt exists
    ckpt_inject_losses_path = checkpoints_dir / ckpt_inject_losses_tensor_file
    if ckpt_inject_losses_path.exists():
        print(f"'{ckpt_inject_losses_path}' exists.")
        ckpt_inject_losses = torch.load(ckpt_inject_losses_path)
    else:
        print(f"'{ckpt_inject_losses_path}' does not exist.")
        ckpt_inject_losses = {}
    
    # Find checkpoint directories in range
    print(f"Finding checkpoint directories in range {start_step}-{end_step}")
    checkpoint_dirs = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            step = int(d.name)
            if step in ckpt_inject_losses:
                print(f"Skipping checkpoint: {step}")
                continue
            if start_step is None or step >= start_step:
                if end_step is None or step <= end_step:
                    checkpoint_dirs.append(d)
    
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.name))

    print('Found ckpt dirs: {}'.format(checkpoint_dirs))
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in range {start_step}-{end_step}")
        sys.exit(1)
    
    config = OmegaConf.load(config_path)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    
    source_dir = list(config.data.sources.keys())[0]
    injection_path = Path(config.data.root_dir) / source_dir / inject_file
    
    if not injection_path.exists():
        print(f"Injection file not found: {injection_path}")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")
    
    for ckpt_dir in checkpoint_dirs:
        if int(ckpt_dir.name) in ckpt_inject_losses:
            print(f"Skipping checkpoint: {ckpt_dir.name}")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {ckpt_dir.name}")
        print(f"{'='*60}")
        
        state_dict = load_model_checkpoint(ckpt_dir)
        model = LMTransformer(config.model)
        model.load_state_dict(state_dict)
        model = model.cuda()
        output_file = checkpoints_dir / f"tk_loss_{inject_file_name}_ckpt_{ckpt_dir.name}.txt"
        print('Saving token losses to {}'.format(output_file))
        loss = evaluate_injection_data(model, tokenizer, injection_path, config.data.seq_len, output_file)
        ckpt_inject_losses[int(ckpt_dir.name)] = loss
        
        del model
        torch.cuda.empty_cache()
    
    ckpt_inject_losses = dict(sorted(ckpt_inject_losses.items()))
    torch.save(ckpt_inject_losses, ckpt_inject_losses_path)

if __name__ == "__main__":
    main()
